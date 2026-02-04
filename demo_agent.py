
import datetime
import inspect
import json
from collections.abc import Callable
from typing import Literal

import pydantic
from jinja2 import Template

import mellea
import mellea.stdlib.components.chat
from mellea.core import FancyLogger
from mellea.stdlib.context import ChatContext

# Turn off the logger
FancyLogger.get_logger().setLevel("ERROR")

react_system_template: Template = Template(
    """Answer the user's question as best you can.

Today is {{- today }} and you can use the following tool names with associated descriptions:
{% for tool in tools %} * {{- tool.get_name() }}: {{- tool.get_description()}}{% endfor %}"""
)

#wrapper over apydantic fuction that stores tools and metadata
# validation in nice JSON schema behaviour
class ReactTool(pydantic.BaseModel):
    """This is a helper model for React tools.

    Args:
        fn: The tool.
        name: The name of the tool. The default value is the function's name.
        description: A description of the tool for the ReACT system prompt. The default value is the first line of the function's docstring.
    """

    fn: Callable
    name: str | None
    description: str | None

    def get_name(self):
        if self.name is None:
            return self.fn.__name__
        else:
            return self.name

    def get_description(self):
        if self.description is None:
            return self.fn.__doc__.splitlines()[0]
        else:
            return self.description

    def args_schema(self):
        sig = inspect.signature(self.fn)
        fields = dict()
        for param_name, param in sig.parameters.items():
            fields[param_name] = (str , ...)
        return pydantic.create_model(
            f"{self.fn.__name__.capitalize()}ToolSchema", **fields
        )
        # What this does:

        # inspect.signature(self.fn) reads the tool function signature (parameter names).
        # Creates a Pydantic model dynamically with those parameter names.
        # So if your tool is:
        # def weather_lookup_fn(zip_code: str):
        # then args_schema() builds a schema like:
        # {"zip_code": <string>}
        # That’s why the model produced JSON with a zip_code key.


class ReactToolbox(pydantic.BaseModel):
    """A convienance wrapper around ReactTool."""

    tools: list[ReactTool]
    #Holds the names of the Tools in a list

    def tool_names(self):
        return [tool.get_name() for tool in self.tools]
    #Returns the names of the Tools in a list

    def tools_dict(self):
        """Formats the tools for passing into backends' tools= parameter."""
        return {tool.get_name(): tool.fn for tool in self.tools}
    #Returns the tools in a dictionary format mapping tool name → Python function.

    def get_tool_from_name(self, name: str) -> ReactTool | None:
        for tool in self.tools:
            if tool.get_name() == name:
                return tool
        return None
    #This format is commonly used for tool passing

    def call_tool(self, tool: ReactTool, kwargs_json: str):
        fn = tool.fn
        kwargs = json.loads(kwargs_json)
        return fn(**kwargs)
    # Finds a tool object by name.
    # Takes JSON string like {"zip_code": "03285"}
    # Converts to a dict
    # Calls the Python function: fn(zip_code="03285")
    # Returns its output (your weather string)

    def tool_name_schema(self):
        names = self.tool_names()
        fields = dict()
        fields["tool"] = Literal[*names]
        return pydantic.create_model("ToolSelectionSchema", **fields)
    # Builds a schema where "tool" must be one of the known tool names.

    def get_tool_from_schema(self, content: str):
        schema = self.tool_name_schema()
        validated = schema.model_validate_json(content)
        return self.get_tool_from_name(validated.tool)
    # Parses the LLM’s tool-selection JSON
    # Validates it against the schema
    # Returns the actual tool object


class IsDoneModel(pydantic.BaseModel):
    is_done: bool


def react(
    m: mellea.MelleaSession,
    goal: str,
    state_description: str | None,
    react_toolbox: ReactToolbox,
):
    """
    This is the main ReACT loop.
    m: session (your LLM connection + context)
    goal: the user’s question
    state_description: extra info about world state (you don’t use it yet)
    react_toolbox: tools the agent can use

    """
    # assert m.ctx.is_chat_context, "ReACT requires a chat context."
    test_ctx_lin = m.ctx.view_for_generation()
    assert test_ctx_lin is not None and len(test_ctx_lin) == 0, ("ReACT expects a fresh context.")

    # Construct the system prompt for ReACT.
    _sys_prompt = react_system_template.render(
        {"today": datetime.date.today(), "tools": react_toolbox.tools}
    )
    """Produces a string system prompt containing:
    -today’s date
    -tool names and descriptions"""

    # Add the system prompt and user's goal question to the chat history.
    m.ctx = m.ctx.add(
        mellea.stdlib.components.chat.Message(role="system", content=_sys_prompt)
    ).add(mellea.stdlib.components.chat.Message(role="user", content=f"{goal}"))
    

    # The main ReACT loop as a dynamic program:
    # ReACT agent loop from the Mellea docs: alternate Thought → Action(tool) → Args → Observation → Done-check, and loop until the model says it’s done
    # (  ?(not done) ;
    #    (thought request ; thought response) ;
    #    (action request ; action response) ;
    #    (action args request ; action args response) ;
    #    observation from the tool call ;
    #    (is done request ; is done response) ;
    #    { ?(model indicated done) ; emit_final_answer ; done := true }
    # )*
    done = False
    turn_num = 0
    while not done:
        turn_num += 1
        print(f"## ReACT TURN NUMBER {turn_num}")

        print("### Thought")
        thought = m.chat(
            "What should you do next? Respond with a description of the next piece of information you need or the next action you need to take."
        )
        print(thought.content)

        print("### Action")
        act = m.chat(
            "Choose your next action. Respond with a nothing other than a tool name.",
            # model_options={mellea.backends.types.ModelOption.TOOLS: react_toolbox.tools_dict()},
            format=react_toolbox.tool_name_schema(),
        )
        selected_tool: ReactTool = react_toolbox.get_tool_from_schema(act.content)
        print(selected_tool.get_name())
        # format=... forces the model to return JSON matching the schema.
        # Even though your prompt says “only a tool name”, the format actually pushes it to return structured JSON like:
        # {"tool": "Get the weather"}

        print("### Arguments for action")
        act_args = m.chat(
            "Choose arguments for the tool. Respond using JSON and include only the tool arguments in your response.",
            format=selected_tool.args_schema(),
        )
        print(f"```json\n{json.dumps(json.loads(act_args.content), indent=2)}\n```")

        print("### Observation")
        tool_output = react_toolbox.call_tool(selected_tool, act_args.content)
        m.ctx = m.ctx.add(
            mellea.stdlib.components.chat.Message(role="tool", content=tool_output)
        )
        print(tool_output)
        print("### Done Check")
        
        is_done = IsDoneModel.model_validate_json(
            m.chat(
                f"Do you have the FINAL ANSWER to the user's original query ({goal})? "
                "Respond with Yes ONLY if you have the specific information requested (e.g., the actual temperature). "
                "If you still need to run a tool, respond with No.",
                format=IsDoneModel,
            ).content
        ).is_done
        if is_done:
            print("Done. Will summarize and return output now.")
            done = True
            return m.chat(
                f"Please provide your final answer to the original query ({goal})."
            ).content
        else:
            print("Not done.")
            done = False


if __name__ == "__main__":
    m = mellea.start_session(ctx=ChatContext())

    def zip_lookup_tool_fn(city: str):
        """Returns the ZIP code for the `city`."""
        return "03285"

    zip_lookup_tool = ReactTool(
        name="Zip Code Lookup",
        fn=zip_lookup_tool_fn,
        description="Returns the ZIP code given a town name.",
    )

    def weather_lookup_fn(zip_code: str):
        """Looks up the weather for a town given a five-digit `zip_code`."""
        return "The weather in Thornton, NH is sunny with a high of 78 and a low of 52. Scattered showers are possible in the afternoon."

    weather_lookup_tool = ReactTool(
        name="Get the weather",
        fn=weather_lookup_fn,
        description="Returns the weather given a ZIP code.",
    )

    result = react(
        m,
        goal="What is today's high temperature in Thornton, NH?",
        state_description=None,
        react_toolbox=ReactToolbox(tools=[zip_lookup_tool, weather_lookup_tool]),
    )

    print("## Final Answer")
    print(result)

    #The agent looked up the zip code (round 1) and used the zip code to call the weather tool (round 2).
