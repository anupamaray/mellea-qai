# file: https://github.com/generative-computing/mellea/blob/main/docs/examples/tutorial/model_options_example.py#L1-L16
import mellea
from mellea.backends import ModelOption  #its mellea.backends
from mellea.backends.ollama import OllamaModelBackend
from mellea.backends import model_ids

m = mellea.MelleaSession(
    backend=OllamaModelBackend(
        model_id="granite4:micro",model_options={ModelOption.SEED: 42}
    )
)

answer = m.instruct(
    "What is 2x2?",
    model_options={
        "temperature": 0.1,
    },
)

print(str(answer))


# # default behavior
# m.instruct("A")  # uses base backend options

# # temporarily switch options
# m.push_model_state(model_options={ModelOption.TEMPERATURE: 0.0, "num_predict": 50})

# m.instruct("B")  # uses the pushed options
# m.instruct("C")  # also uses the pushed options

# # go back to previous options
# m.pop_model_state()

# m.instruct("D")  # back to original base options
