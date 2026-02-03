import mellea

from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.requirements import req,check,simple_validate

requirements = [
    req("The email should have a salutation"),
    req("Use only lower-case letters", validation_fn=simple_validate(lambda s: s.islower())),
    check("The email should not mention the word 'regards'")
        
]

def write_email(m:mellea.MelleaSession, names :str , notes:str)->str:
    email_candidate = m.instruct(
        "Write an email to {{name}} using the notes following: {{notes}}.",
        requirements=requirements,
        strategy = RejectionSamplingStrategy(loop_budget = 5),
        user_variables={"name": names, "notes": notes},
        return_sampling_results = True ,
    )
    if email_candidate.success :
        return str(email_candidate.result)
    else :
        print("Expect sub-par output due to failure to meet requirements.")
        return email_candidate.sampling_results[0].value
    
m=mellea.start_session()
print(write_email(m,"Arush","Arush has been a great team player, always willing to help others and contribute to group projects."))



