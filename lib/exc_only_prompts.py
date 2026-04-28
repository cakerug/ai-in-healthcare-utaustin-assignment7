import json
from langchain_core.prompts import ChatPromptTemplate

from lib import prompts

# BASELINE -----------------------------------------------------------------------

_BASE_SYSTEM_INSTRUCTIONS = """You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Your task is to compare a given Patient Note and a Clinical Trial Description's Exclusion Criteria to determine the patient's eligibility at the criterion level.

The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.

Assess the given patient's eligibility for a clinical trial by evaluating each and every criterion individually.

### EXCLUSION CRITERIA ASSESSMENT
For each exclusion criterion, classify it as one of:
- **Violated:** The patient's data explicitly and unequivocally violates the criterion.
- **Not Violated:** The patient's data confirms compliance with the criterion.
- **Unclear:** Insufficient or missing patient data to verify.
- **Irrelevant:** The criterion does not apply to the patient's context.

### IMPORTANT INSTRUCTIONS
- Each criteria is separated by a new line.
- Ensure all criteria are assessed one-by-one.
- The number of input criteria should match the number of output criteria evaluations.
- Use **only** the provided patient data; **do not infer, assume, or extrapolate beyond the given information.** Justifications must be strictly based on direct evidence from the patient profile."""

# TODO: can make this a pydantic class or something?
_RESPONSE_FORMAT_OBJECT = {
    "exclusion_criteria_evaluation": [
        {
            "criterion": "Exact exclusion criterion text",
            "classification": "Violated | Not Violated | Unclear | Irrelevant",
            "justification": "Clear, evidence-based rationale using ONLY provided data",
        }
    ],
}

_PROMPT_TEMPLATE_BASE = ChatPromptTemplate.from_messages(
    [("system", _BASE_SYSTEM_INSTRUCTIONS), ("user", prompts._USER_INSTRUCTIONS)]
).partial(json_format_string=json.dumps(_RESPONSE_FORMAT_OBJECT, indent=2))

PROMPT_TEMPLATE_ZERO_SHOT = _PROMPT_TEMPLATE_BASE.partial(
    chain_of_thought_instructions=""
)

PROMPT_TEMPLATE_ZERO_SHOT_COT = _PROMPT_TEMPLATE_BASE.partial(
    chain_of_thought_instructions="\nFirst, let's think step-by-step to analyze the criteria.\n"
)
