import json
from langchain_core.prompts import ChatPromptTemplate

from lib import prompts

# BASELINE -----------------------------------------------------------------------

_BASE_SYSTEM_INSTRUCTIONS = """You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. Your task is to compare a given Patient Note and a Clinical Trial Description's Inclusion Criteria to determine the patient's eligibility at the criterion level.

The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.

Assess the given patient's eligibility for a clinical trial by evaluating each and every criterion individually.

### INCLUSION CRITERIA ASSESSMENT
For each inclusion criterion, classify it as one of:
- **Met:** The patient's data explicitly and unequivocally satisfies the criterion.
- **Not Met:** The patient's data explicitly and unequivocally contradicts or fails to satisfy the criterion.
- **Unclear:** Insufficient or missing patient data to verify.
- **Irrelevant:** The criterion does not apply to the patient's context.

### IMPORTANT INSTRUCTIONS
- Each criteria is separated by a new line.
- Ensure all criteria are assessed one-by-one.
- The number of input criteria should match the number of output criteria evaluations.
- Use **only** the provided patient data; **do not infer, assume, or extrapolate beyond the given information.** Justifications must be strictly based on direct evidence from the patient profile."""

# TODO: can make this a pydantic class or something?
_RESPONSE_FORMAT_OBJECT = {
    "inclusion_criteria_evaluation": [
        {
            "criterion": "Exact inclusion criterion text",
            "classification": "Met | Not Met | Unclear | Irrelevant",
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
