import json

from dotenv import load_dotenv

from langchain_core.globals import set_llm_cache, set_debug
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.callbacks import UsageMetadataCallbackHandler

from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState

from langchain_community.cache import SQLiteCache

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from lib import exc_only_prompts, inc_only_prompts, prompts
from lib.eval import (
    run_experiment,
    compute_metrics,
    print_metrics,
    print_overview_metrics,
)
from lib.data_loader import GroundTruthsData, PatientsData, TrialsData
from lib.args import base_argparser

load_dotenv()


def setup_langchain_env(args):
    set_debug(args.debug)
    cache = SQLiteCache(database_path=".langchain.db")
    set_llm_cache(cache)


# TODO: does this even need to extend MessageState or can it just be TypedDict
class EligibilityState(MessagesState):
    clinical_trial: str
    patient_note: str
    # Recruiter output: list of {"role": str, "rationale": str}
    specialist_roles: list[dict]
    # One structured report per specialist (same schema as _RESPONSE_FORMAT_OBJECT)
    specialist_reports: list[dict]
    # Which specialist index to run next (0-based)
    specialist_index: int


# Double {s to prevent it thinking it's a template variable
_RECRUITER_SYSTEM = """You are a clinical trial coordinator. Given a clinical trial description and a patient note, identify exactly 3 medical specialists whose domain expertise is most relevant to evaluating the eligibility criteria.

Return ONLY valid JSON in this exact format:
{{
  "specialists": [
    {{"role": "...", "rationale": "one sentence explaining relevance"}},
    {{"role": "...", "rationale": "..."}},
    {{"role": "...", "rationale": "..."}}
  ]
}}"""

_RECRUITER_USER = """---Start of Clinical Trial Description---
{clinical_trial}
---End of Clinical Trial Description---

---Start of Patient Note---
{patient_note}
---End of Patient Note---

Identify the 3 most relevant specialists."""

PROMPT_TEMPLATE_RECRUITER = ChatPromptTemplate.from_messages(
    [("system", _RECRUITER_SYSTEM), ("user", _RECRUITER_USER)]
)

# Same as the base prompt, just replaced the first line
_SPECIALIST_SYSTEM = """You are a {role} participating in a multidisciplinary clinical trial eligibility review.

Evaluate ALL eligibility criteria below from your domain's perspective. For criteria that fall clearly outside your specialty, you may still classify them — but note in the justification that this is outside your primary domain.

The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.

The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.

Assess the given patient's eligibility for a clinical trial by evaluating each and every criterion individually.

### INCLUSION CRITERIA ASSESSMENT
For each inclusion criterion, classify it as one of:
- **Met:** The patient's data explicitly and unequivocally satisfies the criterion.
- **Not Met:** The patient's data explicitly and unequivocally contradicts or fails to satisfy the criterion.
- **Unclear:** Insufficient or missing patient data to verify.
- **Irrelevant:** The criterion does not apply to the patient's context.

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

_SPECIALIST_SYSTEM_INC_ONLY = """You are a {role} participating in a multidisciplinary clinical trial eligibility review.

Evaluate ALL eligibility criteria below from your domain's perspective. For criteria that fall clearly outside your specialty, you may still classify them — but note in the justification that this is outside your primary domain.

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

_SPECIALIST_SYSTEM_EXC_ONLY = """You are a {role} participating in a multidisciplinary clinical trial eligibility review.

Evaluate ALL eligibility criteria below from your domain's perspective. For criteria that fall clearly outside your specialty, you may still classify them — but note in the justification that this is outside your primary domain.

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


# Synthesizer output format — extends _RESPONSE_FORMAT_OBJECT with provenance fields
_SYNTHESIS_RESPONSE_FORMAT_INCLUSION = {
    "inclusion_criteria_evaluation": [
        {
            "criterion": "Exact inclusion criterion text",
            "classification": "Met | Not Met | Unclear | Irrelevant",
            "justification": "Synthesized rationale citing specialist agreement/disagreement",
            "primary_reviewer": "Role of the specialist weighted most heavily",
            "dissent": "null, or description of any disagreement across specialists",
        }
    ],
}
_SYNTHESIS_RESPONSE_FORMAT_EXCLUSION = {
    "exclusion_criteria_evaluation": [
        {
            "criterion": "Exact exclusion criterion text",
            "classification": "Violated | Not Violated | Unclear | Irrelevant",
            "justification": "Synthesized rationale citing specialist agreement/disagreement",
            "primary_reviewer": "Role of the specialist weighted most heavily",
            "dissent": "null, or description of any disagreement across specialists",
        }
    ],
}
_SYNTHESIS_RESPONSE_FORMAT = {
    **_SYNTHESIS_RESPONSE_FORMAT_INCLUSION,
    **_SYNTHESIS_RESPONSE_FORMAT_EXCLUSION,
}

INCLUSION_LABELS_PROMPT_SUBSTRING = (
    "Inclusion criteria are classified as: Met, Not Met, Unclear, or Irrelevant."
)
EXCLUSION_LABELS_PROMPT_SUBSTRING = "Exclusion criteria are classified as: Violated, Not Violated, Unclear, or Irrelevant."

_SYNTHESIZER_SYSTEM = """You are a senior clinical trial eligibility adjudicator. You will receive independent eligibility assessments from multiple medical specialists and must synthesize them into a single final report.

### YOUR TASK
For each criterion:
1. Review all specialist assessments for that criterion.
2. Weight each specialist's opinion by domain relevance (e.g., a nephrologist's view on eGFR matters more than a cardiologist's).
3. Choose the most defensible final classification.
4. Identify the primary reviewer (the specialist whose opinion you weighted most).
5. Note any meaningful disagreement in the "dissent" field; set to null if unanimous.

### CLASSIFICATION LABELS
{classification_labels}

### IMPORTANT
- Use ONLY the patient note and specialist reports provided.
- Do not infer beyond what is given.
- Return ONLY valid JSON — no markdown fences, no preamble."""

_SYNTHESIZER_USER = """---Start of Clinical Trial Description---
{clinical_trial}
---End of Clinical Trial Description---

---Start of Patient Note---
{patient_note}

The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
---End of Patient Note---

---Start of Specialist Reports---
{specialist_reports}
---End of Specialist Reports---

Synthesize the above into a final eligibility report in this exact JSON format:
{json_format_string}"""


def make_recruiter_node(llm):
    chain = PROMPT_TEMPLATE_RECRUITER | llm | JsonOutputParser()

    def recruiter_node(state: EligibilityState) -> dict:
        result = chain.invoke(
            {
                "clinical_trial": state["clinical_trial"],
                "patient_note": state["patient_note"],
            }
        )
        return {
            "specialist_roles": result["specialists"],
            "specialist_reports": [],
            "specialist_index": 0,
        }

    return recruiter_node


def make_specialist_node(llm, criteria_type):
    if criteria_type == "both":
        system_inst = _SPECIALIST_SYSTEM
        response_format = prompts._RESPONSE_FORMAT_OBJECT
    elif criteria_type == "inclusion":
        system_inst = _SPECIALIST_SYSTEM_INC_ONLY
        response_format = inc_only_prompts._RESPONSE_FORMAT_OBJECT
    elif criteria_type == "exclusion":
        system_inst = _SPECIALIST_SYSTEM_EXC_ONLY
        response_format = exc_only_prompts._RESPONSE_FORMAT_OBJECT

    PROMPT_TEMPLATE_SPECIALIST = ChatPromptTemplate.from_messages(
        [("system", system_inst), ("user", prompts._USER_INSTRUCTIONS)]
    ).partial(
        json_format_string=json.dumps(response_format, indent=2),
        chain_of_thought_instructions="\nFirst, let's think step-by-step to analyze the criteria.\n",
    )

    chain = PROMPT_TEMPLATE_SPECIALIST | llm | JsonOutputParser()

    def specialist_node(state: EligibilityState) -> dict:
        idx = state["specialist_index"]
        specialist = state["specialist_roles"][idx]
        role = specialist["role"]

        report = chain.invoke(
            {
                "role": role,
                "clinical_trial": state["clinical_trial"],
                "patient_note": state["patient_note"],
            }
        )

        # Annotate the report with the specialist's role for the synthesizer
        report["_specialist_role"] = role

        return {
            "specialist_reports": state["specialist_reports"] + [report],
            "specialist_index": idx + 1,
        }

    return specialist_node


def make_synthesizer_node(llm, criteria_type):
    if criteria_type == "both":
        classification_label_str = (
            f"{INCLUSION_LABELS_PROMPT_SUBSTRING}\n{EXCLUSION_LABELS_PROMPT_SUBSTRING}"
        )
        response_format = _SYNTHESIS_RESPONSE_FORMAT
    elif criteria_type == "exclusion":
        classification_label_str = EXCLUSION_LABELS_PROMPT_SUBSTRING
        response_format = _SYNTHESIS_RESPONSE_FORMAT_EXCLUSION
    elif criteria_type == "inclusion":
        classification_label_str = INCLUSION_LABELS_PROMPT_SUBSTRING
        response_format = _SYNTHESIS_RESPONSE_FORMAT_INCLUSION
    else:
        raise Exception("criteria type not implemented")

    PROMPT_TEMPLATE_SYNTHESIZER = ChatPromptTemplate.from_messages(
        [("system", _SYNTHESIZER_SYSTEM), ("user", _SYNTHESIZER_USER)]
    ).partial(
        json_format_string=json.dumps(response_format, indent=2),
        classification_labels=classification_label_str,
    )

    chain = PROMPT_TEMPLATE_SYNTHESIZER | llm | JsonOutputParser()

    def synthesizer_node(state: EligibilityState) -> dict:
        # Format specialist reports as readable JSON for the prompt
        reports_str = json.dumps(state["specialist_reports"], indent=2)

        result = chain.invoke(
            {
                "clinical_trial": state["clinical_trial"],
                "patient_note": state["patient_note"],
                "specialist_reports": reports_str,
            }
        )

        # Store final report as an AIMessage so run_experiment can extract it
        return {"messages": [AIMessage(content=json.dumps(result))]}

    return synthesizer_node


def should_continue_specialists(state: EligibilityState):
    """Route back to specialist_node until all roles are done, then synthesize."""
    if state["specialist_index"] < len(state["specialist_roles"]):
        return "specialist"
    return "synthesizer"


def build_specialist_graph(llm, criteria_type):
    builder = StateGraph(EligibilityState)

    builder.add_node("recruiter", make_recruiter_node(llm))
    builder.add_node("specialist", make_specialist_node(llm, criteria_type))
    builder.add_node("synthesizer", make_synthesizer_node(llm, criteria_type))

    builder.set_entry_point("recruiter")
    builder.add_edge("recruiter", "specialist")
    builder.add_conditional_edges(
        "specialist",
        should_continue_specialists,
        {"specialist": "specialist", "synthesizer": "synthesizer"},
    )
    builder.add_edge("synthesizer", END)

    return builder.compile()


if __name__ == "__main__":
    parser = base_argparser()

    parser.add_argument(
        "--criteria-type", choices=["both", "inclusion", "exclusion"], default="both"
    )

    args = parser.parse_args()

    setup_langchain_env(args)

    gt_data = GroundTruthsData()
    pt_data = PatientsData()
    tr_data = TrialsData()

    rate_limiter = (
        None
        if args.throttle == 0
        else InMemoryRateLimiter(requests_per_second=1 / args.throttle)
    )

    if args.llm == "gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            rate_limiter=rate_limiter,
        )
    elif args.llm == "openai":
        llm = ChatOpenAI(
            model="gpt-5-nano",
            temperature=0,
            rate_limiter=rate_limiter,
        )
    elif args.llm == "ollama":
        print("ignoring ratelimiter since it's a local model")
        llm = ChatOllama(
            model="hf.co/microsoft/phi-4-gguf:Q3_K_S",
            temperature=0,
        )
    elif args.llm.startswith("hf.co/"):
        # llm = "hf.co/microsoft/phi-4-gguf:Q3_K_S"
        print("ignoring ratelimiter since it's a local model")
        llm = ChatOllama(
            model=args.llm,
            temperature=0,
        )

    graph = build_specialist_graph(llm, args.criteria_type)

    # Extract the synthesizer's final JSON from the last AIMessage
    chain = graph | RunnableParallel(
        {
            "full_reasoning": RunnableLambda(
                lambda state: json.dumps(state["specialist_reports"], indent=2)
            ),
            "structured_data": (
                RunnableLambda(lambda state: state["messages"][-1].content)
                | JsonOutputParser()
            ),
        }
    )

    # patient_id, trial_id = next(iter(gt_data.get_patient_trial_pairs()))

    # callback = UsageMetadataCallbackHandler()
    # output = chain.invoke(
    #     {
    #         "clinical_trial": tr_data.get_formatted_trial(trial_id),
    #         "patient_note": pt_data.get(patient_id),
    #     },
    #     config={"callbacks": [callback]},
    # )

    # print(output)

    # structured = output.get("structured_data", {})
    # print(">>>>>>>>>>>>>>>>> structured_data", structured)

    experiment_name = f"specialists__{args.llm}"
    if args.criteria_type == "inclusion":
        experiment_name += "__inclusion"
    elif args.criteria_type == "exclusion":
        experiment_name += "__exclusion"

    rows = run_experiment(
        chain=chain,
        experiment_name=experiment_name,
        gt_data=gt_data,
        pt_data=pt_data,
        tr_data=tr_data,
        inc_exc=args.criteria_type,
    )

    print_metrics(compute_metrics(rows))

    all_metrics = {"specialist": rows}
    print_overview_metrics(all_metrics, gt_data)
