import json

from dotenv import load_dotenv

from pydantic import BaseModel

from langchain_core.globals import set_llm_cache, set_debug
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.rate_limiters import InMemoryRateLimiter

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


class CriterionEvaluation(BaseModel):
    criterion: str
    classification: str
    justification: str


# TODO: consolidate with the object that defined this and use this instead
class EligibilityReport(BaseModel):
    inclusion_criteria_evaluation: list[CriterionEvaluation]
    exclusion_criteria_evaluation: list[CriterionEvaluation]


class EligibilityState(MessagesState):
    clinical_trial: str
    patient_note: str


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
        # LangChain automatically looks for 'OPENAI_API_KEY' if not explicitly passed.
        llm = ChatOpenAI(
            model="gpt-5-nano",  # TODO: change this to something else, but gpt-5-nano is the cheapest for prototyping now
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

    if args.criteria_type == "both":
        generator_system_instructions = prompts._BASE_SYSTEM_INSTRUCTIONS
        response_format = prompts._RESPONSE_FORMAT_OBJECT
    elif args.criteria_type == "inclusion":
        generator_system_instructions = inc_only_prompts._BASE_SYSTEM_INSTRUCTIONS
        response_format = inc_only_prompts._RESPONSE_FORMAT_OBJECT
    elif args.criteria_type == "exclusion":
        generator_system_instructions = exc_only_prompts._BASE_SYSTEM_INSTRUCTIONS
        response_format = exc_only_prompts._RESPONSE_FORMAT_OBJECT

    PROMPT_TEMPLATE_GENERATOR = ChatPromptTemplate.from_messages(
        [
            ("system", generator_system_instructions),
            ("user", prompts._USER_INSTRUCTIONS),
            MessagesPlaceholder(variable_name="messages"),
            (
                "assistant",
                "Remember to finally respond in the format {json_format_string}",
            ),
        ]
    ).partial(
        json_format_string=json.dumps(response_format, indent=2),
        chain_of_thought_instructions="",
    )

    INCLUSION_LABELS_PROMPT_SUBSTRING = (
        "Inclusion criteria are classified as: Met, Not Met, Unclear, or Irrelevant."
    )
    EXCLUSION_LABELS_PROMPT_SUBSTRING = "Exclusion criteria are classified as: Violated, Not Violated, Unclear, or Irrelevant."

    PROMPT_TEMPLATE_REFLECTOR = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a medical expert evaluating the quality of a Clinical Trial Eligibility Report — a classification task that determines a patient's eligibility at the criterion level.

The patient note is the only source of truth. Your critique must be actionable given only what is provided — do not recommend obtaining more information.

### CLASSIFICATION LABELS
{classification_labels}

### YOUR TASK
Review the report against the Clinical Trial Description and Patient Note:
- Verify that every criterion is assessed word-for-word, with no extras added.
- Verify the criterion count matches between the description and the report; call out any missing rows.
- Pay special attention to "Unclear" classifications — challenge them if the patient note contains relevant evidence.
- Do not comment on correctly classified criteria.
- Keep feedback short and concise.

Whenever you are given a Clinical Trial Eligibility Report, evaluate it against this Clinical Trial Description and Patient Note:

---Start of Clinical Trial Description---
{clinical_trial}
---End of Clinical Trial Description---

---Start of Patient Note---
{patient_note}

The patient will provide informed consent, and will comply with the trial protocol without any practical issues.
---End of Patient Note---

First, think step-by-step. Then provide your final recommendations.

End your message with either "Please Revise" if you have recommendations or "No Revisions Necessary" if you have no recommendations.""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "assistant",
                "Remember: do not recommend obtaining more information. The patient note and clinical trial description are fixed. I will end my message with either 'Please Revise' or 'No Revisions Necessary' on its own line.",
            ),
        ]
    )

    generator_chain = PROMPT_TEMPLATE_GENERATOR | llm.with_structured_output(
        EligibilityReport
    )

    if args.criteria_type == "both":
        classification_label_str = (
            f"{INCLUSION_LABELS_PROMPT_SUBSTRING}\n{EXCLUSION_LABELS_PROMPT_SUBSTRING}"
        )
    elif args.criteria_type == "exclusion":
        classification_label_str = EXCLUSION_LABELS_PROMPT_SUBSTRING
    elif args.criteria_type == "inclusion":
        classification_label_str = INCLUSION_LABELS_PROMPT_SUBSTRING
    else:
        raise Exception("criteria type not implemented")

    reflector_chain = (
        PROMPT_TEMPLATE_REFLECTOR.partial(
            classification_labels=classification_label_str
        )
        | llm
        | StrOutputParser()
    )

    def generation_node(state: EligibilityState) -> dict:
        output = generator_chain.invoke(
            {
                "messages": state["messages"],
                "clinical_trial": state["clinical_trial"],
                "patient_note": state["patient_note"],
            }
        )

        # This will append to the messages state
        return {"messages": [AIMessage(content=output.model_dump_json())]}

    def reflection_node(state: EligibilityState) -> dict:
        output = reflector_chain.invoke(
            {
                "messages": state["messages"],
                "clinical_trial": state["clinical_trial"],
                "patient_note": state["patient_note"],
            }
        )

        # Wrap as HumanMessage so the generator sees it as "user feedback"
        # This will append to the messages state
        return {"messages": [HumanMessage(content=output)]}

    builder = StateGraph(EligibilityState)

    GENERATOR_NODE_NAME = "generate"
    REFLECTOR_NODE_NAME = "reflect"

    builder.add_node(GENERATOR_NODE_NAME, generation_node)
    builder.add_node(REFLECTOR_NODE_NAME, reflection_node)
    builder.set_entry_point(GENERATOR_NODE_NAME)

    def should_continue_after_reflector(state: EligibilityState):
        last_message = state["messages"][-1]
        if "No Revisions Necessary" in last_message.content:
            return END
        if len(state["messages"]) >= 6:
            return END
        return GENERATOR_NODE_NAME

    builder.add_edge(GENERATOR_NODE_NAME, REFLECTOR_NODE_NAME)
    builder.add_conditional_edges(
        REFLECTOR_NODE_NAME,
        should_continue_after_reflector,
        {
            # Return Value : Target Node/Constant
            END: END,
            GENERATOR_NODE_NAME: GENERATOR_NODE_NAME,
        },
    )

    graph = builder.compile()

    # import sys

    # with open("reflection.png", "wb") as f:
    #     f.write(graph.get_graph().draw_mermaid_png())
    # sys.exit()

    chain = (
        graph
        | RunnableLambda(
            lambda state: EligibilityReport.model_validate_json(
                state["messages"][-2].content
            )
        )
        # Just to match the format of the other ones - the reasoning is encoded in the back and forth
        | RunnableParallel(
            {
                "full_reasoning": lambda x: "N/A",
                "structured_data": lambda x: x.model_dump(),
            }
        )
    )

    # patient_id, trial_id = next(iter(gt_data.get_patient_trial_pairs()))
    # output = chain.invoke(
    #     {
    #         "clinical_trial": tr_data.get_formatted_trial(trial_id),
    #         "patient_note": pt_data.get(patient_id),
    #     },
    # )

    # print(output)

    experiment_name = f"reflection__{args.llm}"
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

    all_metrics = {"reflection": rows}
    print_overview_metrics(all_metrics, gt_data)
