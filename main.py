from dotenv import load_dotenv

from langchain_core.globals import set_llm_cache, set_debug
from langchain_core.runnables import RunnableParallel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.cache import SQLiteCache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from lib import prompts
from lib import exc_only_prompts
from lib import inc_only_prompts
from lib.eval import (
    run_experiment,
    compute_metrics,
    print_metrics,
    print_trialgpt_metrics,
    print_overview_metrics,
    apply_self_consistency,
    summarize_ties,
)
from lib.data_loader import GroundTruthsData, PatientsData, TrialsData
from lib.args import base_argparser

load_dotenv()


def setup_langchain_env(args):
    set_debug(args.debug)

    if not args.no_cache:
        cache = SQLiteCache(database_path=".langchain.db")
        set_llm_cache(cache)


if __name__ == "__main__":
    parser = base_argparser()

    # TODO: now this is in all of them, so you can move this to the base_argparser()
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

    llm_temp05 = None

    if args.llm == "gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            rate_limiter=rate_limiter,
        )
        llm_temp05 = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.5,
            rate_limiter=rate_limiter,
        )
    elif args.llm == "openai":
        # LangChain automatically looks for 'OPENAI_API_KEY' if not explicitly passed.
        llm = ChatOpenAI(
            model="gpt-5-nano",  # gpt-5-nano is the cheapest - maybe gpt-5-mini is best for cheap, but hopefully i see results with just this!
            temperature=0,
            rate_limiter=rate_limiter,
        )
        llm_temp05 = ChatOpenAI(
            model="gpt-5-nano", temperature=0.5, rate_limiter=rate_limiter
        )
    elif args.llm == "ollama":
        print("ignoring ratelimiter since it's a local model")
        llm = ChatOllama(
            model="hf.co/microsoft/phi-4-gguf:Q3_K_S",
            temperature=0,
        )
        llm_temp05 = ChatOllama(
            model="hf.co/microsoft/phi-4-gguf:Q3_K_S",
            temperature=0.5,
        )
    # elif args.llm.startswith("hf.co/"):
    #     # llm = "hf.co/microsoft/phi-4-gguf:Q3_K_S"
    #     print("ignoring ratelimiter since it's a local model")
    #     llm = ChatOllama(
    #         model=args.llm,
    #         temperature=0,
    #     )
    #     llm_temp05 = ChatOllama(model=args.llm, temperature=0.5)

    if args.criteria_type == "both":
        experiments = [
            (
                f"zero_shot__{args.llm}",
                (
                    prompts.PROMPT_TEMPLATE_ZERO_SHOT
                    | llm
                    | RunnableParallel(
                        {
                            "full_reasoning": lambda x: "N/A",
                            "structured_data": JsonOutputParser(),
                        }
                    )
                ),
                {},
            ),
            (
                f"cot__{args.llm}",
                (
                    prompts.PROMPT_TEMPLATE_ZERO_SHOT_COT
                    | llm
                    | RunnableParallel(
                        {
                            "full_reasoning": StrOutputParser(),
                            "structured_data": JsonOutputParser(),
                        }
                    )
                ),
                {},
            ),
        ]
    elif args.criteria_type == "inclusion":
        experiments = [
            (
                f"zero_shot__{args.llm}__inc_only",
                (
                    inc_only_prompts.PROMPT_TEMPLATE_ZERO_SHOT
                    | llm
                    | RunnableParallel(
                        {
                            "full_reasoning": lambda x: "N/A",
                            "structured_data": JsonOutputParser(),
                        }
                    )
                ),
                {"inc_exc": "inclusion"},
            ),
            (
                f"cot__{args.llm}__inc_only",
                (
                    inc_only_prompts.PROMPT_TEMPLATE_ZERO_SHOT_COT
                    | llm
                    | RunnableParallel(
                        {
                            "full_reasoning": StrOutputParser(),
                            "structured_data": JsonOutputParser(),
                        }
                    )
                ),
                {"inc_exc": "inclusion"},
            ),
        ]
    elif args.criteria_type == "exclusion":
        experiments = [
            (
                f"zero_shot__{args.llm}__exc_only",
                (
                    exc_only_prompts.PROMPT_TEMPLATE_ZERO_SHOT
                    | llm
                    | RunnableParallel(
                        {
                            "full_reasoning": lambda x: "N/A",
                            "structured_data": JsonOutputParser(),
                        }
                    )
                ),
                {"inc_exc": "exclusion"},
            ),
            (
                f"cot__{args.llm}__exc_only",
                (
                    exc_only_prompts.PROMPT_TEMPLATE_ZERO_SHOT_COT
                    | llm
                    | RunnableParallel(
                        {
                            "full_reasoning": StrOutputParser(),
                            "structured_data": JsonOutputParser(),
                        }
                    )
                ),
                {"inc_exc": "exclusion"},
            ),
        ]

    # Run core experiments
    all_metrics = {}
    for name, chain, experiment_opts in experiments:
        rows = run_experiment(
            chain=chain,
            experiment_name=name,
            gt_data=gt_data,
            pt_data=pt_data,
            tr_data=tr_data,
            **experiment_opts,
        )

        all_metrics[name] = rows

    # Run self-consistency experiments
    if not args.no_cache:
        print(
            "WARNING: skipping the self-consistency ones because if you don't use --no-cache, all results will be the same"
        )
    else:
        kind = args.criteria_type
        self_consistency_prefix = f"cot__{args.llm}__temp05"
        if kind != "both":
            self_consistency_prefix += f"__{kind}"

        if kind == "both":
            prompt = prompts.PROMPT_TEMPLATE_ZERO_SHOT_COT
        elif kind == "inclusion":
            prompt = inc_only_prompts.PROMPT_TEMPLATE_ZERO_SHOT_COT
        elif kind == "exclusion":
            prompt = exc_only_prompts.PROMPT_TEMPLATE_ZERO_SHOT_COT

        self_consistency_experiments = [
            (
                f"{self_consistency_prefix}__{i}",
                (
                    prompt
                    | llm_temp05
                    | RunnableParallel(
                        {
                            "full_reasoning": StrOutputParser(),
                            "structured_data": JsonOutputParser(),
                        }
                    )
                ),
                {"inc_exc": kind},
            )
            for i in range(3)
        ]

        self_consistency_metrics = {}
        for name, chain, experiment_opts in self_consistency_experiments:
            rows = run_experiment(
                chain=chain,
                experiment_name=name,
                gt_data=gt_data,
                pt_data=pt_data,
                tr_data=tr_data,
                **experiment_opts,
            )
            self_consistency_metrics[name] = rows
            # don't bother printing out the detailed ones
            # print_metrics(compute_metrics(rows), experiment_name=name)

        sc_rows, tie_log, dropped_log = apply_self_consistency(
            f"{self_consistency_prefix}__self_consistency", self_consistency_metrics
        )

        # add the aggregate to the overall metric
        all_metrics[f"{self_consistency_prefix}__self_consistency"] = sc_rows

    # Print everything
    for name, rows in all_metrics.items():
        print_metrics(compute_metrics(rows), experiment_name=name)

    print_trialgpt_metrics(gt_data)

    print_overview_metrics(all_metrics, gt_data)
