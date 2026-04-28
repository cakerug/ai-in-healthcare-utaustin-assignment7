import argparse


def base_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--throttle",
        type=float,
        default=4.0,
        help="Set the throttle seconds value as a float (default: 4.0)",
    )

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--llm", type=str, default="ollama")

    parser.add_argument("--no-cache", action="store_true")

    return parser
