import argparse
import re
import sys

import jax

from src.data import encode, decode
from src.pure_jax.train import N_LAYERS, D_MODEL, N_HEADS, D_FF, VOCAB_SIZE
from src.pure_jax.model import init_model
from src.pure_jax.checkpoint import load_checkpoint
from src.pure_jax.eval import generate

DEFAULT_N_DIGITS = 3
DEFAULT_CHECKPOINT_PATH = "checkpoints/model.msgpack"

PROBLEM_RE = re.compile(r"^\d+\+\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively query the trained addition model."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--n-digits",
        type=int,
        default=DEFAULT_N_DIGITS,
        help=f"Digits per operand the model was trained on (default: {DEFAULT_N_DIGITS})",
    )
    return parser.parse_args()


def format_prompt(problem: str, n_digits: int) -> str:
    """Pad a 'a+b' problem to the model's expected zero-padded width and add '='."""
    a_str, b_str = problem.split("+")
    return f"{int(a_str):0{n_digits}d}+{int(b_str):0{n_digits}d}="


def main() -> None:
    args = parse_args()
    n_digits = args.n_digits
    solution_len = n_digits + 1

    print(f"Loading checkpoint from {args.checkpoint} ...")
    template = init_model(
        jax.random.PRNGKey(0), N_LAYERS, D_MODEL, N_HEADS, D_FF, VOCAB_SIZE
    )
    try:
        params = load_checkpoint(template, args.checkpoint)
    except FileNotFoundError:
        print(f"No checkpoint found at {args.checkpoint}. Train a model first.")
        sys.exit(1)

    print(f"Ready. Enter addition problems as 'a+b' (up to {n_digits} digits each).")
    print("Type 'quit' or press Ctrl+D to exit.\n")

    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            print()
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit"):
            break
        if not PROBLEM_RE.match(raw):
            print("Please enter a problem in the form 'a+b', e.g. '123+456'.")
            continue

        a_str, b_str = raw.split("+")
        if len(a_str) > n_digits or len(b_str) > n_digits:
            print(f"Each operand must have at most {n_digits} digits.")
            continue

        prompt = format_prompt(raw, n_digits)
        tokens = encode(prompt)
        generated = generate(params, tokens, N_HEADS, solution_len)
        predicted = decode(generated[-solution_len:].tolist()[::-1])
        predicted_value = int(predicted)

        expected = int(a_str) + int(b_str)
        marker = "" if predicted_value == expected else f"  (expected {expected})"
        print(f"{int(a_str)} + {int(b_str)} = {predicted_value}{marker}")


if __name__ == "__main__":
    main()