import jax
import jax.numpy as jnp

from src.data import decode, generate_data
from src.pure_jax.checkpoint import load_checkpoint
from src.pure_jax.model import forward, init_model
from src.pure_jax.train import D_FF, D_MODEL, N_HEADS, N_LAYERS, VOCAB_SIZE

N_DIGITS = 3
SOLUTION_LEN = N_DIGITS + 1
NUM_EVAL_SAMPLES = 200
CHECKPOINT_PATH = "checkpoints/model.msgpack"


def generate(
    params: dict, prompt_tokens: jax.Array, n_heads: int, n_tokens: int
) -> jax.Array:
    """Greedily generate n_tokens continuation tokens after prompt_tokens."""
    tokens = prompt_tokens
    for _ in range(n_tokens):
        logits = forward(params, tokens, n_heads)
        idx = jnp.argmax(logits[-1], axis=-1)
        tokens = jnp.append(tokens, idx)
    return tokens


def evaluate(
    params: dict,
    key: jax.Array,
    num_samples: int = NUM_EVAL_SAMPLES,
    n_digits: int = N_DIGITS,
    verbose_failures: int = 5,
) -> float:
    """Check exact-match accuracy on fresh, randomly generated addition problems."""
    x, y = generate_data(key, n_digits=n_digits, batch_size=num_samples)

    correct = 0
    shown = 0
    for i in range(num_samples):
        generated = generate(params, x[i], N_HEADS, SOLUTION_LEN)
        predicted = generated[-SOLUTION_LEN:]
        target = y[i]

        if jnp.array_equal(predicted, target):
            correct += 1
        elif shown < verbose_failures:
            problem = decode(x[i].tolist())
            true_answer = decode(target.tolist()[::-1])
            predicted_answer = decode(predicted.tolist()[::-1])
            print(f"  {problem} true={true_answer} predicted={predicted_answer}")
            shown += 1

    accuracy = correct / num_samples
    print(f"Exact-match accuracy on {num_samples} held-out problems: {accuracy:.2%}")
    return accuracy


if __name__ == "__main__":
    key = jax.random.PRNGKey(123)
    template = init_model(
        jax.random.PRNGKey(0), N_LAYERS, D_MODEL, N_HEADS, D_FF, VOCAB_SIZE
    )
    params = load_checkpoint(template, CHECKPOINT_PATH)
    evaluate(params, key)
