import jax
import jax.numpy as jnp

VOCAB_TO_ID = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "=": 11,
}

ID_TO_VOCAB = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "+",
    11: "=",
}


def encode(s: str) -> jax.Array:
    """Convert s to a list of token ids"""
    return jnp.array([VOCAB_TO_ID[c] for c in s])


def decode(nums: list[int]) -> str:
    """Convert token ids back to string"""
    return "".join([ID_TO_VOCAB[num] for num in nums])


def sample_problem(key: jax.Array, n_digits: int) -> tuple[int, int, int]:
    """Generate a random triple [a, b, c] where a + b = c"""
    assert n_digits >= 0, "n_digits must be greater than 0"
    min_val = 0
    max_val = 10 ** (n_digits)
    a, b = jax.random.randint(key, shape=(2,), minval=min_val, maxval=max_val)
    return a, b, a + b


def format_problem(a: int, b: int, n_digits: int) -> str:
    """Format the generated problem into a string and pad up to n_digits"""
    assert a < 10**n_digits
    assert b < 10**n_digits
    return f"{a:0{n_digits}d}+{b:0{n_digits}d}="


def format_solution(c: int, n_digits) -> str:
    """Format a solution by padding to n_digits, reversing to improve model training"""
    assert c < 10 ** (n_digits + 1)
    return f"{c:0{n_digits + 1}d}"[::-1]


def generate_data(
    key: jax.Array, n_digits: int = 3, batch_size: int = 16
) -> tuple[jax.Array, jax.Array]:
    """Generate batch_size sample problems"""
    keys = jax.random.split(key, batch_size)
    inputs = []
    outputs = []

    for k in keys:
        a, b, c = sample_problem(k, n_digits)
        inputs.append(encode(format_problem(a, b, n_digits)))
        outputs.append(encode(format_solution(c, n_digits)))

    return jnp.array(inputs), jnp.array(outputs)
