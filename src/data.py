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
    " ": 12,
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
    12: " ",
}


def encode(s: str) -> list[int]:
    """Convert s to a list of token ids"""
    return [VOCAB_TO_ID[c] for c in s]


def decode(nums: list[int]) -> str:
    """Convert token ids back to string and strip padding"""
    return "".join([ID_TO_VOCAB[num] for num in nums]).strip()


def generate_data(
    key: jax.Array, n_digits: int = 3, batch_size: int = 16
) -> tuple[jax.Array, jax.Array]:
    _, k1, k2 = jax.random.split(key, 3)
    a = jax.random.randint(k1, shape=(batch_size, 1), minval=100, maxval=1000)
    b = jax.random.randint(k2, shape=(batch_size, 1), minval=100, maxval=1000)
    c = a + b
    return jnp.concatenate([a, b], axis=1), c
