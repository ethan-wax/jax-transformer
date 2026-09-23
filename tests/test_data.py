import jax
import jax.numpy as jnp
from src.data import (
    decode,
    encode,
    format_problem,
    format_solution,
    generate_data,
    sample_problem,
)


def test_encode():
    s = "0123456789+="
    result = jnp.array([n for n in range(12)])
    assert (encode(s) == result).all()


def test_decode():
    nums = [n for n in range(12)]
    result = "0123456789+="
    assert decode(nums) == result


def test_encode_decode_inverse():
    s = "0123456789+="
    assert decode(encode(s).tolist()) == s


def test_sample_problem():
    key = jax.random.PRNGKey(42)
    a, b, c = sample_problem(key, 3)
    assert 0 <= a < 1000
    assert 0 <= b < 1000
    assert 0 <= c < 1998
    assert a + b == c


def test_format_problem():
    s = "123+456="
    assert format_problem(123, 456, 3) == s


def test_format_problem_pads():
    s = "001+002="
    assert format_problem(1, 2, 3) == s


def test_format_solution():
    s = "4321"
    assert format_solution(1234, 3) == s


def test_format_solution_pads():
    s = "1000"
    assert format_solution(1, 3) == s


def test_generate_data():
    key = jax.random.PRNGKey(42)
    inputs, outputs = generate_data(key, 3, 16)
    assert inputs.shape[0] == outputs.shape[0]
    assert inputs.shape[0] == 16
    assert inputs.shape[1] == 8
    assert outputs.shape[1] == 4
