import jax
import jax.numpy as jnp

from src.pure_jax.layers import init_linear_params, linear


def test_init_linear_params():
    key = jax.random.PRNGKey(42)
    params = init_linear_params(key, 128, 128)
    assert "w" in params
    assert "b" in params
    assert params["w"].shape[0] == 128
    assert params["w"].shape[1] == 128
    assert params["b"].shape[0] == 128


def test_linear():
    w = jnp.eye(128)
    b = jnp.ones(128)
    params = {"w": w, "b": b}
    x = jnp.array([i for i in range(128)])
    assert (linear(params, x) == x + 1).all()


def test_linear_rectangular():
    w = jnp.eye(128, 64)
    b = jnp.zeros(128)
    params = {"w": w, "b": b}
    x = jnp.array([i for i in range(64)])
    assert (linear(params, x) == jnp.concat([x, jnp.zeros(64)])).any()
