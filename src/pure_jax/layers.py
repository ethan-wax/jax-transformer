import jax
import jax.numpy as jnp


def init_linear_params(key: jax.Array, d_in: int, d_out: int) -> dict[str, jax.Array]:
    """Initialize a set of parameters for a linear layer"""
    std = d_in**-0.5
    w = jax.random.normal(key, (d_in, d_out)) * std
    b = jnp.zeros(d_out)
    return {"w": w, "b": b}


def linear(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """Multiply the params matrix w by the activations x, and add the biases b"""
    w = params["w"]
    b = params["b"]
    return w @ x + b


def layer_norm(x: jax.Array) -> jax.Array:
    """Normalize x"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return (x - mean) / jnp.sqrt(var)
