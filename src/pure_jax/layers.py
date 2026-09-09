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


def positonal_encoding(x: jax.Array) -> jax.Array:
    """Adds the positional element to the vector x"""
    seq_len, d_model = x.shape
    pos = jnp.arange(seq_len)[:, None]
    dims = jnp.arange(d_model)[None, :]
    omegas = 1.0 / (10000 ** (2 * dims / d_model))
    angles = pos * omegas
    pos_encs = jnp.where(dims % 2 == 0, jnp.sin(angles), jnp.cos(angles))
    return x + pos_encs
