import jax
from jax._src.pjit import with_layout_constraint
import jax.numpy as jnp


def init_linear_params(key: jax.Array, d_model: int) -> dict[str, jax.Array]:
    """Initialize a set of parameters for a linear layer"""
    std = d_model**-0.5
    w = jax.random.normal(key, (d_model, d_model)) * std
    b = jnp.zeros(d_model)
    return {"w": w, "b": b}


def linear(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    """Multiply the params matrix w by the activations x, and add the biases b"""
    w = params["w"]
    b = params["b"]
    return x @ w + b


def layer_norm(x: jax.Array) -> jax.Array:
    """Normalize x"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return (x - mean) / jnp.sqrt(var)


def positonal_encoding(x: jax.Array) -> jax.Array:
    """Add the positional element to the vector x"""
    seq_len, d_model = x.shape
    pos = jnp.arange(seq_len)[:, None]
    dims = jnp.arange(d_model)[None, :]
    omegas = 1.0 / (10000 ** (2 * dims / d_model))
    angles = pos * omegas
    pos_encs = jnp.where(dims % 2 == 0, jnp.sin(angles), jnp.cos(angles))
    return x + pos_encs


def init_embedding_params(key: jax.Array, vocab_size: int, d_model: int) -> jax.Array:
    """Initialize a set of parameters for the embedding matrix"""
    std = d_model**-0.5
    return jax.random.normal(key, (vocab_size, d_model)) * std


def embedding_lookup(embedding_matrix: jax.Array, tokens: jax.Array) -> jax.Array:
    """Return the embeddings for each token in tokens"""
    return embedding_matrix[tokens]


def init_attention_params(key: jax.Array, d_model: int) -> dict[str, jax.Array]:
    scalar = d_model**-0.5
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "w_k": jax.random.normal(k1, (d_model, d_model)) * scalar,
        "w_q": jax.random.normal(k2, (d_model, d_model)) * scalar,
        "w_v": jax.random.normal(k3, (d_model, d_model)) * scalar,
        "w_o": jax.random.normal(k4, (d_model, d_model)) * scalar,
    }


def multihead_attention(
    params: dict[str, jax.Array], x: jax.Array, n_heads: int
) -> jax.Array:
    seq_len, d_model = x.shape
    d_head = d_model // n_heads

    k = (x @ params["w_k"]).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    q = (x @ params["w_q"]).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    v = (x @ params["w_v"]).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    scores = (q @ k.transpose(0, 2, 1)) / jnp.sqrt(d_head)
    rows, cols = jnp.indices((seq_len, seq_len))
    mask = jnp.where(cols > rows, -float("inf"), 0)
    weights = jax.nn.softmax(scores + mask) @ v

    h = weights.transpose(1, 0, 2).reshape(seq_len, d_model)
    o = h @ params["w_o"]

    return o
