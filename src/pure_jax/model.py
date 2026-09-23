import jax
import jax.numpy as jnp
from src.pure_jax.layers import (
    embedding_lookup,
    init_embedding_params,
    init_linear_params,
    init_transformer_params,
    positonal_encoding,
    transformer,
    layer_norm,
    linear,
)


def forward(params: dict, x: jax.Array, n_heads: int):
    embeddings = embedding_lookup(params["embeddings"], jnp.array(x))
    h = positonal_encoding(embeddings)

    for layers in params["transformers"]:
        h = transformer(layers, h, n_heads)

    h = layer_norm(h)
    z = linear(params["output"], h)
    return z


def init_model(
    key: jax.Array,
    n_layers: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    vocab_size: int,
):
    assert n_layers >= 0
    keys = jax.random.split(key, n_layers + 2)
    params = {
        "embeddings": init_embedding_params(keys[0], vocab_size, d_model),
        "transformers": [
            init_transformer_params(k, n_heads, d_model, d_ff) for k in keys[1:-1]
        ],
        "output": init_linear_params(keys[-1], d_model, vocab_size),
    }
    return params
