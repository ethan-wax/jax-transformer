import jax
import jax.numpy as jnp

from src.pure_jax.layers import (
    init_linear_params,
    linear,
    layer_norm,
    positonal_encoding,
    init_embedding_params,
    embedding_lookup,
    init_attention_params,
    multihead_attention,
)


def test_init_linear_params():
    key = jax.random.PRNGKey(42)
    params = init_linear_params(key, 128)
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


def test_linear_output_shape():
    w = jnp.eye(64, 128)
    b = jnp.zeros(128)
    params = {"w": w, "b": b}
    x = jnp.array([i for i in range(64)])
    assert (linear(params, x) == jnp.concat([x, jnp.zeros(64)])).any()


def test_layer_norm_output_shape():
    x = jnp.ones(128)
    assert layer_norm(x).shape == x.shape


def test_layer_norm():
    key = jax.random.PRNGKey(42)
    x = jax.random.randint(key, (128,), 0, 128)
    result = layer_norm(x)
    assert jnp.allclose(jnp.mean(result), 0)
    assert jnp.allclose(jnp.var(result), 1)


def test_positional_encodings():
    x = jnp.zeros((128, 64))
    pe = positonal_encoding(x)
    assert pe.shape == x.shape
    for i in range(128):
        if i % 2 == 0:
            assert jnp.allclose(pe[0, 0::2], 0.0)
        else:
            assert jnp.allclose(pe[0, 1::2], 1.0)


def test_init_embeddings_params():
    key = jax.random.PRNGKey(42)
    params = init_embedding_params(key, 12, 128)
    assert params.shape[0] == 12
    assert params.shape[1] == 128


def test_embeddings_lookup():
    embeddings_matrix = jnp.eye(12, 128)
    tokens = jnp.arange(6)
    result = embedding_lookup(embeddings_matrix, tokens)
    assert jnp.allclose(result, jnp.eye(6, 128))


def test_init_attention_params():
    key = jax.random.PRNGKey(42)
    params = init_attention_params(key, 128)
    assert "w_k" in params
    assert "w_q" in params
    assert "w_v" in params
    assert "w_o" in params
    assert params["w_k"].shape[0] == 128
    assert params["w_k"].shape[1] == 128
    assert params["w_q"].shape[0] == 128
    assert params["w_q"].shape[1] == 128
    assert params["w_v"].shape[0] == 128
    assert params["w_v"].shape[1] == 128
    assert params["w_o"].shape[0] == 128
    assert params["w_o"].shape[1] == 128


def test_multihead_attention():
    w_k = jnp.zeros((128, 128))
    w_q = jnp.zeros((128, 128))
    w_v = jnp.zeros((128, 128))
    w_o = jnp.zeros((128, 128))
    params = {"w_k": w_k, "w_q": w_q, "w_v": w_v, "w_o": w_o}
    x = jnp.zeros((50, 128))
    o = multihead_attention(params, x, 16)
    assert o.shape[0] == 50
    assert o.shape[1] == 128
