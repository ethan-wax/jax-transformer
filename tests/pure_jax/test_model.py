import jax
import jax.numpy as jnp

from src.data import encode
from src.pure_jax.layers import embedding_lookup, layer_norm, linear, positonal_encoding
from src.pure_jax.model import forward, init_model

VOCAB_SIZE = 12


def test_init_model_shapes():
    key = jax.random.PRNGKey(42)
    n_layers = 3
    d_model = 16
    n_heads = 4
    d_ff = 32
    params = init_model(key, n_layers, d_model, n_heads, d_ff, VOCAB_SIZE)

    assert params["embeddings"].shape == (VOCAB_SIZE, d_model)
    assert len(params["transformers"]) == n_layers
    assert params["output"]["w"].shape == (d_model, VOCAB_SIZE)
    assert params["output"]["b"].shape == (VOCAB_SIZE,)
    for layer in params["transformers"]:
        assert layer["attention"]["w_k"].shape == (d_model, d_model)
        assert layer["mlp"]["linear_1"]["w"].shape == (d_model, d_ff)
        assert layer["mlp"]["linear_2"]["w"].shape == (d_ff, d_model)


def test_init_model_zero_layers():
    key = jax.random.PRNGKey(0)
    params = init_model(key, 0, 8, 2, 16, VOCAB_SIZE)
    assert params["transformers"] == []


def test_init_model_different_keys_give_different_params():
    d_model = 8
    params_a = init_model(jax.random.PRNGKey(1), 1, d_model, 2, 16, VOCAB_SIZE)
    params_b = init_model(jax.random.PRNGKey(2), 1, d_model, 2, 16, VOCAB_SIZE)
    assert not jnp.allclose(params_a["embeddings"], params_b["embeddings"])


def test_forward_output_shape():
    key = jax.random.PRNGKey(42)
    n_layers = 2
    d_model = 16
    n_heads = 4
    d_ff = 32
    params = init_model(key, n_layers, d_model, n_heads, d_ff, VOCAB_SIZE)

    x = encode("123+456=")
    z = forward(params, x, n_heads)
    assert z.shape == (len(x), VOCAB_SIZE)


def test_forward_output_shape_varies_with_input_length():
    key = jax.random.PRNGKey(2)
    n_heads = 2
    params = init_model(key, 1, 8, n_heads, 16, VOCAB_SIZE)

    z_short = forward(params, encode("1+2="), n_heads)
    z_long = forward(params, encode("123+456="), n_heads)

    assert z_short.shape == (4, VOCAB_SIZE)
    assert z_long.shape == (8, VOCAB_SIZE)


def test_forward_deterministic():
    key = jax.random.PRNGKey(1)
    n_heads = 2
    params = init_model(key, 1, 8, n_heads, 16, VOCAB_SIZE)

    x = encode("5+5=")
    z1 = forward(params, x, n_heads)
    z2 = forward(params, x, n_heads)
    assert jnp.allclose(z1, z2)


def test_forward_zero_layers_matches_manual_projection():
    key = jax.random.PRNGKey(0)
    d_model = 8
    n_heads = 2
    params = init_model(key, 0, d_model, n_heads, 16, VOCAB_SIZE)

    x = encode("12+3=")
    z = forward(params, x, n_heads)

    tokens = jnp.array(x)
    embeddings = embedding_lookup(params["embeddings"], tokens)
    h = positonal_encoding(embeddings)
    h = layer_norm(h)
    expected = linear(params["output"], h)

    assert jnp.allclose(z, expected)
