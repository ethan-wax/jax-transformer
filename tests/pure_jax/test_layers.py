import jax
import jax.numpy as jnp

from src.pure_jax.layers import (
    feedforward,
    init_feedforward_params,
    init_linear_params,
    linear,
    layer_norm,
    positonal_encoding,
    init_embedding_params,
    embedding_lookup,
    init_attention_params,
    multihead_attention,
    init_transformer_params,
    transformer,
)


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
    z = multihead_attention(params, x, 16)
    assert z.shape[0] == 50
    assert z.shape[1] == 128


def test_feedforward():
    key = jax.random.PRNGKey(42)
    params = init_feedforward_params(key, 128, 128 * 4)
    x = jnp.zeros((50, 128))
    z = feedforward(params, x)
    assert z.shape[0] == 50
    assert z.shape[1] == 128


def test_init_feedforward_params():
    key = jax.random.PRNGKey(42)
    params = init_feedforward_params(key, 128, 512)
    assert params["linear_1"]["w"].shape == (128, 512)
    assert params["linear_1"]["b"].shape == (512,)
    assert params["linear_2"]["w"].shape == (512, 128)
    assert params["linear_2"]["b"].shape == (128,)


def test_feedforward_zero_params_is_zero():
    key = jax.random.PRNGKey(0)
    params = init_feedforward_params(key, 32, 64)
    zero_params = jax.tree.map(jnp.zeros_like, params)
    x = jax.random.normal(key, (10, 32))
    z = feedforward(zero_params, x)
    assert jnp.allclose(z, jnp.zeros_like(x))


def test_linear_output_dtype_and_batch():
    key = jax.random.PRNGKey(1)
    params = init_linear_params(key, 16, 32)
    x = jax.random.normal(key, (5, 16))
    z = linear(params, x)
    assert z.shape == (5, 32)


def test_embeddings_lookup_repeated_and_reordered_tokens():
    embeddings_matrix = jnp.arange(12 * 4).reshape(12, 4).astype(jnp.float32)
    tokens = jnp.array([3, 3, 0, 7])
    result = embedding_lookup(embeddings_matrix, tokens)
    assert result.shape == (4, 4)
    assert jnp.allclose(result[0], result[1])
    assert jnp.allclose(result[0], embeddings_matrix[3])
    assert jnp.allclose(result[2], embeddings_matrix[0])
    assert jnp.allclose(result[3], embeddings_matrix[7])


def test_positional_encoding_varies_by_position():
    x = jnp.zeros((10, 16))
    pe = positonal_encoding(x)
    assert not jnp.allclose(pe[0], pe[1])
    assert not jnp.allclose(pe[1], pe[2])


def test_positional_encoding_seq_len_one():
    x = jnp.zeros((1, 8))
    pe = positonal_encoding(x)
    assert pe.shape == x.shape


def test_multihead_attention_first_position_only_attends_to_itself():
    d_model = 4
    n_heads = 2
    params = {
        "w_k": jnp.zeros((d_model, d_model)),
        "w_q": jnp.zeros((d_model, d_model)),
        "w_v": jnp.eye(d_model),
        "w_o": jnp.eye(d_model),
    }
    x = jnp.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    z = multihead_attention(params, x, n_heads)
    assert jnp.allclose(z[0], x[0])


def test_multihead_attention_causal_mask_blocks_future_positions():
    d_model = 4
    n_heads = 2
    params = {
        "w_k": jnp.zeros((d_model, d_model)),
        "w_q": jnp.zeros((d_model, d_model)),
        "w_v": jnp.eye(d_model),
        "w_o": jnp.eye(d_model),
    }
    x = jnp.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    z1 = multihead_attention(params, x, n_heads)

    x2 = x.at[3].set(jnp.array([9.0, 9.0, 9.0, 9.0]))
    z2 = multihead_attention(params, x2, n_heads)

    assert jnp.allclose(z1[:3], z2[:3])
    assert not jnp.allclose(z1[3], z2[3])


def test_multihead_attention_single_head():
    key = jax.random.PRNGKey(7)
    params = init_attention_params(key, 32)
    x = jax.random.normal(key, (6, 32))
    z = multihead_attention(params, x, 1)
    assert z.shape == (6, 32)


def test_multihead_attention_zero_params_is_zero():
    d_model = 16
    n_heads = 4
    params = {
        "w_k": jnp.zeros((d_model, d_model)),
        "w_q": jnp.zeros((d_model, d_model)),
        "w_v": jnp.zeros((d_model, d_model)),
        "w_o": jnp.zeros((d_model, d_model)),
    }
    key = jax.random.PRNGKey(3)
    x = jax.random.normal(key, (7, d_model))
    z = multihead_attention(params, x, n_heads)
    assert jnp.allclose(z, jnp.zeros_like(x))


def test_init_transformer_params():
    key = jax.random.PRNGKey(42)
    d_model = 128
    d_ff = 512
    n_heads = 8
    params = init_transformer_params(key, n_heads, d_model, d_ff)
    assert "attention" in params
    assert "mlp" in params
    assert params["attention"]["w_k"].shape == (d_model, d_model)
    assert params["attention"]["w_q"].shape == (d_model, d_model)
    assert params["attention"]["w_v"].shape == (d_model, d_model)
    assert params["attention"]["w_o"].shape == (d_model, d_model)
    assert params["mlp"]["linear_1"]["w"].shape == (d_model, d_ff)
    assert params["mlp"]["linear_2"]["w"].shape == (d_ff, d_model)


def test_transformer_output_shape():
    key = jax.random.PRNGKey(0)
    d_model = 32
    d_ff = 64
    n_heads = 4
    seq_len = 9
    params = init_transformer_params(key, n_heads, d_model, d_ff)
    x = jax.random.normal(key, (seq_len, d_model))
    z = transformer(params, x, n_heads)
    assert z.shape == (seq_len, d_model)


def test_transformer_identity_with_zero_params():
    d_model = 8
    d_ff = 16
    n_heads = 2
    seq_len = 5
    zero_attn = {
        "w_k": jnp.zeros((d_model, d_model)),
        "w_q": jnp.zeros((d_model, d_model)),
        "w_v": jnp.zeros((d_model, d_model)),
        "w_o": jnp.zeros((d_model, d_model)),
    }
    zero_mlp = {
        "linear_1": {"w": jnp.zeros((d_model, d_ff)), "b": jnp.zeros(d_ff)},
        "linear_2": {"w": jnp.zeros((d_ff, d_model)), "b": jnp.zeros(d_model)},
    }
    params = {"attention": zero_attn, "mlp": zero_mlp}
    x = jax.random.normal(jax.random.PRNGKey(0), (seq_len, d_model))
    z = transformer(params, x, n_heads)
    assert jnp.allclose(z, x)


def test_linear_explicit_values():
    w = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([1.0, 1.0])
    params = {"w": w, "b": b}
    x = jnp.array([1.0, 1.0])
    z = linear(params, x)
    assert jnp.allclose(z, jnp.array([5.0, 7.0]))


def test_layer_norm_explicit_values():
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    z = layer_norm(x)
    expected = jnp.array([-1.34164079, -0.4472136, 0.4472136, 1.34164079])
    assert jnp.allclose(z, expected)


def test_positional_encoding_explicit_values():
    x = jnp.zeros((2, 4))
    pe = positonal_encoding(x)
    expected_pos0 = jnp.array([0.0, 1.0, 0.0, 1.0])
    expected_pos1 = jnp.array([0.84147098, 0.99995, 0.0001, 1.0])
    assert jnp.allclose(pe[0], expected_pos0)
    assert jnp.allclose(pe[1], expected_pos1, atol=1e-5)


def test_feedforward_explicit_values():
    w1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    b1 = jnp.array([0.5, -0.5])
    w2 = jnp.array([[2.0, 0.0], [0.0, 2.0]])
    b2 = jnp.array([0.0, 0.0])
    params = {"linear_1": {"w": w1, "b": b1}, "linear_2": {"w": w2, "b": b2}}
    x = jnp.array([[1.0, -1.0]])

    z = feedforward(params, x)
    expected = jnp.array([[2 * 1.3995715769802328, 2 * -0.10042842301976707]])
    assert jnp.allclose(z, expected)


def test_multihead_attention_explicit_values():
    d_model = 2
    n_heads = 1
    identity = jnp.eye(d_model)
    params = {"w_k": identity, "w_q": identity, "w_v": identity, "w_o": identity}
    x = jnp.eye(2)

    z = multihead_attention(params, x, n_heads)
    expected = jnp.array(
        [[1.0, 0.0], [0.3302384506733431, 0.6697615493266569]]
    )
    assert jnp.allclose(z, expected)
