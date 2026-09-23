import jax
import jax.numpy as jnp
from src.data import encode, decode
from src.pure_jax.train import N_LAYERS, D_MODEL, N_HEADS, D_FF, VOCAB_SIZE
from src.pure_jax.model import forward, init_model
from src.pure_jax.checkpoint import load_checkpoint

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
template = init_model(subkey, N_LAYERS, D_MODEL, N_HEADS, D_FF, VOCAB_SIZE)
params = load_checkpoint(template, "checkpoints/model.msgpack")

while True:
    s = input()
    tokens = encode(s)
    for _ in range(4):
        key, subkey = jax.random.split(key)
        logits = forward(params, tokens, N_HEADS)
        idx = jax.random.categorical(subkey, logits[-1])
        tokens = jnp.append(tokens, idx)
    print(decode(tokens.tolist()))
