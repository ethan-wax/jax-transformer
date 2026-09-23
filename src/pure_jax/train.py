import jax
import jax.numpy as jnp
import optax
from src.pure_jax.model import forward, init_model
from src.pure_jax.checkpoint import save_checkpoint
from src.data import generate_data

N_LAYERS = 3
N_HEADS = 4
D_MODEL = 64
D_FF = 256
VOCAB_SIZE = 12
NUM_ROUNDS = 10000
CHECKPOINT_PATH = "checkpoints/model.msgpack"
CHECKPOINT_EVERY = 100

batched_forward = jax.vmap(forward, in_axes=(None, 0, None))
key = jax.random.PRNGKey(42)
params = init_model(key, N_LAYERS, D_MODEL, N_HEADS, D_FF, VOCAB_SIZE)
optimizer = optax.adam(3e-4)


def loss_fn(params: dict, x: jax.Array, y: jax.Array):
    full = jnp.concat([x, y], axis=1)
    model_input = full[:, :-1]
    targets = full[:, 1:]

    prompt_len = x.shape[1]
    positions = jnp.arange(full.shape[1] - 1)
    loss_mask = positions >= (prompt_len - 1)

    logits = batched_forward(params, model_input, N_HEADS)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)

    return jnp.sum(token_loss * loss_mask) / jnp.sum(loss_mask)


def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


train_step = jax.jit(train_step)
opt_state = optimizer.init(params)

keys = jax.random.split(key, NUM_ROUNDS)
for i, k in enumerate(keys):
    x, y = generate_data(k)
    params, opt_state, loss = train_step(params, opt_state, x, y)
    if i != 0 and (i + 1) % 100 == 0:
        print(f"Loss after {i + 1} rounds of training: {loss}")
    if (i + 1) % CHECKPOINT_EVERY == 0:
        save_checkpoint(params, CHECKPOINT_PATH)

save_checkpoint(params, CHECKPOINT_PATH)
