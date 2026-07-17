import jax
import jax.numpy as jnp


def generate_data(batch_size: int, seed: int) -> jax.Array:
    key = jax.random.key(seed)
    _, k1, k2 = jax.random.split(key, 3)
    a = jax.random.randint(k1, shape=(batch_size, 1), minval=100, maxval=1000)
    b = jax.random.randint(k2, shape=(batch_size, 1), minval=100, maxval=1000)
    c = a + b
    return jnp.concatenate([a, b, c], axis=1)


print(generate_data(5, 1))
