import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
from typing import List
from flax.training.train_state import TrainState

N_SAMPLES = 200
LAYERS = [1, 10, 10, 10, 1]
LEARNING_RATE = 0.1
N_EPOCHS = 30_000

key = jax.random.PRNGKey(42)

# We split the key into 3. 
# key: We keep the key for other things
# xkey: to produce the x -> y samples
# ynosekey: to introduce noise in y
key, xkey, ynoisekey = jax.random.split(key, 3)

# We generate x samples from the uniform distribution. A one dimentional tensor
# of N_SAMPLES
x_samples = jax.random.uniform(xkey, (N_SAMPLES, 1), minval = 0.0, maxval = 2*jnp.pi)

# We create a y_sample vector + some random noise
y_samples = jnp.sin(x_samples) + jax.random.normal(ynoisekey, (N_SAMPLES, 1)) * 0.3

#plt.scatter(x_samples, y_samples)
#plt.show()

# Multi-layer perceptron
class SimpleMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features = 10, name = 'input_layer')(x)
        x = nn.sigmoid(x)

        x = nn.Dense(features = 10, name = 'hidden_layer_1')(x)
        x = nn.sigmoid(x)

        x = nn.Dense(features = 10, name = 'hidden_layer_2')(x)
        x = nn.sigmoid(x)

        x = nn.Dense(features = 10, name = 'hidden_layer_3')(x)
        x = nn.sigmoid(x)

        x = nn.Dense(features = 1, name = 'output_layer')(x)

        return x

model = SimpleMLP()
dummy_input = jnp.ones((10,1)) # Dummy imput to trigger shape inference
key, key2 = jax.random.split(key)
params = model.init(key2, dummy_input)

output = model.apply(params, jnp.array([0.5]))
#print(output)

#plt.scatter(x_samples, y_samples)
# we need to map the model across all the y_samples
#plt.scatter(x_samples, model.apply({'params': params}, x_samples))
#plt.show()

state = TrainState.create(
        apply_fn = model.apply,
        params = params['params'],
        tx = optax.adam(LEARNING_RATE)
)


def loss_fn(params, xs, ys):
    prediction = state.apply_fn({'params': params}, xs)
    delta = prediction - ys
    loss = jnp.mean(delta**2)
    return loss

#print(loss_fn(state.params, jnp.array([0.5]), jnp.array([1])))

@jax.jit
def train_step(state, xs, ys):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, xs, ys)
    state = state.apply_gradients(grads=grads)
    return loss, state


loss_history = []

for epoch in range(N_EPOCHS):
    loss, state = train_step(state, x_samples, y_samples)
    if epoch % 100 == 0:
        print(loss)
    loss_history.append(loss)

#plt.plot(loss_history)
#plt.yscale("log")
#plt.show()

plt.scatter(x_samples, y_samples)
plt.scatter(x_samples, model.apply({'params': state.params}, x_samples))
plt.show()
