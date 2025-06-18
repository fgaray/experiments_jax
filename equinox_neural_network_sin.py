import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from typing import List

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
class SimpleMLP(eqx.Module):
    layers: List[eqx.nn.Linear]

    def __init__(self, layer_sizes, key):
        self.layers = []
        for (din, dout) in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(din, dout, use_bias = True, key = subkey))

    def __call__(self, x):
        a = x
        # The last layer do not need to apply the non linear function
        for layer in self.layers[:-1]:
            a = jax.nn.sigmoid(layer(a))

        a = self.layers[-1](a)
        return a

model = SimpleMLP(LAYERS, key = key)

#plt.scatter(x_samples, y_samples)
# we need to map the model across all the y_samples
#plt.scatter(x_samples, jax.vmap(model)(y_samples))
#plt.show()


def model_to_loss(model, xs, ys):
    prediction = jax.vmap(model)(xs)
    delta = prediction - ys
    loss = jnp.mean(delta**2)
    return loss

print("Initial loss ", model_to_loss(model, x_samples, y_samples))

model_to_loss_and_grad = eqx.filter_value_and_grad(model_to_loss)

# Stochastic gradiant descent
opt = optax.sgd(LEARNING_RATE)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def make_step(model, opt_state, xs, ys):
    loss, grad = model_to_loss_and_grad(model, xs, ys)
    updates, opt_state = opt.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

loss_history = []
for epoch in range(N_EPOCHS):
    model, opt_state, loss = make_step(model, opt_state, x_samples, y_samples)
    loss_history.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss {loss}")


plt.plot(loss_history)
plt.yscale("log")
plt.show()
