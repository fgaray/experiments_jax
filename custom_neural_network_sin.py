# Custom neural network to aproximate sin function
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


N_SAMPLES = 200
LAYERS = [1, 10, 10, 10, 1] # First layer is dimension 1
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


# Weight initialization

weight_matrices = []
bias_vector = []
activation_functions = []

# Create the neural network
for (din, dout) in zip(LAYERS[:-1], LAYERS[1:]):
    # Xavier Uniform Initialization
    #
    # This ensures that the variance of the outputs of a layer is approximately
    # equal to the variance of its inputs, helping to mitigate
    # vanishing/exploding gradients during training, especially in networks with
    # many layers.
    kernel_matrix_uniform_limit =  jnp.sqrt(6 / (din + dout))

    key, wkey = jax.random.split(key)
    W = jax.random.uniform(wkey, (din, dout), minval=-kernel_matrix_uniform_limit, maxval = kernel_matrix_uniform_limit)
    # The bias is applied to the output dimension
    b = jnp.zeros(dout)

    weight_matrices.append(W)
    bias_vector.append(b)
    activation_functions.append(jax.nn.sigmoid)

# Last one is just the identity
activation_functions[-1] = lambda x: x

def network_forward(x, weights, biases, activations):
    a = x # activated state in each layer

    for W, b, f in zip(weights, biases, activations):
        # @ Matrix multiplication
        a = f(a @ W + b)
    return a

#plt.scatter(x_samples, y_samples)
#plt.scatter(x_samples, network_forward(x_samples, weight_matrices, bias_vector, activation_functions))
#plt.show()

#print(network_forward(jnp.array([[0.5], [0.6]]), weight_matrices, bias_vector, activation_functions))

def loss_forward(y_guess, y_ref):
    delta = y_guess - y_ref
    return 0.5*jnp.mean(delta**2) # Mean square error


loss_and_grad_fun = jax.value_and_grad(lambda Ws, bs: loss_forward(network_forward(x_samples, Ws, bs, activation_functions), y_samples), argnums=(0, 1))
loss_and_grad_fun = jax.jit(loss_and_grad_fun)

initial_loss, (initial_weight_gradient, initial_bias_gradient) = loss_and_grad_fun(weight_matrices, bias_vector)
print("Initial loss: ", initial_loss)



# Training loop
loss_history = []
(weight_gradient, bias_gradient) = (initial_weight_gradient, initial_bias_gradient)
for epoch in range(N_EPOCHS):
    loss, (weight_gradients, bias_gradients) = loss_and_grad_fun(weight_matrices, bias_vector)

    weight_matrices = jax.tree.map(
            lambda W, W_grad: W - LEARNING_RATE*W_grad,
            weight_matrices,
            weight_gradients
    )
    bias_vector = jax.tree.map(
            lambda b, b_grad: b - LEARNING_RATE*b_grad,
            bias_vector,
            bias_gradients
    )

    if epoch % 100 == 0:
        print(f"epoch = {epoch}, loss = {loss}")

    loss_history.append(loss)

plt.plot(loss_history)
plt.yscale("log")
plt.show()


plt.scatter(x_samples, y_samples)
plt.scatter(x_samples, network_forward(x_samples, weight_matrices, bias_vector, activation_functions))
plt.show()
