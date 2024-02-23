import jax
import jax.numpy as jnp
import haiku as hk
import optax
from jax import random
from _common_ml import *
from prettyprint import prettyPrint


# hyperparameters
hp = {
    "dropoutRate": 0.2
}



# Define the neural network with dropout
def net_fn(batch, is_training=True):
    mlp = hk.Sequential([
        hk.Linear(1891), jax.nn.relu,
        # Apply dropout only during training, with corrected argument order
        lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=hp["dropoutRate"]) if is_training else x, 

        # fully connected layer with dropout
        hk.Linear(1891), jax.nn.relu,
        lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=hp["dropoutRate"]) if is_training else x,       

        # fully connected layer with dropout
        hk.Linear(1891), jax.nn.relu,
        lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=hp["dropoutRate"]) if is_training else x,  

        # fully connected layer with dropout
        hk.Linear(100), jax.nn.relu,
        lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=hp["dropoutRate"]) if is_training else x,         
        
        hk.Linear(5)  # Assuming a 10-class classification problem
    ])
    return mlp(batch)

# Transform the function into a form that Haiku can work with
net = hk.transform(net_fn)



def train(obj):

    prettyPrint(obj)

    # Loss function
    def loss_fn(params, rng, inputs, targets):
        predictions = net.apply(params, rng, inputs, is_training=True)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=predictions, labels=targets))
        return loss

    # Update function
    @jax.jit
    def update(params, opt_state, rng, inputs, targets):
        grads = jax.grad(loss_fn)(params, rng, inputs, targets)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    # Placeholder for your data
    # Replace these with actual data loading code
    X_train = jnp.array(obj['data'])
    y_train = jnp.array(obj['labels'])

    prettyPrint(X_train)
    print(X_train.shape)
    prettyPrint(y_train)
    print(y_train.shape)


    batch_size = X_train.shape[0]

    # Initialize the model
    rng = random.PRNGKey(42)
    init_rng, train_rng = jax.random.split(rng)
    params = net.init(init_rng, X_train[:batch_size], is_training=True)

    # Training loop
    num_epochs = 3600*2*8
    num_batches = X_train.shape[0] // batch_size

    # Learning rate schedule: linear ramp-up and then constant
    ramp_up_epochs = 3600*2*4  # Number of epochs to linearly ramp up the learning rate
    total_ramp_up_steps = ramp_up_epochs * num_batches
    lr_schedule = optax.linear_schedule(init_value=1e-6, 
                                        end_value =1e-4, 
                                        transition_steps=total_ramp_up_steps)

    # Optimizer
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)


    for epoch in range(num_epochs):
        for i in range(num_batches):
            batch_rng = random.fold_in(train_rng, i)
            batch_start, batch_end = i * batch_size, (i + 1) * batch_size
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            params, opt_state = update(params, opt_state, batch_rng, X_batch, y_batch)

        # Quick loss check
        if epoch % 1 == 0:
            epoch_loss = loss_fn(params, batch_rng, X_train, y_train)
            print(f"Epoch {epoch}, Loss: {epoch_loss}")


if __name__ == "__main__":
    cmd_line(train, "naive2")


'''
    # Inference
    # Placeholder for new data
    X_new = jnp.zeros((10, 784))  # Example new data

    # Run inference (set is_training=False to disable dropout)
    predictions = net.apply(params, train_rng, X_new, is_training=False)
    probabilities = jax.nn.softmax(predictions)
    print("Inference probabilities:", probabilities)
'''