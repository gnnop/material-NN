import jax
import jax.numpy as jnp
import haiku as hk
import optax
from jax import random
from _common_ml import *
from prettyprint import prettyPrint

# hyperparameters
hp = {
    "dropoutRate": 0.05
}



# Define the neural network with dropout
def net_fn(batch, is_training=False, dropout_rate=0):
    mlp = hk.Sequential([
        # fully connected layer with dropout
        hk.Linear(1000), jax.nn.relu,
        # Apply dropout only during training, with corrected argument order
        lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=dropout_rate) if is_training else x,  

        hk.Linear(5)  # Assuming full set of categories
    ])
    return mlp(batch)
# end of net_fn

# Transform the function into a form that Haiku can work with
net = hk.transform(net_fn)


# Function to partition the dataset into training and validation sets
def partition_dataset(data, labels, validation_percentage):
    data = data.astype(jnp.float16)
    labels = labels.astype(jnp.float16)
    # Calculate the number of validation samples
    num_data = data.shape[0]
    num_val_samples = int(num_data * validation_percentage)

    # Generate shuffled indices
    indices = jnp.arange(num_data)
    shuffled_indices = jax.random.permutation(jax.random.PRNGKey(314), indices)

    # Split the data and labels into training and validation sets
    val_indices = shuffled_indices[:num_val_samples]
    train_indices = shuffled_indices[num_val_samples:]

    X_train, y_train = data[train_indices], labels[train_indices]
    X_val, y_val = data[val_indices], labels[val_indices]

    return X_train, y_train, X_val, y_val
# end of partition_dataset

def train(obj):

    prettyPrint(obj)

    # Loss function for training
    def loss_fn(params, rng, inputs, targets):
        predictions = net.apply(params, rng, inputs, is_training=True, dropout_rate=hp['dropoutRate'])
        loss = jnp.sum(optax.softmax_cross_entropy(logits=predictions, labels=targets))
        return loss
    
    # Accuracy function for us to evaluate the model
    def accuracy_fn(params, rng, inputs, targets):
        predictions = net.apply(params, rng, inputs, is_training=False)
        accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == jnp.argmax(targets, axis=-1))
        return accuracy

    # Update function
    @jax.jit
    def update(params, opt_state, rng, inputs, targets):
        grads = jax.grad(loss_fn)(params, rng, inputs, targets)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    # Placeholder for your data
    X_train = jnp.array(obj['data'])
    y_train = jnp.array(obj['labels'])
    print(f"X_train original shape: {X_train.shape}")
    print(f"y_train original shape: {y_train.shape}")

    # Partition the dataset into training and validation
    X_train, y_train, X_val, y_val = partition_dataset(X_train, y_train, 0.1)

    print("After partitioning:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"{X_val.shape[0]/(X_train.shape[0] + X_val.shape[0]) * 100}% of the data is used for validation")


    batch_size = X_train.shape[0]

    # Initialize the model
    rng = random.PRNGKey(0x09F911029D74E35BD84156C5635688C0 % 2**32)
    init_rng, train_rng = jax.random.split(rng)
    params = net.init(init_rng, X_train[:batch_size], is_training=True)

    # Training loop
    num_epochs = 5000
    num_batches = X_train.shape[0] // batch_size

    # Learning rate schedule: linear ramp-up and then constant
    ramp_up_epochs = 1000  # Number of epochs to linearly ramp up the learning rate
    total_ramp_up_steps = ramp_up_epochs * num_batches
    lr_schedule = optax.linear_schedule(init_value=1e-6, 
                                        end_value =5e-5, 
                                        transition_steps=total_ramp_up_steps)

    # Optimizer
    optimizer = optax.noisy_sgd(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(num_epochs):
        for i in range(num_batches):
            batch_rng = random.fold_in(train_rng, i)
            batch_start, batch_end = i * batch_size, (i + 1) * batch_size
            X_batch = X_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]
            params, opt_state = update(params, opt_state, batch_rng, X_batch, y_batch)

        # Save the training and validation loss
        train_loss = loss_fn(params, batch_rng, X_train, y_train)
        val_loss = loss_fn(params, batch_rng, X_val, y_val)
        train_accuracy = accuracy_fn(params, batch_rng, X_train, y_train)
        val_accuracy = accuracy_fn(params, batch_rng, X_val, y_val)
        # if epoch % 2 == 0:
        print(f"Epoch {epoch}, Training loss: {train_loss}, Validation loss: {val_loss}, Training accuracy: {train_accuracy}, Validation accuracy: {val_accuracy}")
    # end of for epoch
        
    # TODO save the model


if __name__ == "__main__":
    cmd_line(train, "naive")


'''
    # Inference
    # Placeholder for new data
    X_new = jnp.zeros((10, 784))  # Example new data

    # Run inference (set is_training=False to disable dropout)
    predictions = net.apply(params, train_rng, X_new, is_training=False)
    probabilities = jax.nn.softmax(predictions)
    print("Inference probabilities:", probabilities)
'''