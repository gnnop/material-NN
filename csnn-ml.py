from typing import Iterator, Mapping, Tuple
from _common_ml import *
import csv
from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

def net_fn(batch: np.ndarray) -> jnp.ndarray:
  """Standard LeNet-300-100 MLP network."""
  x = batch.astype(jnp.float32)
  
  clp = hk.Sequential([
    hk.Linear(20), jax.nn.relu,
    hk.Linear(20), jax.nn.relu,
    hk.Linear(15)
  ])

  parallelMap = hk.Sequential([
    hk.Linear(27), jax.nn.relu,
    hk.Linear(40), jax.nn.relu,
    hk.Linear(60), jax.nn.relu,
    hk.Linear(80), jax.nn.relu,
    hk.Linear(100), jax.nn.relu
  ])

  globSpec = jnp.repeat(x[:, 0:6], 60)
  altSpec = jax.numpy.reshape(x[:, 6:-1], (27, 60))
  feedIn = jnp.concatenate(globSpec, altSpec)
  symmatrix = hk.Flatten(jax.lax.reduce(jax.lax.add, jax.vmap(parallelMap)(feedIn)))#no clue if this works

  #Now, add the global vector back in

  symM = jnp.concatenate(x[:, 0:6], symmatrix)


  mlp = hk.Sequential([#2530 - attempt deep
      hk.Flatten(),
      hk.Linear(500), jax.nn.relu,
      hk.Linear(200), jax.nn.relu,
      hk.Linear(50), jax.nn.relu,
      hk.Linear(50), jax.nn.relu,
      hk.Linear(3),
  ])



  return mlp(symM)#It might make sense to add in an extra flatten to make this make sense.

def main(_):
  # Make the network and optimiser.
  net = hk.without_apply_rng(hk.transform(net_fn))
  opt = optax.adam(1e-3)

  # Training loss (cross-entropy).
  def loss(params: hk.Params, batch: np.ndarray) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    logits = net.apply(params, batch[:,:-1])
    labels = jax.nn.one_hot(batch[:,-1], 3)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent + 1e-4 * l2_loss

  # Evaluation metric (classification accuracy).
  @jax.jit
  def accuracy(params: hk.Params, batch: np.ndarray) -> jnp.ndarray:
    predictions = net.apply(params, batch[:,:-1])
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch[:,-1])

  @jax.jit
  def update(
      params: hk.Params,
      opt_state: optax.OptState,
      batch: np.ndarray,
  ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(params, batch)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  # We maintain avg_params, the exponential moving average of the "live" params.
  # avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
  @jax.jit
  def ema_update(params, avg_params):
    return optax.incremental_update(params, avg_params, step_size=0.001)

  # Make datasets.
  with open('data.csv', 'r', newline='') as file:
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    data = list(reader)
    #now all data is in a list of lists
    processedStuff = np.array(data)
    iterData = list(np.array_split(processedStuff, 1000))

  # Initialize network and optimiser; note we draw an input to get shapes.
  params = avg_params = net.init(jax.random.PRNGKey(42), iterData[0][:,:-1])
  opt_state = opt.init(params)

  ii = 1
  # Train/eval loop.
  for step in range(100001):
    ii+=1
    if ii == len(iterData):
      ii = 1
    
    if step % 100 == 0:
      # Periodically evaluate classification accuracy on train & test sets.
      train_accuracy = accuracy(avg_params, iterData[ii])
      test_accuracy = accuracy(avg_params, iterData[0])
      train_accuracy, test_accuracy = jax.device_get(
          (train_accuracy, test_accuracy))
      print(f"[Step {step}] Train / Test accuracy: "
            f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

    
    # Do SGD on a batch of training examples.
    params, opt_state = update(params, opt_state, iterData[ii])
    avg_params = ema_update(params, avg_params)


cmd_line(main, "csnn")


"""
In theory good for multi-threading, but I don't know enough about it
"""
#if __name__ == "__main__":
#  app.run(main)