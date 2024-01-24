from typing import Iterator, Mapping, Tuple
from _common_ml import *
import csv
from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

# we have l | l l l so the sizes are the same
def skipit(width, layers):
  first = hk.Sequential([hk.Linear(width), jax.nn.relu])

  rest = hk.Sequential([
    *[a for i in range(layers - 1) for a in (hk.Linear(width), jax.nn.leaky_relu)]
  ])
  return lambda a : first(a) + rest(a)

def net_fn(batch: np.ndarray) -> jnp.ndarray:
  global globals
  """Standard LeNet-300-100 MLP network."""

  # Define the neural network to apply to the concatenated vectors
  plp = hk.Sequential([skipit(40, 3) , skipit(60,3)])

  global_elements = batch[:, :globals["dataSize"]]
  reshaped_x = jnp.reshape(batch[:, globals["dataSize"]:], (batch.shape[0], 60, 27))
  combined_input = jnp.concatenate([jnp.repeat(global_elements[:, None, :], 60, axis=1), reshaped_x], axis=-1)

  # Apply MLP to each of the 60 subvectors
  transformed_vectors = jax.vmap(plp)(combined_input)

  # Combine the results by summing along the second axis
  combined_result = jnp.sum(transformed_vectors, axis=1)

  final_feedthrough = jnp.concatenate([global_elements, combined_result], axis=-1)

  mlp = hk.Sequential([skipit(600, 3), skipit(300, 3), skipit(50, 3), 
                       hk.Linear(10), jax.nn.relu, hk.Linear(globals["labelSize"])])

  return mlp(final_feedthrough)

#maybe a better way to shuffle data exists, but... ?
#Also, could try vmapping
def mixAtoms(listOfData):
  return listOfData

def main(obj):
  # Make the network and optimiser.
  net = hk.without_apply_rng(hk.transform(net_fn))
  opt = optax.adam(1e-3)

  # Training loss (cross-entropy).
  def loss(params: hk.Params, batch: np.ndarray, labels: np.ndarray) -> jnp.ndarray:
    global globals
    """Compute the loss of the network, including L2."""
    logits = net.apply(params, batch)
    #The labels are prehotted

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent + 1e-4 * l2_loss

  # Evaluation metric (classification accuracy).
  @jax.jit
  def accuracy(params: hk.Params, batch: np.ndarray, labels: np.ndarray) -> jnp.ndarray:
    global globals
    predictions = net.apply(params, batch)
    hot_pred = jax.lax.map(
      lambda a: jax.lax.eq(jnp.arange(globals["labelSize"]), jnp.argmax(a)).astype(float),
      predictions)
    return jnp.mean(jnp.equal(hot_pred, labels).all(axis=1))

  @jax.jit
  def update(
      params: hk.Params,
      opt_state: optax.OptState,
      batch: np.ndarray,
      labels: np.ndarray,
  ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(params, batch, labels)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  # We maintain avg_params, the exponential moving average of the "live" params.
  # avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
  @jax.jit
  def ema_update(params, avg_params):
    return optax.incremental_update(params, avg_params, step_size=0.001)

  # Make datasets.
  iterData = list(np.array_split(np.array(obj), 1000))

  # Initialize network and optimiser; note we draw an input to get shapes.
  params = avg_params = net.init(jax.random.PRNGKey(42), iterData[0][:,:-globals["labelSize"]])
  opt_state = opt.init(params)

  ii = 1
  # Train/eval loop.
  for step in range(100001):
    ii+=1
    if ii == len(iterData):
      ii = 1
    
    if step % 100 == 0:
      # Periodically evaluate classification accuracy on train & test sets.
      train_accuracy = accuracy(avg_params, 
                                mixAtoms(iterData[ii][:,:-globals["labelSize"]]),
                                          iterData[ii][:, -globals["labelSize"]:])
      test_accuracy = accuracy(avg_params,
                                mixAtoms(iterData[0][:,:-globals["labelSize"]]),
                                         iterData[0][:, -globals["labelSize"]:])
      train_accuracy, test_accuracy = jax.device_get(
          (train_accuracy, test_accuracy))
      print(f"[Step {step}] Train / Test accuracy: "
            f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

    
    # Do SGD on a batch of training examples.
    params, opt_state = update(params, opt_state, 
                               mixAtoms(iterData[ii][:,:-globals["labelSize"]]), 
                                         iterData[ii][:, -globals["labelSize"]:])
    avg_params = ema_update(params, avg_params)


cmd_line(main, "csnn")