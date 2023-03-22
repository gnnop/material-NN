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
  global globals
  """Standard LeNet-300-100 MLP network."""
  x = batch.astype(jnp.float32)

  a1 = hk.Sequential([
    hk.Flatten(),
    hk.Linear(500), jax.nn.relu,
    hk.Linear(500), jax.nn.relu,
    hk.Linear(500), jax.nn.relu,
    hk.Linear(500), jax.nn.relu,
  ])

  a2 = hk.Sequential([
    hk.Linear(500), jax.nn.relu,
    hk.Linear(500), jax.nn.relu,
    hk.Linear(500), jax.nn.relu,
    hk.Linear(500), jax.nn.relu,
  ])

  #No softmax required, because it's order preserving, and we only need the 
  #cross-entropy for training
  a3 = hk.Sequential([
    hk.Linear(400), jax.nn.relu,
    hk.Linear(300), jax.nn.relu,
    hk.Linear(200), jax.nn.relu,
    hk.Linear(100), jax.nn.relu,
    hk.Linear(50), jax.nn.relu,
    hk.Linear(20), jax.nn.relu,
    hk.Linear(globals["labelSize"])
  ])

  y1 = a1(x)
  y2 = y1 + a2(y1)
  y3 = a3(y2)

  return y3

#maybe a better way to shuffle data exists, but... ?
#Also, could try vmapping
def mixAtoms(listOfData):
  global globals
  permList = list(range(globals["dataSize"] + 27*60))

  copyOfData = listOfData
  for i in range(len(listOfData)):
    perm = np.random.permutation(60)
    permList[globals["dataSize"]:] = [globals["dataSize"] + j + perm[i]*27 for j in range(27) for i in range(60)]
    copyOfData[i] = listOfData[i, permList]
  return copyOfData

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


cmd_line(main, "naive")