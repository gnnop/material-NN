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
  """
  #This network got a final score of 1 on the training data and 0.395 on the test data.
  #
  mlp = hk.Sequential([#2530
      hk.Flatten(),
      hk.Linear(500), jax.nn.relu,
      hk.Linear(500), jax.nn.relu,
      hk.Linear(200), jax.nn.relu,
      hk.Linear(50), jax.nn.relu,
      hk.Linear(3),
  ])
  """

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

  y1 = a1(x)
  y2 = y1 + a2(y1)
  clp = hk.Sequential([
    hk.Linear(20), jax.nn.relu,
    hk.Linear(20), jax.nn.relu,
    hk.Linear(15)
  ])

  inte = jax.numpy.array([clp(x[:,globals["dataSize"] + i*27:globals["dataSize"] + (i+1)*27]) for i in range(60)])
  interm = jax.numpy.concatenate(inte, axis=1)
  intermezzo = jax.numpy.concatenate((x[:,0:globals["dataSize"]],interm), axis=1)

  mlp = hk.Sequential([#2530 - attempt deep
      hk.Flatten(),
      hk.Linear(500), jax.nn.relu,
      hk.Linear(200), jax.nn.relu,
      hk.Linear(50), jax.nn.relu,
      hk.Linear(50), jax.nn.relu,
      hk.Linear(3),
  ])
  #I need a list of ways to modify training data.
  #1 - permute data
  #2 - expand lattice vectors. All expanded vectors will be trivial
  #3 - comon material substitutions?
  #4 - use a tree structure to determine atoms - decrease atom representation size
  return mlp(intermezzo)

def FetchData(listOfData):
  global globals
  print("FetchData: ", globals["labelSize"])
  permList = list(range(globals["dataSize"] + 27*60 + globals["labelSize"]))

  copyOfData = listOfData
  for i in range(len(listOfData)):
    perm = np.random.permutation(60)
    permList[globals["dataSize"]:-globals["labelSize"]] = [globals["dataSize"] + j + perm[i]*27 for j in range(27) for i in range(60)]
    copyOfData[i] = listOfData[i, permList]
  return copyOfData

def main(obj):
  # Make the network and optimiser.
  net = hk.without_apply_rng(hk.transform(net_fn))
  opt = optax.adam(1e-3)

  # Training loss (cross-entropy).
  def loss(params: hk.Params, batch: np.ndarray) -> jnp.ndarray:
    global globals
    """Compute the loss of the network, including L2."""
    logits = net.apply(params, batch[:,:-1])
    labels = jax.nn.one_hot(batch[:,-1], globals["labelSize"])

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent + 1e-4 * l2_loss

  # Evaluation metric (classification accuracy).
  @jax.jit
  def accuracy(params: hk.Params, batch: np.ndarray) -> jnp.ndarray:
    global globals
    predictions = net.apply(params, batch[:,:-globals["labelSize"]])
    return jnp.mean(jnp.equal(predictions, batch[:, -globals["labelSize"]]).all(axis=1))

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
  iterData = list(np.array_split(np.array(obj), 1000))

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
      train_accuracy = accuracy(avg_params, FetchData(iterData[ii]))
      test_accuracy = accuracy(avg_params, iterData[0])
      train_accuracy, test_accuracy = jax.device_get(
          (train_accuracy, test_accuracy))
      print(f"[Step {step}] Train / Test accuracy: "
            f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

    
    # Do SGD on a batch of training examples.
    params, opt_state = update(params, opt_state, FetchData(iterData[ii]))
    avg_params = ema_update(params, avg_params)


cmd_line(main, "naive")