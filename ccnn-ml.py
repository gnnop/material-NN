import functools
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
from os.path import exists
import haiku as hk
import math
import optax
import pickle
import numpy as np
import networkx as nx
from typing import Any, Callable, Dict, List, Optional, Tuple
from random import shuffle
import time
from multiprocessing.dummy import Pool as ThreadPool
import jax.numpy as jnp
import haiku as hk
import trimesh

def convFunc(batch):

  mlp = hk.Sequential([
    hk.Conv3D(out_channels=60, kernel_shape=3, stride=1), jnp.nn.relu,
    hk.Conv3D(out_channels=60, kernel_shape=3, stride=1), jnp.nn.reul,
    hk.Conv3D(out_channels=60, kernel_shape=3, stride=1), jnp.nn.reul,
    hk.Conv3D(out_channels=40, kernel_shape=3, stride=1), jnp.nn.reul,
    hk.Conv3D(out_channels=20, kernel_shape=3, stride=1), jnp.nn.relu,
    hk.Conv3D(out_channels=20, kernel_shape=3, stride=1), jnp.nn.relu,
    hk.Reshape(output_shape=(-1)),
    hk.Linear(400), jnp.nn.relu,
    hk.Linear(300), jnp.nn.relu,
    hk.Linear(200), jnp.nn.relu,
    hk.Linear(100), jnp.nn.relu,
    hk.Linear(50), jnp.nn.relu,
    hk.Linear(40), jnp.nn.relu,
    hk.Linear(20), jnp.nn.relu,
    hk.Linear(10), jnp.nn.relu,
    hk.Linear(5)
  ])

  return mlp(batch)


def compute_loss(params: hk.Params, graph: jraph.GraphsTuple, label: jnp.ndarray,
                 net: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes loss and accuracy."""
  logits = net.apply(params, batch[:,:-1])
  labels = jax.nn.one_hot(batch[:,-1], 3)

  l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
  softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
  softmax_xent /= labels.shape[0]

  loss = softmax_xent + 1e-4 * l2_loss
  accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == batch[:,-1])
  
  return loss, accuracy

def prep_data():
  return 0

def prep_label():
  return 0

def evaluate(dataset: List[Any],
             dataLabels: List[Any],
             params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Evaluation Script."""
  # Transform impure `net_fn` to pure functions with hk.transform.
  net = hk.without_apply_rng(hk.transform(net_fn))
  # Get a candidate graph and label to initialize the network.

  #This should get a timer statement too
  #print(dataset)
  graph = dataset[0]
  accumulated_loss = 0
  accumulated_accuracy = 0
  compute_loss_fn = jax.jit(functools.partial(compute_loss, net=net))
  for idx in range(len(dataLabels)):
    graph = dataset[idx]#graph
    label = dataLabels[idx]#label
    graph = pad_graph_to_nearest_power_of_two(graph)
    label = jnp.concatenate((jnp.array([label]), jnp.array([0])))
    loss, acc = compute_loss_fn(params, graph, label)
    accumulated_accuracy += acc
    accumulated_loss += loss
  print('Completed evaluation.')
  loss = accumulated_loss / idx
  accuracy = accumulated_accuracy / idx
  print(f'Eval loss: {loss}, accuracy {accuracy}')
  return loss, accuracy


def implantDicInArrWMask(dic, array, global_points):
  #Doesn't work, but is indicative
  coords = trimesh.points.PointCloud(global_points).convex_hull.contains(np.indices((maxDim, maxDim, maxDim)))
  #SHould be coords which need a 1:
  array[coords, maxRep - 2] = 1#something like this
  for i in dic:
    array[dic, :] = i
  
  return array#NEEDS LOTS OF CHECKS



def main():
    print("initializing")
    #We first need to load all the data.

    with open("big-baddest-dataset.pyobj", "rb") as file:
      obj = pickle.load(file)

    print("loaded")

    #Then, we loop over the data a couple of times with periodic evals.
    #The data needs to be shuffled.
    graphs, labels = (obj[0], obj[1])#unison_shuffled_copies(obj[0], obj[1])

    net = hk.without_apply_rng(hk.transform(net_fn))
    # Initialize the network.
    params = net.init(jax.random.PRNGKey(42), graphs[0])

    #modify this to look for a current parameter set.
    if exists("model-dump-5.params"):
      with open('model-dump-5.params', 'rb') as file:
        params = pickle.load(file)

    # Initialize the optimizer.
    opt_init, opt_update = optax.adam(1e-4)
    opt_state = opt_init(params)

    compute_loss_fn = functools.partial(compute_loss, net=net)
    # We jit the computation of our loss, since this is the main computation.
    # Using jax.jit means that we will use a single accelerator. If you want
    # to use more than 1 accelerator, use jax.pmap. More information can be
    # found in the jax documentation.
    compute_loss_fn = jax.jit(jax.value_and_grad(
        compute_loss_fn, has_aux=True))

    num_train_steps = 150000


    testGraphs = graphs[:(math.floor(len(labels) / 10))]
    testLabels = labels[:(math.floor(len(labels) / 10))]
    trainingGraphs = graphs[math.floor(len(labels) / 10):]
    trainingLabels = labels[math.floor(len(labels) / 10):]

    print("starting training")
    batchSize = 20
    num = len(trainingLabels) - batchSize

    #I try to fix the performance issue by reshaping the list ahead of time, then measuring its in-app performance
    #Time the prep stage

    print("point 1")

    accumGraph = [0]*math.floor(num / batchSize)
    accumLabel = [0]*math.floor(num / batchSize)

    for i in range(math.floor(num / batchSize)):
      accumGraph[i] = trainingGraphs[batchSize * i:batchSize * (i+1)]
      accumLabel[i] = trainingLabels[batchSize * i:batchSize * (i+1)]

    print("point 2")

    try:
        for idy in range(num_train_steps):

            #TODO: determined manually, dedicate a thread to processing data, and another thread to feeding it to the GPU:

            temp_expand = map(lambda g: ,batchedGraph[idy % len(batchedLabel)])

            

            #t0 = time.time()
            (loss, acc), grad = compute_loss_fn(params, batchedGraph[idy % len(batchedLabel)], batchedLabel[idy % len(batchedLabel)])

            updates, opt_state = opt_update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            #t2 = time.time()
            #print("time two", t2 - t0)
            if idy % 100 == 0:
                print(f'step: {idy}, loss: {loss}, acc: {acc}')
            if idy % 2000 == 999:
                evaluate(testGraphs, testLabels, params)
    except KeyboardInterrupt:
        file = open('model-dump-5.params', 'wb')
        pickle.dump(params, file)
        file.close()
        print("Failsafe")
        exit()

    evaluate(testGraphs, testLabels, params)
    print('Training finished')

    #If training is finished, just dump everything, so I can use it again, or start from the new stuff.

    file = open('trained-model-5.params', 'wb')
    pickle.dump(params, file)
    file.close()
    print("Bye")
    exit()

if __name__ == "__main__":
    main()
