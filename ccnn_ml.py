from _common_ml import *
import functools
import jax
import jax.numpy as jnp
from os.path import exists
import haiku as hk
import math
import optax
import pickle
import numpy as np
from typing import Any, List, Tuple
import time
from multiprocessing.dummy import Pool as ThreadPool
import jax.numpy as jnp
import haiku as hk
import trimesh
import time
import sys, os
sys.path.append( os.curdir )
from ccnn_config import *

#Converts stuff to stuff.
def atomToArray(position, axes):
    basePoint = centre - conversionFactor * ((axes[0] + axes[1] + axes[2]) / 2)
    return conversionFactor * np.matmul(position, axes) + basePoint#Is this right?

def net_fn(batch):

  #Literally insane. This is still too large
  cnet = hk.Sequential([
    hk.Conv3D(output_channels=60, kernel_shape=3, stride=1), jax.nn.relu,
    hk.Conv3D(output_channels=60, kernel_shape=3, stride=1), jax.nn.relu,
    hk.Conv3D(output_channels=60, kernel_shape=3, stride=1), jax.nn.relu,
    hk.Conv3D(output_channels=40, kernel_shape=3, stride=1), jax.nn.relu,
    hk.Conv3D(output_channels=20, kernel_shape=3, stride=1), jax.nn.relu,
    hk.Conv3D(output_channels=5, kernel_shape=3, stride=1), jax.nn.relu])
  
  y1 = cnet(batch)

  #DANGER! The MLP conversion is the first number * y1.size. In a batch, this
  #easily exceeds single GPU VRAM with 62^3 * 4 * 20 * 20 * ? = 50 GB. I have 8GB, so reduce everything!

  mlp = hk.Sequential([
    hk.Flatten(),
    hk.Linear(40), jax.nn.relu,
    hk.Linear(30), jax.nn.relu,
    hk.Linear(20), jax.nn.relu,
    hk.Linear(10), jax.nn.relu,
    hk.Linear(globals["labelSize"])
  ])

  return mlp(y1)

#@jax.jit
def compute_loss(params, batch, label, net):
  """Computes loss and accuracy."""
  logits = net.apply(params, batch)

  l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
  #Glitch here.
  print(type(label))
  print(type(logits))
  softmax_xent = -jnp.sum(label * jax.nn.log_softmax(logits))
  softmax_xent /= label.shape[0]

  loss = softmax_xent + 1e-4 * l2_loss
  accuracy = jnp.average(
      (jnp.argmax(logits, axis=1) == jnp.argmax(label, axis=1)))
  return loss, accuracy


cube_points = np.array([[0,0,0],[1,1,1], 
                        [0, 0, 1],[0, 1, 0],[1, 0, 0],
                        [1, 1, 0],[1, 0, 1],[0, 1, 1]])

#Uses mutability  @jax.jit
def prep_data(dicnglobal):
  #this is formatted as: [axes, [global, dictionary]]
  space = np.zeros((maxDims, maxDims, maxDims, maxRep + dicnglobal[1][0].size))
  
  #We get the global points by putting in the corners of a box
  convexPoints = np.array([atomToArray(i, dicnglobal[0]) for i in cube_points])

  indices = np.transpose(np.indices((maxDims, maxDims, maxDims)).reshape(3, -1))
  
  coords = trimesh.points.PointCloud(convexPoints).convex_hull.contains(indices)
  
  # Convert coords to indices using numpy.where
  true_indices = indices[coords]

  # Assign 1 to space at the true_indices
  space[tuple(true_indices.T), maxRep - 2] = 1

  # The final step is to fill in the global values so I can use a uniform convolutional network:
  for idx, ls in zip(dicnglobal[1][1][0], dicnglobal[1][1][1]):
     space[idx[0], idx[1], idx[2], -maxRep:] = ls

  #Finally, fill in the tiled global values:
  globs = np.tile(dicnglobal[1][0], ((maxDims, maxDims, maxDims, 1)))

  space[:,:,:,-dicnglobal[1][0].size:] = globs
  return space

def prep_label():
  return 0

#@jax.jit
def evaluate(dataset: List[Any],
             dataLabels: List[Any],
             params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Evaluation Script."""
  # Transform impure `net_fn` to pure functions with hk.transform.
  net = hk.without_apply_rng(hk.transform(net_fn))
  # Get a candidate graph and label to initialize the network.

  #This should get a timer statement too
  #print(dataset)
  accumulated_loss = 0
  accumulated_accuracy = 0
  compute_loss_fn = jax.jit(functools.partial(compute_loss, net=net))
  for idx in range(len(dataLabels)):
    obj = dataset[idx]#graph
    label = dataLabels[idx]#label
    temp_objs = map(prep_data, [obj])
    temp_labels = jnp.array([label])
    loss, acc = compute_loss_fn(params, temp_objs, temp_labels)
    accumulated_accuracy += acc
    accumulated_loss += loss
  print('Completed evaluation.')
  loss = accumulated_loss / idx
  accuracy = accumulated_accuracy / idx
  print(f'Eval loss: {loss}, accuracy {accuracy}')
  return loss, accuracy

def main(obj):
    print("initializing")

    #test pool based multithreading later to see if there's
    #a speed up

    pool = ThreadPool(20)

    #Then, we loop over the data a couple of times with periodic evals.
    #The data needs to be shuffled.
    objs, labels = (obj[0], obj[1])#unison_shuffled_copies(obj[0], obj[1])

    net = hk.without_apply_rng(hk.transform(net_fn))

    print("pre init")
    # Initialize the network.
    params = net.init(jax.random.PRNGKey(42), np.expand_dims(prep_data(objs[0]), axis=0))

    print("initted")

    #modify this to look for a current parameter set.
    if exists("ccnn.params"):
      with open('ccnn.params', 'rb') as file:
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


    testObjects = objs[:(math.floor(len(labels) / 10))]
    testLabels = labels[:(math.floor(len(labels) / 10))]
    trainingObjects = objs[math.floor(len(labels) / 10):]
    trainingLabels = labels[math.floor(len(labels) / 10):]

    print("starting training")
    batchSize = 10
    num = len(trainingLabels) - batchSize

    #I try to fix the performance issue by reshaping the list ahead of time, then measuring its in-app performance
    #Time the prep stage

    print("point 1")

    accumObject = [0]*math.floor(num / batchSize)
    accumLabel = [0]*math.floor(num / batchSize)

    for i in range(math.floor(num / batchSize)):
      accumObject[i] = trainingObjects[batchSize * i:batchSize * (i+1)]
      accumLabel[i] = jnp.array(trainingLabels[batchSize * i:batchSize * (i+1)])

    print("point 2")

    try:
        for idy in range(num_train_steps):
            print(idy)

            #TODO: determined manually, dedicate a thread to processing data, and another thread to feeding it to the GPU:
            #convolutions are currently linearly expanded with each batch on the fly:

            start = time.time()
            temp_objs = np.array(list(pool.map(prep_data, accumObject[idy])))
            temp_labels = accumLabel[idy]
            end = time.time()
            print("Data preprocessing: " + str(end - start))

            #t0 = time.time()
            (loss, acc), grad = compute_loss_fn(params, temp_objs, temp_labels)

            updates, opt_state = opt_update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            end2 = time.time()
            print("NN application: " + str(end2 - end))

            #t2 = time.time()
            #print("time two", t2 - t0)
            if idy % 100 == 0:
                print(f'step: {idy}, loss: {loss}, acc: {acc}')
            if idy % 2000 == 999:
                evaluate(testObjects, testLabels, params)
    except KeyboardInterrupt:
        file = open('model-dump-5.params', 'wb')
        pickle.dump(params, file)
        file.close()
        print("Failsafe")
        exit()

    evaluate(testObjects, testLabels, params)
    print('Training finished')

    #If training is finished, just dump everything, so I can use it again, or start from the new stuff.

    file = open('trained-model-5.params', 'wb')
    pickle.dump(params, file)
    file.close()
    print("Bye")
    exit()

cmd_line(main, "ccnn")