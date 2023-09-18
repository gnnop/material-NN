from _common_data_preprocessing import *
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
import itertools
import time
import sys, os
sys.path.append( os.curdir )
from ccnn_config import *
from ccnn_shared import *
import concurrent.futures

def givenPointDetermineCubeAndOverlap(position):
    index = np.round(position).astype(int)
    comp = lambda i, p : -1 if p < i else 1
    indices = list(itertools.product(*[[0, comp(index[i], position[i])] for i in range(3)]))
    points = [tuple(index + np.array(indices[i])) for i in range(8)]
    shared_vol = [np.prod( 1 - np.abs(position - i)) for i in points]
    return (points, shared_vol)

def net_fn(batch):

  #Literally insane. This is still too large
  cnet = hk.Sequential([
    hk.Conv3D(output_channels=40, kernel_shape=3, stride=1), jax.nn.relu,
    hk.Conv3D(output_channels=40, kernel_shape=3, stride=1), jax.nn.relu,
    hk.Conv3D(output_channels=40, kernel_shape=3, stride=1), jax.nn.relu,
    hk.Conv3D(output_channels=30, kernel_shape=3, stride=1), jax.nn.relu,
    hk.Conv3D(output_channels=15, kernel_shape=3, stride=1), jax.nn.relu])
  
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

  #l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
  #Glitch here.
  softmax_xent = -jnp.sum(label * jax.nn.log_softmax(logits))
  softmax_xent /= label.shape[0]

  loss = softmax_xent# + 1e-4 * l2_loss
  successes = jnp.argmax(logits, axis=1) == jnp.argmax(label, axis=1)
  accuracy = jnp.average(successes)

  return loss, accuracy


cube_points = np.array([[0,0,0],[1,1,1], 
                        [0, 0, 1],[0, 1, 0],[1, 0, 0],
                        [1, 1, 0],[1, 0, 1],[0, 1, 1]])

#Uses mutability  @jax.jit
def testConvexity(index, axes, invAxes):
  a = np.apply_along_axis(lambda r: arrayToAtom(r, axes, invAxes), 0, index)
  b = (0.0 < a) & (a < 1.0)
  c = np.all(b, axis=0)
  return c if 1 else 0


def prep_data(row):
  #atoms
  denseEncode = np.zeros((maxDims, maxDims, maxDims, maxRep))
  
  #atomic representation
  poscar = list(map(lambda a: a.strip(), row[0].split("\\n")))
  #this line is non-deterministic and you can tell that things will go south.
  #axes = randomRotateBasis(getGlobalDataVector(poscar))
  axes = getGlobalDataVector(poscar)

  atoms = poscar[5].split()
  numbs = poscar[6].split()

  total = 0
  for i in range(len(numbs)):
    total+=int(numbs[i])
    numbs[i] = total
  
  curIndx = 0
  atomType = 0
  for i in range(total):
    curIndx+=1
    if curIndx > numbs[atomType]:
      atomType+=1

    for j in completeTernary:
      #This tiles out everything. Then, I dither the pixels or whatevery
      points, vol = givenPointDetermineCubeAndOverlap(atomToArray(unpackLine(poscar[8+i]) + np.array(j), axes))
      for jj in range(len(points)):
        #Additional logical check. Points need to be in the ranges specified
        #It's expected that some points be outside of bounds since this is the tesselated space
        if -1 < points[jj][0] < maxDims and -1 < points[jj][1] < maxDims and -1 < points[jj][2] < maxDims and points[jj][0] >= 0 and points[jj][1] >= 0 and points[jj][2] >= 0:
          denseEncode[points[jj][0],points[jj][1],points[jj][2], :] = [vol[jj], *serializeAtom(atoms[atomType], poscar, i)]
  
  #mask over primitive cell
  invAxes = np.linalg.inv(axes)
  mask = np.fromfunction(lambda i, j, k, l: testConvexity(np.array((i,j,k)), axes, invAxes), 
                   (maxDims, maxDims, maxDims, 1))

  #Finally, fill in the tiled global values:
  #get symm back. I'm trying to make this similar to the other ML programs.
  #However, because I'm now expanding things on the fly, I need to use the symmetry and this is the least invasive
  #way to do that:
  if globals["dataSize"] == 14:
    sym = "f"
  elif globals["dataSize"] == 38:
    sym = "c"
  else:
    sym = ""
  
  globs = np.tile(np.array(getGlobalData(poscar, row, sym)), ((maxDims, maxDims, maxDims, 1)))

  space = np.concatenate((denseEncode, mask, globs), axis=-1)

  return space

def prep_label(lab):
  #This is already parsed by preprocessing
  return lab

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
    temp_objs = np.array(list(map(prep_data, [obj])))
    temp_labels = jnp.array([label])
    # loss, acc = compute_loss(params, temp_objs, temp_labels, net=net)
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

    accumChunks = [0]*math.floor(num / batchSize)

    for i in range(math.floor(num / batchSize)):
      accumChunks[i] = (trainingObjects[batchSize * i:batchSize * (i+1)], 
                        jnp.array(trainingLabels[batchSize * i:batchSize * (i+1)]))

    print("point 2")

    #The following functions are wrappers for the multiprocessing modules to play with:
    def preprocess_chunk(data_obs):
      return np.array(list(pool.map(prep_data, data_obs[0]))), data_obs[1]
    
    #must be called on single thread!!!
    def final_process_data(data_objs, ii):
      nonlocal opt_state
      nonlocal params
      nonlocal testObjects
      nonlocal testLabels
      (loss, acc), grad = compute_loss_fn(params, data_objs[0], data_objs[1])
      updates, opt_state = opt_update(grad, opt_state, params)
      params = optax.apply_updates(params, updates)

      if ii % 1 == 0:
        print(f'step: {ii}, loss: {loss}, acc: {acc}')
      #evaluate needs the same type of speed-up to be reasonable
      #if ii > 0 and ii % 100 == 0:
        #evaluate(testObjects, testLabels, params)

    try:
      """
      The following is getting rearranged as a multiprocessed array. 
      Testing takes a while so don't do it too frequently
      It's dif
      """
        
      with concurrent.futures.ThreadPoolExecutor(max_workers=20) as preprocessing_executor:
        processed_chunks = 0

        preprocess_future = [preprocessing_executor.submit(preprocess_chunk, chunk) for chunk in accumChunks]

        print("preprocessors scheduled")

        #Potentially bad if the preprocessing starts outpacing the ML...
        for future in concurrent.futures.as_completed(preprocess_future):
          preprocessed_batch = future.result()
          final_process_data(preprocessed_batch, processed_chunks)
          processed_chunks += 1

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