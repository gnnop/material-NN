from _common_ml import *
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

#I'm taking in the label separate to speed up the array creation
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    index_shuf = list(range(len(a)))
    shuffle(index_shuf)
    return  [a[i] for i in index_shuf],  [b[i] for i in index_shuf]


# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/_src/models.py#L506
def GraphConvolution(update_node_fn: Callable,
                     aggregate_nodes_fn: Callable = jax.ops.segment_sum,
                     add_self_edges: bool = False,
                     symmetric_normalization: bool = True) -> Callable:
  """Returns a method that applies a Graph Convolution layer.

  Graph Convolutional layer as in https://arxiv.org/abs/1609.02907,
  NOTE: This implementation does not add an activation after aggregation.
  If you are stacking layers, you may want to add an activation between
  each layer.
  Args:
    update_node_fn: function used to update the nodes. In the paper a single
      layer MLP is used.
    aggregate_nodes_fn: function used to aggregates the sender nodes.
    add_self_edges: whether to add self edges to nodes in the graph as in the
      paper definition of GCN. Defaults to False.
    symmetric_normalization: whether to use symmetric normalization. Defaults to
      True.

  Returns:
    A method that applies a Graph Convolution layer.
  """

  def _ApplyGCN(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Applies a Graph Convolution layer."""
    nodes, _, receivers, senders, _, _, _ = graph

    # First pass nodes through the node updater.
    nodes = update_node_fn(nodes)
    # Equivalent to jnp.sum(n_node), but jittable
    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
    #In the original example, self-edges were included. 
    #based on how I arranged the data, it shouldn't be necessary.

    conv_senders = senders
    conv_receivers = receivers

    # pylint: disable=g-long-lambda
    if symmetric_normalization:
      # Calculate the normalization values.
      count_edges = lambda x: jax.ops.segment_sum(
          jnp.ones_like(conv_senders), x, total_num_nodes)
      sender_degree = count_edges(conv_senders)
      receiver_degree = count_edges(conv_receivers)

      # Pre normalize by sqrt sender degree.
      # Avoid dividing by 0 by taking maximum of (degree, 1).
      nodes = tree.tree_map(
          lambda x: x * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
          nodes,
      )
      # Aggregate the pre-normalized nodes.
      nodes = tree.tree_map(
          lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                       total_num_nodes), nodes)
      # Post normalize by sqrt receiver degree.
      # Avoid dividing by 0 by taking maximum of (degree, 1).
      nodes = tree.tree_map(
          lambda x:
          (x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None]),
          nodes,
      )
    else:
      nodes = tree.tree_map(
          lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                       total_num_nodes), nodes)
    # pylint: enable=g-long-lambda
    return graph._replace(nodes=nodes)

  return _ApplyGCN


# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
def _nearest_bigger_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than x for padding."""
  y = 2
  while y < x:
    y *= 2
  return y

def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Pads a batched `GraphsTuple` to the nearest power of two.
  For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
  would pad the `GraphsTuple` nodes and edges:
    7 nodes --> 8 nodes (2^3)
    5 edges --> 8 edges (2^3)
  And since padding is accomplished using `jraph.pad_with_graphs`, an extra
  graph and node is added:
    8 nodes --> 9 nodes
    3 graphs --> 4 graphs
  Args:
    graphs_tuple: a batched `GraphsTuple` (can be batch size 1).
  Returns:
    A graphs_tuple batched to the nearest power of two.
  """
  # Add 1 since we need at least one padding node for pad_with_graphs.
  pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
  pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
  return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                               pad_graphs_to)

# Adapted from https://github.com/deepmind/jraph/blob/master/jraph/ogb_examples/train.py
@jraph.concatenated_args
def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Edge update function for graph net."""
  net = hk.Sequential(
      [
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(128)])
  return net(feats)

@jraph.concatenated_args
def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Node update function for graph net."""
  net = hk.Sequential(
      [
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(128)])
  return net(feats)

@jraph.concatenated_args
def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Global update function for graph net."""
  # MUTAG is a binary classification task, so output pos neg logits.
  net = hk.Sequential(
      [
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(256), jax.nn.relu,
       hk.Linear(128)])
  return net(feats)

def net_fn(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
  global globals

  #The shape of the graph is wrong.
  #The graph globals structure relies on the globals being in
  #a matrix. Uhhhh

  collector = jraph.GraphMapFeatures(
    hk.Sequential(
      [
      hk.Linear(128)]),
    hk.Sequential(
      [
      hk.Linear(128)]),
    hk.Sequential(
      [
      hk.Linear(512), jax.nn.relu,
      hk.Linear(512), jax.nn.relu,
      hk.Linear(512), jax.nn.relu,
      hk.Linear(256), jax.nn.relu,
      hk.Linear(128), jax.nn.relu,
      hk.Linear(globals["labelSize"])]))

  embedder = jraph.GraphMapFeatures(
      hk.Sequential(
      [
       hk.Linear(256), jax.nn.relu,
       hk.Linear(512), jax.nn.relu,
       hk.Linear(512)]), 
      hk.Sequential(
      [
       hk.Linear(256), jax.nn.relu,
       hk.Linear(512), jax.nn.relu,
       hk.Linear(512), jax.nn.relu,
       hk.Linear(512), jax.nn.relu,
       hk.Linear(512), jax.nn.relu,
       hk.Linear(512)]), 
      hk.Sequential(
      [
       hk.Linear(256), jax.nn.relu,
       hk.Linear(512), jax.nn.relu,
       hk.Linear(512), jax.nn.relu,
       hk.Linear(512), jax.nn.relu,
       hk.Linear(512), jax.nn.relu,
       hk.Linear(512)]))
  net = jraph.GraphNetwork(
      update_node_fn=node_update_fn,
      update_edge_fn=edge_update_fn,
      update_global_fn=update_global_fn)
  
  print(graph.edges.shape)
  print(graph.globals.shape)
  print(graph.n_edge.shape)
  print(graph.n_node.shape)
  print(graph.receivers.shape)
  print(graph.senders.shape)
  print("graph minor")
  x1 = embedder(graph)
  print(x1.edges.shape)
  print(x1.globals.shape)
  print(x1.n_edge.shape)
  print(x1.n_node.shape)
  print(x1.receivers.shape)
  print(x1.senders.shape)
  print("graph proper")
  x2 = net(x1)#something's wrong this line
  print(x2)
  x3 = collector(x2)

  #return graph
  return x3


def compute_loss(params: hk.Params, graph: jraph.GraphsTuple, label: jnp.ndarray,
                 net: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
  global globals
  """Computes loss and accuracy."""
  pred_graph = net.apply(params, graph)
  #I need to check the shape of this to determine if the network is processing something

  preds = jax.nn.log_softmax(pred_graph.globals)

  # Since we have an extra 'dummy' graph in our batch due to padding, we want
  # to mask out any loss associated with the dummy graph.
  # Since we padded with `pad_with_graphs` we can recover the mask by using
  # get_graph_padding_mask.
  mask = jraph.get_graph_padding_mask(pred_graph)

  # Cross entropy loss.
  loss = -jnp.mean(preds * label * mask[:, None])

  # Accuracy taking into account the mask.
  accuracy = jnp.sum(
      (jnp.argmax(pred_graph.globals, axis=1) == jnp.argmax(label, axis=1)) * mask) / jnp.sum(mask)
  return loss, accuracy

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





def main(obj):
    print("initializing")

    #we loop over the data a couple of times with periodic evals.
    #The data needs to be shuffled.
    graphs, labels = (obj[0], obj[1])#unison_shuffled_copies(obj[0], obj[1])

    #graphs = graphs[0:500]
    #labels = labels[0:500]

    net = hk.without_apply_rng(hk.transform(net_fn))
    # Initialize the network.

    params = net.init(jax.random.PRNGKey(42), graphs[0])

    if exists("cgnn.params"):
      with open('cgnn.params', 'rb') as file:
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


    testGraphs = graphs[:(math.floor(len(labels) / 75))]
    testLabels = labels[:(math.floor(len(labels) / 75))]
    trainingGraphs = graphs[math.floor(len(labels) / 75):]
    trainingLabels = labels[math.floor(len(labels) / 75):]

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

    #parallelize
    pool = ThreadPool(20)
    batchedGraph = pool.map(lambda a: pad_graph_to_nearest_power_of_two(jraph.batch(a)), accumGraph)
    print("out")
    batchedLabel = pool.map(lambda a: jnp.concatenate((jnp.array(a), jnp.zeros((1,globals["labelSize"])))), accumLabel)

    print(batchedLabel)
    print("point 2")

    try:
        for idy in range(num_train_steps):

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

cmd_line(main, "cgnn")