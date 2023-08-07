import csv
from http.client import REQUEST_URI_TOO_LONG
import re
import numpy as np
import jax.numpy as jnp
import jraph
import pickle
import networkx as nx
from typing import Any, Dict, List
from multiprocessing.dummy import Pool as ThreadPool
from _common_data_preprocessing import *

from haiku import switch



#The following is from:
#https://colab.research.google.com/github/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb#scrollTo=e7q5ySSmVL3x
#This is not important. This is just a sanity check to make certain that the graph structures are
#lining up correctly.


def convert_jraph_to_networkx_graph(jraph_graph):
    nodes, edges, receivers, senders, _, _, _ = jraph_graph
    nx_graph = nx.DiGraph()
    if nodes is None:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n)
    else:
        for n in range(jraph_graph.n_node[0]):
            nx_graph.add_node(n, node_feature=nodes[n])
    if edges is None:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(int(senders[e]), int(receivers[e]))
    else:
        for e in range(jraph_graph.n_edge[0]):
            nx_graph.add_edge(
            int(senders[e]), int(receivers[e]), edge_feature=edges[e])
    return nx_graph


def draw_jraph_graph_structure(jraph_graph):
    nx_graph = convert_jraph_to_networkx_graph(jraph_graph)
    pos = nx.spring_layout(nx_graph)
    nx.draw(
        nx_graph, pos=pos, with_labels=True, node_size=500, font_color='yellow')


#The complicated part is that we need to compute the minimum distance, so I'm including the
#distance between the atom and the one in the next cell. It is unclear to me whether the encoding
#requires that the edges be identified.

def reflectAtomDist(lis):
    return map(lambda a: [a[0], -a[1], -a[2], -a[3]], lis)


def atomDistance(params, loc1, loc2):
    global completeTernary
    maxBondDistance = 3.1 #this seems to be reasonable, but the min problem is what to do with multiple bonding.
    #I found a paper where they just ignore the problem, so that's the first approach here;

    dist = []
    
    for i in completeTernary:
        dir = np.array(loc1) - np.array(loc2) + i[0]*np.array(params[0]) + i[1]*np.array(params[1]) + i[2]*np.array(params[2])
        if np.linalg.norm(dir) < maxBondDistance:
            dist.append([np.linalg.norm(dir),*dir])
    
    #Do I need the subtraction info?
    return dist




tots = 0

def extractGraph(row, sym):
    global globals
    global tots

    tots += 1
    if tots % 100 == 0:
        print(tots)
    
    str = row[0]

    str = re.sub(r"\s+", "", str)
    poscar = list(map(lambda a: a.strip(), row[0].split("\\n")))

    globalData = [0.0]*(globals["dataSize"] + globals["labelSize"])
    globalData[0:globals["dataSize"]] = getGlobalData(poscar, row, sym)


    atoms = poscar[5].split()
    numbs = poscar[6].split()

    total = 0
    for i in range(len(numbs)):
        total+=int(numbs[i])
        numbs[i] = total

    
    nodesArray = []

    curIndx = 0
    atomType = 0
    for i in range(total):
        curIndx+=1
        if curIndx > numbs[atomType]:
            atomType+=1
        
        #We have the whole representation right now - after this I add it into the graph for clarity
        nodesArray.append(serializeAtom(atoms[atomType], poscar, i))
    
    #Now we do an n^2 operation on the atoms:

    senderArray = []
    receiverArray = []
    edgeFeatures = []

    for i in range(len(nodesArray)):
        for j in range(i + 1):
            ls = atomDistance(getGlobalDataVector(poscar), nodesArray[i][0:3], nodesArray[j][0:3])
            #This is a list of
            senderArray.extend([i]*len(ls))
            receiverArray.extend([j]*len(ls))
            edgeFeatures.extend(ls)
            if i != j:
                senderArray.extend([j]*len(ls))
                receiverArray.extend([j]*len(ls))
                edgeFeatures.extend(reflectAtomDist(ls))

    #everything should be lined up now, but we need to add it to a graph

    return jraph.GraphsTuple(
        nodes=jnp.array(nodesArray), 
        senders=jnp.array(senderArray), 
        receivers=jnp.array(receiverArray), 
        edges=jnp.array(edgeFeatures), 
        globals=jnp.array([globalData]),
        n_node=jnp.array([len(nodesArray)]),
        n_edge=jnp.array([len(senderArray)]))

#This isn't writing the predictions yet. I'm going to dump those in a separate file for ease of thought

def format(read_obj, write_obj, sym, topo):
    with open(read_obj, 'r', newline='') as file:
        reader = csv.reader(file)
        totalFiles = list(reader)

        pool = ThreadPool(20)
        graphs = pool.map(lambda g : extractGraph(g, sym), totalFiles)
        labels = pool.map( lambda row: convertTopoToIndex(row, topo), totalFiles)
        with open(write_obj, 'wb') as file:
            print(graphs[0].globals.shape)
            pickle.dump([graphs, labels], file)


cmd_line(format, "cgnn")