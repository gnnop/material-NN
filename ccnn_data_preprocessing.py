import csv
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
import itertools
from _common_data_preprocessing import *
import pickle
import sys, os
sys.path.append( os.curdir )
from ccnn_config import *

#Work on minifying data
#The atom is modeled as a width one box. The points are modelled as width 1 boxes centered at the thing.
#We determine rough inclusion based on overlap
#the position should be in terms of the indices
def givenPointDetermineCubeAndOverlap(position):
    index = np.round(position).astype(int)
    comp = lambda i, p : -1 if p < i else 1
    indices = list(itertools.product(*[[0, comp(index[i], position[i])] for i in range(3)]))
    points = [tuple(index + np.array(indices[i])) for i in range(8)]
    shared_vol = [np.prod( 1 - np.abs(position - i)) for i in points]
    return (points, shared_vol)

#already sampled as a thingy
def randomRotateBasis(vecOfVecs):
	rot = R.random().as_matrix()
	return np.matmul(rot, vecOfVecs)


#Converts stuff to stuff.
def atomToArray(position, axes):
    basePoint = centre - conversionFactor * ((axes[0] + axes[1] + axes[2]) / 2)
    return conversionFactor * np.matmul(position, axes) + basePoint#Is this right?




def dataEncoder(row, sym):

    poscar = list(map(lambda a: a.strip(), row[0].split("\\n")))
    globalInfo = np.array(getGlobalData(poscar, row, sym))
    #this line is non-deterministic and you can tell that things will go south.
    #axes = randomRotateBasis(getGlobalDataVector(poscar))
    axes = getGlobalDataVector(poscar)

	#Choose a center to tile everything out with

    atoms = poscar[5].split()
    numbs = poscar[6].split()

    encoding = {}

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
        
        #individual atomic data here.

        for j in completeTernary:
            #This tiles out everything. Then, I dither the pixels or whatevery
            points, vol = givenPointDetermineCubeAndOverlap(atomToArray(np.dot(unpackLine(poscar[8+i]), axes) + np.dot(j, axes), axes))
            for jj in range(len(points)):
                #Additional logical check. Points need to be in the ranges specified:
                if points[jj][0] < maxDims and points[jj][1] < maxDims and points[jj][2] < maxDims and points[jj][0] >= 0 and points[jj][1] >= 0 and points[jj][2] >= 0:
                    if points[jj] not in encoding:
                        encoding[points[jj]] = (vol[jj], *serializeAtom(atoms[atomType], poscar, i))
                    else:
                        print(row)
                        print("Enlarge the encoding! It's too small")
                        exit()
    
    #Add in convex points deteremining unit cell, to deteremine the ones array
    #Need to return axes for reconstruction
    return (axes, (globalInfo, (list(encoding.keys()), list(encoding.values()))))

def format(read_file, write_file, sym, topo):
    with open(read_file, 'r', newline='') as file:
        reader = csv.reader(file)

        dataset = []
        datalabels = []

        ii = 0
        for row in reader:
            ii = ii + 1
            print(ii)
            dataset.append(dataEncoder(row, sym))
            datalabels.append(convertTopoToIndex(row, topo))
        
        os.makedirs(os.path.dirname(write_file), exist_ok=True)

        with open(write_file, 'wb') as file:
            pickle.dump((dataset, datalabels), file)

cmd_line(format, "ccnn")