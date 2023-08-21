import numpy as np
from ccnn_config import *

"""
The atom is taken to be represented in the Direct notation in terms of the basis vectors.

It is not a direct Cartesian representation
"""


#Converts physical position to voxel position.
def atomToArray(position, axes):
    basePoint = centre - conversionFactor * ((axes[0] + axes[1] + axes[2]) / 2)
    return conversionFactor * np.matmul(position, axes) + basePoint#Is this right?

#Converts voxel position to physical position.
def arrayToAtom(position, axes, axesInv):
    basePoint = centre - conversionFactor * ((axes[0] + axes[1] + axes[2]) / 2)
    return np.matmul((position - basePoint) / conversionFactor, axesInv)