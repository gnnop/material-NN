import numpy as np
from ccnn_config import *

#Converts stuff to stuff.
def atomToArray(position, axes):
    basePoint = centre - conversionFactor * ((axes[0] + axes[1] + axes[2]) / 2)
    return conversionFactor * np.matmul(position, axes) + basePoint#Is this right?

