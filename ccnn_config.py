import numpy as np


'''
The input vector is a 3D lattice of size maxDims x maxDims x maxDims of
atoms, represented as vectors of size maxRep.
'''
maxDims = 44

'''
It is assumed that all atoms have the same size.
This is the relative radius of each atom in lattice units.
Always scale the maxDims with the conversionFactor
'''
conversionFactor = 1.35


'''
The size of the vector that represents a single atom. This includes:
* row on the periodic table         7
* column on the periodic table / 2  16      <- the rows and columns are one-hot encoded
* column on the periodic table % 2  1
* normalized x, y, z coordinates    3
* the presence of an atom           1
'''
maxRep = 7 + 16 + 1 + 3 + 1 

'''
The input vector is a 3D lattice of size maxDims x maxDims x maxDims of
atoms, represented as vectors of size maxRep.
'''
dims = (maxDims, maxDims, maxDims, maxRep)

'''
a 3D vector that represents the center of the lattice in cell units,
equal to maxDims / 2
'''
centre = np.array([maxDims / 2, maxDims / 2, maxDims / 2])


