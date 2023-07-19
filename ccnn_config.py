import numpy as np

"""
Not synced up. Make changes at the same time
"""
maxDims = 16#Number of cells 60 atom max. cubic root is 4. *2 for space =8, *2.5 for tesselation is 20 *2 (arbitrary) for 40-1.6MB
conversionFactor = 2.7#Always scale the maxDims with the conversionFactor
#need to be able to rep atoms, probably have 3* max unit cell
maxRep = 7 + 16 + 1 + 3 + 1 #3 - atomic distance, 1 - unit cell mask
dims = (maxDims, maxDims, maxDims, maxRep)
centre = np.array([maxDims / 2, maxDims / 2, maxDims / 2])
