import numpy as np
import itertools
import re
import sys
import csv
import jax

#entry format:
"""
b'generated by phonopy\n
   1.0\n
   -4.2915000000000001    4.2915000000000001    4.4085000000000001\n
   4.2915000000000001   -4.2915000000000001    4.4085000000000001\n
   4.2915000000000001    4.2915000000000001   -4.4085000000000001\n
   O Na Si Be Al Cl\n
   12    4    4    1    1    1\n
   Direct\n
   0.7040000000000001  0.4237000000000000  0.0139000000000001\n
   0.4097999999999998  0.6900999999999999  0.9860999999999999\n
   0.3099000000000001  0.2959999999999999  0.7196999999999999\n
   0.5763000000000000  0.5902000000000002  0.2803000000000001\n
   0.6955000000000000  0.3126000000000000  0.3105000000000000\n
   0.0021000000000000  0.3850000000000000  0.6895000000000000\n
   0.6150000000000000  0.3045000000000000  0.6171000000000000\n
   0.6874000000000000  0.9979000000000000  0.3829000000000000\n
   0.2913999999999999  0.7146999999999999  0.2754999999999999\n
   0.4392000000000000  0.0159000000000000  0.7245000000000001\n
   0.9841000000000000  0.7086000000000001  0.4233000000000000\n
   0.2853000000000001  0.5608000000000000  0.5767000000000000\n
   0.9760000000000000  0.6214999999999999  0.9604999999999999\n
   0.6610000000000000  0.0155000000000000  0.0395000000000001\n
   0.9844999999999999  0.0240000000000000  0.6455000000000000\n
   0.3785000000000001  0.3390000000000000  0.3545000000000000\n
   0.5178000000000000  0.2508999999999999  0.7599000000000000\n
   0.4909999999999999  0.7579000000000000  0.2401000000000000\n
   0.2421000000000000  0.4822000000000000  0.7330999999999999\n
   0.7491000000000001  0.5090000000000001  0.2669000000000001\n
   0.7500000000000000  0.2500000000000000  0.5000000000000000\n
   0.2500000000000000  0.7500000000000000  0.5000000000000000\n
   0.0000000000000000  0.0000000000000000  0.0000000000000000\n'
"""

completeTernary = list(itertools.product([-1, 0, 1], repeat=3))


#to make this line up slightly better, I
#attach the f orbital stuff to the right side of the
#periodic table. This makes sense, since the cols
#aren't labelled
#also, I'm just going to chuck the bottom row
#of the table, no one wants those elements
def atomToCoords(name):
    if name == "H":
        return [0, 0, 0]
    elif name == "He":
        return [0, 0, 1]#different, i likey here
    elif name == "Li":
        return [1, 0, 0]
    elif name == "Be":
        return [1, 0, 1]
    elif name == "B":
        return [1, 6, 0]
    elif name == "C":
        return [1, 6, 1]
    elif name == "N":
        return [1, 7, 0]
    elif name == "O":
        return [1, 7, 1]
    elif name == "F":
        return [1, 8, 0]
    elif name == "Ne":
        return [1, 8, 1]
    elif name == "Na":
        return [2, 0, 0]
    elif name == "Mg":
        return [2, 0, 1]
    elif name == "Al":
        return [2, 6, 0]
    elif name == "Si":
        return [2, 6, 1]
    elif name == "P":
        return [2, 7, 0]
    elif name == "S":
        return [2, 7, 1]
    elif name == "Cl":
        return [2, 8, 0]
    elif name == "Ar":
        return [2, 8, 1]
    elif name == "K":
        return [3, 0, 0]
    elif name == "Ca":
        return [3, 0, 1]
    elif name == "Sc":
        return [3, 1, 0]
    elif name == "Ti":
        return [3, 1, 1]
    elif name == "V":
        return [3, 2, 0]
    elif name == "Cr":
        return [3, 2, 1]
    elif name == "Mn":
        return [3, 3, 0]
    elif name == "Fe":
        return [3, 3, 1]
    elif name == "Co":
        return [3, 4, 0]
    elif name == "Ni":
        return [3, 4, 1]
    elif name == "Cu":
        return [3, 5, 0]
    elif name == "Zn":
        return [3, 5, 1]
    elif name == "Ga":
        return [3, 6, 0]
    elif name == "Ge":
        return [3, 6, 1]
    elif name == "As":
        return [3, 7, 0]
    elif name == "Se":
        return [3, 7, 1]
    elif name == "Br":
        return [3, 8, 0]
    elif name == "Kr":
        return [3, 8, 1]
    elif name == "Rb":
        return [4, 0, 0]
    elif name == "Sr":
        return [4, 0, 1]
    elif name == "Y":
        return [4, 1, 0]
    elif name == "Zr":
        return [4, 1, 1]
    elif name == "Nb":
        return [4, 2, 0]
    elif name == "Mo":
        return [4, 2, 1]
    elif name == "Tc":
        return [4, 3, 0]
    elif name == "Ru":
        return [4, 3, 1]
    elif name == "Rh":
        return [4, 4, 0]
    elif name == "Pd":
        return [4, 4, 1]
    elif name == "Ag":
        return [4, 5, 0]
    elif name == "Cd":
        return [4, 5, 1]
    elif name == "In":
        return [4, 6, 0]
    elif name == "Sn":
        return [4, 6, 1]
    elif name == "Sb":
        return [4, 7, 0]
    elif name == "Te":
        return [4, 7, 1]
    elif name == "I":
        return [4, 8, 0]
    elif name == "Xe":
        return [4, 8, 1]
    elif name == "Cs":
        return [5, 0, 0]
    elif name == "Ba":
        return [5, 0, 1]
    elif name == "Lu":
        return [5, 1, 0]
    elif name == "Hf":
        return [5, 1, 1]
    elif name == "Ta":
        return [5, 2, 0]
    elif name == "W":
        return [5, 2, 1]
    elif name == "Re":
        return [5, 3, 0]
    elif name == "Os":
        return [5, 3, 1]
    elif name == "Ir":
        return [5, 4, 0]
    elif name == "Pt":
        return [5, 4, 1]
    elif name == "Au":
        return [5, 5, 0]
    elif name == "Hg":
        return [5, 5, 1]
    elif name == "Tl":
        return [5, 6, 0]
    elif name == "Pb":
        return [5, 6, 1]
    elif name == "Bi":
        return [5, 7, 0]
    elif name == "Po":
        return [5, 7, 1]
    elif name == "At":
        return [5, 8, 0]
    elif name == "Rn":
        return [5, 8, 1]
    elif name == "Fr":
        return [6, 0, 0]
    elif name == "Ra":
        return [6, 0, 1]
    elif name == "Lr":
        return [6, 1, 0]
    elif name == "Rf":
        return [6, 1, 1]
    elif name == "Db":
        return [6, 2, 0]
    elif name == "Sg":
        return [6, 2, 1]
    elif name == "Bh":
        return [6, 3, 0]
    elif name == "Hs":
        return [6, 3, 1]
    elif name == "Mt":
        return [6, 4, 0]
    elif name == "Ds":
        return [6, 4, 1]
    elif name == "Rg":
        return [6, 5, 0]
    elif name == "Cn":
        return [6, 5, 1]
    elif name == "Nh":
        return [6, 6, 0]
    elif name == "Fl":
        return [6, 6, 1]
    elif name == "Mc":
        return [6, 7, 0]
    elif name == "Lv":
        return [6, 7, 1]
    elif name == "Ts":
        return [6, 8, 0]
    elif name == "Og":
        return [6, 8, 1]
    elif name == "La":
        return [5, 9, 0]
    elif name == "Ce":
        return [5, 9, 1]
    elif name == "Pr":
        return [5, 10, 0]
    elif name == "Nd":
        return [5, 10, 1]
    elif name == "Pm":
        return [5, 11, 0]
    elif name == "Sm":
        return [5, 11, 1]
    elif name == "Eu":
        return [5, 12, 0]
    elif name == "Gd":
        return [5, 12, 1]
    elif name == "Tb":
        return [5, 13, 0]
    elif name == "Dy":
        return [5, 13, 1]
    elif name == "Ho":
        return [5, 14, 0]
    elif name == "Er":
        return [5, 14, 1]
    elif name == "Tm":
        return [5, 15, 0]
    elif name == "Yb":
        return [5, 15, 1]
    elif name == "Ac":
        return [6, 9, 0]
    elif name == "Th":
        return [6, 9, 1]
    elif name == "Pa":
        return [6, 10, 0]
    elif name == "U":
        return [6, 10, 1]
    elif name == "Np":
        return [6, 11, 0]
    elif name == "Pu":
        return [6, 11, 1]
    elif name == "Am":
        return [6, 12, 0]
    elif name == "Cm":
        return [6, 12, 1]
    elif name == "Bk":
        return [6, 13, 0]
    elif name == "Cf":
        return [6, 13, 1]
    elif name == "Es":
        return [6, 14, 0]
    elif name == "Fm":
        return [6, 14, 1]
    elif name == "Md":
        return [6, 15, 0]
    elif name == "No":
        return [6, 15, 1]
    else:
        print(name)
        exit()

def serializeAtom(atom, poscar, index):
    atomRep = atomToCoords(atom)
    rw = [0.0]*7
    cl = [0.0]*16
    sp = [0.0]*1
    rw[atomRep[0]] = 1.0
    cl[atomRep[1]] = 1.0
    sp[0] = atomRep[2]#The spin is a single bit, so we drop it in one thing
    return [*unpackLine(poscar[8+index]), *cl, *rw, *sp]

def serializeSpaceGroup(num):
    if 1<=num<=1:
        return 0
    elif 2<=num<=2:
        return 1
    elif 3<=num<=5:
        return 2
    elif 6<=num<=9:
        return 3
    elif 10<=num<=15:
        return 4
    elif 16<=num<=24:
        return 5
    elif 25<=num<=46:
        return 6
    elif 47<=num<=74:
        return 7
    elif 75<=num<=80:
        return 8
    elif 81<=num<=82:
        return 9
    elif 83<=num<=88:
        return 10
    elif 89<=num<=98:
        return 11
    elif 99<=num<=110:
        return 12
    elif 111<=num<=122:
        return 13
    elif 123<=num<=142:
        return 14
    elif 143<=num<=146:
        return 15
    elif 147<=num<=148:
        return 16
    elif 149<=num<=155:
        return 17
    elif 156<=num<=161:
        return 18
    elif 162<=num<=167:
        return 19
    elif 168<=num<=173:
        return 20
    elif 174<=num<=174:
        return 21
    elif 175<=num<=176:
        return 22
    elif 177<=num<=182:
        return 23
    elif 183<=num<=186:
        return 24
    elif 187<=num<=190:
        return 25
    elif 191<=num<=194:
        return 26
    elif 195<=num<=199:
        return 27
    elif 200<=num<=206:
        return 28
    elif 207<=num<=214:
        return 29
    elif 215<=num<=220:
        return 30
    elif 221<=num<=230:
        return 31
    else:
        print(num)
        exit()

#We have the following data flow:
#SM:
#  ES
#  ESFD
#TI:
#  NLC
#  SEBR
#trivial:
#  LCEBR
#other - error

def convertTopoToIndex(rows, topo):
    if topo == "f":
        arr = [0.0]*5
        r = rows[5].strip()
        if r == "ES":
            arr[0] = 1.0
        elif r == "ESFD":
            arr[1] = 1.0
        elif r == "NLC":
            arr[2] = 1.0
        elif r == "SEBR":
            arr[3] = 1.0
        elif r == "LCEBR":
            arr[4] = 1.0
    else:
        arr = [0.0]*3
        r = rows[4].strip();
        if r == "trivial":
            arr[0] = 1.0
        elif r == "TI":
            arr[1] = 1.0
        elif r == "SM":
            arr[2] = 1.0
    return arr

maxSize = 60
atomSize = 27


def unpackLine(str):
    x = str.split()
    return list(map(float, x))


def getGlobalData(poscar, row, spec_str):
    a = np.array(unpackLine(poscar[2]))
    b = np.array(unpackLine(poscar[3]))
    c = np.array(unpackLine(poscar[4]))
    alpha = np.arccos(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c))) / np.pi
    beta = np.arccos(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))) / np.pi
    gamma = np.arccos(np.dot(b, a) / (np.linalg.norm(b) * np.linalg.norm(a))) / np.pi

    if spec_str == "f":
        return [np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c), alpha, beta, gamma,
                *np.unpackbits(np.array([int(row[2].strip())],dtype=np.uint8))] #6 + log(128) = 13 +1 because there's a useless bit
    elif spec_str == "c":
        return [np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c), alpha, beta, gamma, 
            *jax.nn.one_hot(serializeSpaceGroup(int(row[2].strip())), 32)] #6 + 32 = 38
    else:
        return [np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c), alpha, beta, gamma] #6

def getGlobalDataVector(poscar):
    return np.array([unpackLine(poscar[2]), unpackLine(poscar[3]), unpackLine(poscar[4])])

globals = {
    "dataSize" : 0,
    "labelSize" : 0
}

def print_usage():
    print('''

Usage: %s input_file_name symmetry_type topology_type
          
input_file_name: input file name (without extension, must be csv)
    examples: tqc-filtered, tqc-mini, tqc-full
          
symmetry_type: how to classify crystal symmetry by its space group.
    must be one of the following:
        f (full)       : binary representation of the space group number
        c (compressed) : one-hot encoding of the space group type
        n (nan)        : don't include symmetries at all

topology_type: how to classify crystals by their topological quantum chemistry.
    must be one of the following:
        f (full)       : use five classes:
            LCEBR (linear combination EBR)
            SEBR  (Split EBR)
            NLC   (nonlinear crystal)
            ES    (enforced semimetals)
            ESFD  (enforced semimetals with Fermi degeneracy)
        c (compressed) : use three classes:
            LCEBR (linear combination EBR)
            TI    (topological insulator: SEBR or NLC)
            SM    (semimetal: ES or ESFD)
            

'''% sys.argv[0])
# end print_usage


def cmd_line(func, name):
    global globals

    # sanity check: make sure we have the right number of arguments
    if len(sys.argv) != 4:
        print("Arguments:")
        print(sys.argv)
        print("Expected four arguments, but only %d were given" % (len(sys.argv)))
        print_usage()
        return
    # end if
    
    # sanity check: make sure symmetry and topology are valid
    if not (sys.argv[2] == "f" or sys.argv[2] == "c" or sys.argv[2] == "n"):
        print("symmetry not specified, use f, c, n : full, compressed, nan")
        print_usage()
        exit()
    if not (sys.argv[3] == "f" or sys.argv[3] == "c"):
        print("topology not specified, use f, c : full, compressed")
        print_usage()
        exit()
    
    # set globals
    # symmetry
    if sys.argv[2] == "f":
        globals["dataSize"] = 14
    elif sys.argv[2] == "c":
        globals["dataSize"] = 38
    else:
        globals["dataSize"] = 6
    # topology
    if sys.argv[3] == "f":
        globals["labelSize"] = 5
    else:
        globals["labelSize"] = 3

    # run the function with the given parameters
    func(sys.argv[1] + ".csv", "./data/" + sys.argv[1] + "-" + sys.argv[2] + "-" + sys.argv[3] + "-" + name +"-data.obj", sys.argv[2], sys.argv[3])
# end cmd_line
