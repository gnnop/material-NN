import sys
import pickle

# to make this line up slightly better, I
# attach the f orbital stuff to the right side of the
# periodic table. This makes sense, since the cols
# aren't labelled
# also, I'm just going to chuck the bottom row
# of the table, no one wants those elements


def atomToCoords(name):
    match name:
        case "H":
            return [0, 0, 0]
        case "He":
            return [0, 0, 1]  # different, i likey here
        case "Li":
            return [1, 0, 0]
        case "Be":
            return [1, 0, 1]
        case "B":
            return [1, 6, 0]
        case "C":
            return [1, 6, 1]
        case "N":
            return [1, 7, 0]
        case "O":
            return [1, 7, 1]
        case "F":
            return [1, 8, 0]
        case "Ne":
            return [1, 8, 1]
        case "Na":
            return [2, 0, 0]
        case "Mg":
            return [2, 0, 1]
        case "Al":
            return [2, 6, 0]
        case "Si":
            return [2, 6, 1]
        case "P":
            return [2, 7, 0]
        case "S":
            return [2, 7, 1]
        case "Cl":
            return [2, 8, 0]
        case "Ar":
            return [2, 8, 1]
        case "K":
            return [3, 0, 0]
        case "Ca":
            return [3, 0, 1]
        case "Sc":
            return [3, 1, 0]
        case "Ti":
            return [3, 1, 1]
        case "V":
            return [3, 2, 0]
        case "Cr":
            return [3, 2, 1]
        case "Mn":
            return [3, 3, 0]
        case "Fe":
            return [3, 3, 1]
        case "Co":
            return [3, 4, 0]
        case "Ni":
            return [3, 4, 1]
        case "Cu":
            return [3, 5, 0]
        case "Zn":
            return [3, 5, 1]
        case "Ga":
            return [3, 6, 0]
        case "Ge":
            return [3, 6, 1]
        case "As":
            return [3, 7, 0]
        case "Se":
            return [3, 7, 1]
        case "Br":
            return [3, 8, 0]
        case "Kr":
            return [3, 8, 1]
        case "Rb":
            return [4, 0, 0]
        case "Sr":
            return [4, 0, 1]
        case "Y":
            return [4, 1, 0]
        case "Zr":
            return [4, 1, 1]
        case "Nb":
            return [4, 2, 0]
        case "Mo":
            return [4, 2, 1]
        case "Tc":
            return [4, 3, 0]
        case "Ru":
            return [4, 3, 1]
        case "Rh":
            return [4, 4, 0]
        case "Pd":
            return [4, 4, 1]
        case "Ag":
            return [4, 5, 0]
        case "Cd":
            return [4, 5, 1]
        case "In":
            return [4, 6, 0]
        case "Sn":
            return [4, 6, 1]
        case "Sb":
            return [4, 7, 0]
        case "Te":
            return [4, 7, 1]
        case "I":
            return [4, 8, 0]
        case "Xe":
            return [4, 8, 1]
        case "Cs":
            return [5, 0, 0]
        case "Ba":
            return [5, 0, 1]
        case "Lu":
            return [5, 1, 0]
        case "Hf":
            return [5, 1, 1]
        case "Ta":
            return [5, 2, 0]
        case "W":
            return [5, 2, 1]
        case "Re":
            return [5, 3, 0]
        case "Os":
            return [5, 3, 1]
        case "Ir":
            return [5, 4, 0]
        case "Pt":
            return [5, 4, 1]
        case "Au":
            return [5, 5, 0]
        case "Hg":
            return [5, 5, 1]
        case "Tl":
            return [5, 6, 0]
        case "Pb":
            return [5, 6, 1]
        case "Bi":
            return [5, 7, 0]
        case "Po":
            return [5, 7, 1]
        case "At":
            return [5, 8, 0]
        case "Rn":
            return [5, 8, 1]
        case "Fr":
            return [6, 0, 0]
        case "Ra":
            return [6, 0, 1]
        case "Lr":
            return [6, 1, 0]
        case "Rf":
            return [6, 1, 1]
        case "Db":
            return [6, 2, 0]
        case "Sg":
            return [6, 2, 1]
        case "Bh":
            return [6, 3, 0]
        case "Hs":
            return [6, 3, 1]
        case "Mt":
            return [6, 4, 0]
        case "Ds":
            return [6, 4, 1]
        case "Rg":
            return [6, 5, 0]
        case "Cn":
            return [6, 5, 1]
        case "Nh":
            return [6, 6, 0]
        case "Fl":
            return [6, 6, 1]
        case "Mc":
            return [6, 7, 0]
        case "Lv":
            return [6, 7, 1]
        case "Ts":
            return [6, 8, 0]
        case "Og":
            return [6, 8, 1]
        case "La":
            return [5, 9, 0]
        case "Ce":
            return [5, 9, 1]
        case "Pr":
            return [5, 10, 0]
        case "Nd":
            return [5, 10, 1]
        case "Pm":
            return [5, 11, 0]
        case "Sm":
            return [5, 11, 1]
        case "Eu":
            return [5, 12, 0]
        case "Gd":
            return [5, 12, 1]
        case "Tb":
            return [5, 13, 0]
        case "Dy":
            return [5, 13, 1]
        case "Ho":
            return [5, 14, 0]
        case "Er":
            return [5, 14, 1]
        case "Tm":
            return [5, 15, 0]
        case "Yb":
            return [5, 15, 1]
        case "Ac":
            return [6, 9, 0]
        case "Th":
            return [6, 9, 1]
        case "Pa":
            return [6, 10, 0]
        case "U":
            return [6, 10, 1]
        case "Np":
            return [6, 11, 0]
        case "Pu":
            return [6, 11, 1]
        case "Am":
            return [6, 12, 0]
        case "Cm":
            return [6, 12, 1]
        case "Bk":
            return [6, 13, 0]
        case "Cf":
            return [6, 13, 1]
        case "Es":
            return [6, 14, 0]
        case "Fm":
            return [6, 14, 1]
        case "Md":
            return [6, 15, 0]
        case "No":
            return [6, 15, 1]
        case _:
            print(name)
            exit()

# We have the following data flow:
# SM:
#  ES
#  ESFD
# TI:
#  NLC
#  SEBR
# trivial:
#  LCEBR
# other - error


def convertTopoToIndex(name):
    arr = [0.0]*5
    match name:
        case "ES":
            arr[0] = 1.0
        case "ESFD":
            arr[1] = 1.0
        case "NLC":
            arr[2] = 1.0
        case "SEBR":
            arr[3] = 1.0
        case "LCEBR":
            arr[4] = 1.0
    return arr


def convertSimpleTopoToIndex(name):
    match name:
        case "trivial":
            return 0.0
        case "TI":
            return 1.0
        case "SM":
            return 2.0
        case _:
            print("error")
            exit()


globals = {
    "dataSize" : 0,
    "labelSize" : 0
}

# unlike the preprocessing module, this infers everything directly from the file name


def cmd_line(func, name):
    global globals

    # decompose the file by - :
    if len(sys.argv) == 2:
        # something
        nombre = sys.argv[1]
        nameLs = nombre.split("-")
        # 0 index is useless
        # 1 index specifies data, no more
        # 2 specified representation, pass on f, c, n for full, compressed or nan symmetry
        if nameLs[2] == "f":
            globals["dataSize"] = 13
        elif nameLs[2] == "c":
            globals["dataSize"] = 38
        else:
            globals["dataSize"] = 6
        # 3 specified representation, pass on f, c for full or compressed labeling system
        if nameLs[3] == "f":
            globals["labelSize"] = 5
        else:
            globals["labelSize"] = 3
        # 4 format type
        if not name == nameLs[4]:
            print("Wrong data type")
            exit()
        # 5 index is useless

        with open(nombre, "rb") as file:
            obj = pickle.load(file)

        func(obj)  # variables are passed as globals ;(
    else:
        print("Wrong number of arguments: ", sys.argv)
        exit()
