from _common_data_preprocessing import *
import pickle

#This isn't writing the predictions yet. I'm going to dump those in a separate file for ease of thought
def format(rFile, wFile, sym, topo):
    global globals
    global maxSize
    global atomSize
    with open(rFile, 'r', newline='') as file:
        reader = csv.reader(file)
        outPutls = []
        for row in reader:
            poscar = list(map(lambda a: a.strip(), row[0].split("\\n")))

            arr = [0]*(globals["dataSize"] + atomSize*maxSize + globals["labelSize"])#9 init vals, maxSize spots for atoms, 5 final categories - 1 for the simple topo

            arr[:globals["dataSize"]] = getGlobalData(poscar, row, sym)

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

                arr[globals["dataSize"] + i*atomSize:globals["dataSize"] + (i+1)*atomSize] = serializeAtom(atoms[atomType], poscar, i)
            
            #This should be dumping an array into the end of the array
            arr[-globals["labelSize"]:] = convertTopoToIndex(row, topo)

            outPutls.append(arr)
        
        with open(wFile, 'wb') as wFile:
            pickle.dump(outPutls, wFile)

cmd_line(format, "csnn")