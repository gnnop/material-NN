import csv
from http.client import REQUEST_URI_TOO_LONG
import re
import numpy as np
from _common_data_preprocessing import *

#enables me to get rid of checks

maxSize = 60

def unpackLine(str):
    x = str.split()
    return list(map(float, x))

maxi = 0

globalDataSize = 15

track = 0
#This isn't writing the predictions yet. I'm going to dump those in a separate file for ease of thought
with open("tqc-original.csv", 'r', newline='') as file:
    reader = csv.reader(file)

    for row in reader:
        track += 1
        if track % 100 == 0:
            print(track)
        
        str = row[0]

        str = re.sub(r"\s+", "", str)
        poscar = list(map(lambda a: a.strip(), row[0].split("\\n")))

        arr = [0]*(globalDataSize + 27*maxSize + 1)#9 init vals, maxSize spots for atoms, 5 final categories - 1 for the simple topo
        if len(poscar) < 7:
            print(row)
        if poscar[1].strip() == '1.0' and poscar[7].strip() == 'Direct':

            a = np.array(unpackLine(poscar[2]))
            b = np.array(unpackLine(poscar[3]))
            c = np.array(unpackLine(poscar[4]))
            alpha = np.arccos(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c))) / np.pi
            beta = np.arccos(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))) / np.pi
            gamma = np.arccos(np.dot(b, a) / (np.linalg.norm(b) * np.linalg.norm(a))) / np.pi
            arr[0:6] = [np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c), alpha, beta, gamma]
            arr[7:15] = np.unpackbits(np.array([int(row[2].strip())],dtype=np.uint8))

            atoms = poscar[5].split()
            numbs = poscar[6].split()

            total = 0
            for i in range(len(numbs)):
                total+=int(numbs[i])
                numbs[i] = total
            
            if maxi > maxSize:
                print(row[3])
                print("MAX SIZE EXCEEDED")
        else:
            print(row[3])
            print("DEFECTIVE FILE")