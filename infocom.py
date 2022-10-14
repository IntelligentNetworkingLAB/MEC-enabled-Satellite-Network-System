import random as rand
import math
from re import L
from turtle import goto
import numpy as np
from random import *
from matplotlib import pyplot as plt
import copy
import time
import pandas as pd
from sympy import LC
import onnxruntime as ort
import pandas as pd
import openpyxl

CubeSatNum = 5 # for each LEOSat
LEOSatNum = 5

alpha1 = 0.5
alpha2 = 0.5
Xtran = [0.8, 1.2, 3.0]
Xcomp = [0.08, 0.3, 10.0]
#tB = [4e+2, 2e+3, 5e+4]
tB = [2e+3, 2e+3, 2e+3]
tP = [1e+2, 8e+2, 5e+3]
#tP = [8e+2, 8e+2, 8e+2]

# 0:CubeSat, 1:LEO, 2:GEO
SNR = 1

class SubTask():
    def __init__(self, index):
        self.index = index
        self.memory = rand.randrange(10, 90)
        #self.memory = 50
        self.power = rand.randrange(15, 70)
        #self.power = 50
        self.kindLEO = rand.randint(0, LEOSatNum)
        self.kindCube = rand.randint(0, CubeSatNum)

        self.X = 2 # 2:GEO, 1:LEO, 0:CubeSat
        self.y = 1/SubTaskNum
        self.beta = 1/SubTaskNum

    def Print(self):
        print(str(self.index) + ":" + str(self.X) + "/" + str(self.y) + "/" + str(self.beta))

SubTaskNum = 0
SubTaskSet = []

# Load Learning Model
cube_onnx_path = "model/CubeSat.onnx"
cube_ort_sess = ort.InferenceSession(cube_onnx_path)
LMS_onnx_path = "model/LMSSat.onnx"
LMS_ort_sess = ort.InferenceSession(LMS_onnx_path)
CC_onnx_path = "model/CCSat.onnx"
CC_ort_sess = ort.InferenceSession(CC_onnx_path)

def ProposedX():
    for i in SubTaskSet:
        i.X = 2

    leoIndexs = [[],[],[],[],[]]
    for i in range(SubTaskNum):
        if SubTaskSet[i].kindLEO != 0:
            leoIndexs[SubTaskSet[i].kindLEO - 1].append(SubTaskSet[i].index)
    
    for leoi in leoIndexs:
        for cu in range(1, 6): # Assign for each CubeSat
            cubeInput = []
            cubeInputIndex = []
            count = 0
            masks = [False for _ in range(100)]
            for i in leoi:
                if SubTaskSet[i].kindCube == cu and SubTaskSet[i].X != 0:
                    tmp = [0 for _ in range(100)]
                    masks[count] = True
                    tmp[count] = 1
                    tmp.append(SubTaskSet[i].memory * 1e-2)
                    tmp.append(SubTaskSet[i].power * 1e-2)
                    cubeInput.append(tmp)
                    count += 1
                    cubeInputIndex.append(SubTaskSet[i].index)
            #print(count)
            for _ in range(count, 100):
                tmp = [0 for _ in range(100)]
                tmp.append(0)
                tmp.append(0)
                cubeInput.append(tmp)

            if count == 0: break
            for _ in range(20):
                # print(len(cubeInput))
                outputs = cube_ort_sess.run(['discrete_actions'], {'obs_0': [cubeInput], 'action_masks': [masks]})
                ch = outputs[0][0][0]
                SubTaskSet[cubeInputIndex[ch]].X = 0
                masks[ch] = False
       
        for _ in range(100):
            # Assign for LEOSat
            leoInput = []
            leoInputIndex = []
            count = 0
            masks = [True for _ in range(100)]
            for i in leoi:
                if count == 100: break
                if SubTaskSet[i].X == 2:
                    tmp = [0 for _ in range(100)]
                    tmp[count] = 1
                    tmp.append(SubTaskSet[i].memory * 1e-2)
                    tmp.append(SubTaskSet[i].power * 1e-2)
                    leoInput.append(tmp)
                    count += 1
                    leoInputIndex.append(SubTaskSet[i].index)

            for i in range(count, 100):
                masks[i] = False
                tmp = [0 for _ in range(100)]
                tmp.append(0)
                tmp.append(0)
                leoInput.append(tmp)
    
            if count == 0: break
            outputs = LMS_ort_sess.run(['discrete_actions'], {'obs_0': [leoInput], 'action_masks': [masks]})
            ch = outputs[0][0][0]
            SubTaskSet[leoInputIndex[ch]].X = 1
    return

def CCppoX():
    for i in SubTaskSet:
        i.X = 2

    leoIndexs = [[],[],[],[],[]]
    for i in range(SubTaskNum):
        if SubTaskSet[i].kindLEO != 0:
            leoIndexs[SubTaskSet[i].kindLEO - 1].append(SubTaskSet[i].index)
    
    for leoi in leoIndexs:
        for cu in range(1, 6): # Assign for each CubeSat
            cubeInput = []
            cubeInputIndex = []
            count = 0
            masks = [False for _ in range(100)]
            for i in leoi:
                if SubTaskSet[i].kindCube == cu and SubTaskSet[i].X != 0:
                    tmp = [0 for _ in range(100)]
                    masks[count] = True
                    tmp[count] = 1
                    tmp.append(SubTaskSet[i].memory * 1e-2)
                    tmp.append(SubTaskSet[i].power * 1e-2)
                    cubeInput.append(tmp)
                    count += 1
                    cubeInputIndex.append(SubTaskSet[i].index)
            #print(count)
            for _ in range(count, 100):
                tmp = [0 for _ in range(100)]
                tmp.append(0)
                tmp.append(0)
                cubeInput.append(tmp)

            if count == 0: break
            for _ in range(20):
                # print(len(cubeInput))
                outputs = CC_ort_sess.run(['discrete_actions'], {'obs_0': [cubeInput], 'obs_1': [[0]], 'action_masks': [masks]})
                ch = outputs[0][0][0]
                SubTaskSet[cubeInputIndex[ch]].X = 0
                # masks[ch] = False
       
        for _ in range(100):
            # Assign for LEOSat
            leoInput = []
            leoInputIndex = []
            count = 0
            masks = [True for _ in range(100)]
            for i in leoi:
                if count == 100: break
                if SubTaskSet[i].X == 2:
                    tmp = [0 for _ in range(100)]
                    tmp[count] = 1
                    tmp.append(SubTaskSet[i].memory * 1e-2)
                    tmp.append(SubTaskSet[i].power * 1e-2)
                    leoInput.append(tmp)
                    count += 1
                    leoInputIndex.append(SubTaskSet[i].index)

            for i in range(count, 100):
                masks[i] = False
                tmp = [0 for _ in range(100)]
                tmp.append(0)
                tmp.append(0)
                leoInput.append(tmp)
    
            if count == 0: break
            outputs = CC_ort_sess.run(['discrete_actions'], {'obs_0': [cubeInput], 'obs_1': [[1]], 'action_masks': [masks]})
            ch = outputs[0][0][0]
            SubTaskSet[leoInputIndex[ch]].X = 1
    return

def StaticX():
    for i in SubTaskSet:
        i.X = 2

    leoIndexs = [[],[],[],[],[]]
    for i in range(SubTaskNum):
        if SubTaskSet[i].kindLEO != 0:
            leoIndexs[SubTaskSet[i].kindLEO - 1].append(SubTaskSet[i].index)
    
    for leoi in leoIndexs:
        for cu in range(1, 6): # Assign for each CubeSat
            cubeInputIndex = []
            for i in leoi:
                if SubTaskSet[i].kindCube == cu:
                    cubeInputIndex.append(SubTaskSet[i].index)
            
            for _ in range(20):
                if len(cubeInputIndex) == 0:
                    break
                maxi = 0
                for i in range(len(cubeInputIndex)):
                    if (SubTaskSet[cubeInputIndex[maxi]].power + SubTaskSet[cubeInputIndex[maxi]].memory) > (SubTaskSet[cubeInputIndex[i]].power + SubTaskSet[cubeInputIndex[i]].memory):
                        maxi = i 
                SubTaskSet[cubeInputIndex[maxi]].X = 0
                cubeInputIndex.pop(maxi)
                
                    
        # Assign for LEOSat
        leoInputIndex = []
        for i in leoi:
            if SubTaskSet[i].X == 2:
                leoInputIndex.append(SubTaskSet[i].index)
        
        for _ in range(100):
            if len(leoInputIndex) == 0:
                break
            maxi = 0
            for i in range(len(leoInputIndex)):
                if (SubTaskSet[leoInputIndex[maxi]].power + SubTaskSet[leoInputIndex[maxi]].memory) > (SubTaskSet[leoInputIndex[i]].power + SubTaskSet[leoInputIndex[i]].memory):
                    maxi = i 
            SubTaskSet[leoInputIndex[maxi]].X = 1
            leoInputIndex.pop(maxi)
    return

def RandomX():
    for i in SubTaskSet:
        i.X = 2

    leoIndexs = [[],[],[],[],[]]
    for i in range(SubTaskNum):
        if SubTaskSet[i].kindLEO != 0:
            leoIndexs[SubTaskSet[i].kindLEO - 1].append(SubTaskSet[i].index)
    
    for leoi in leoIndexs:
        for cu in range(1, 6): # Assign for each CubeSat
            cubeInputIndex = []
            for i in leoi:
                if SubTaskSet[i].kindCube == cu:
                    cubeInputIndex.append(SubTaskSet[i].index)
            #print(len(cubeInputIndex))
            for _ in range(20):
                SubTaskSet[cubeInputIndex[rand.randint(0, len(cubeInputIndex) - 1)]].X = 0

        # Assign for LEOSat
        leoInputIndex = []
        for i in leoi:
            if SubTaskSet[i].X == 2:
                leoInputIndex.append(SubTaskSet[i].index)
        
        #print(len(leoInputIndex))
        for i in range(100):
            SubTaskSet[leoInputIndex[rand.randint(0, len(leoInputIndex) - 1)]].X = 1

    return

def ProposedResource():
    global SubTaskSet

    balance = 0.1
    iota = 1e+1

    leoIndexs = [[],[],[],[],[]]
    for i in range(SubTaskNum):
        if SubTaskSet[i].kindLEO != 0:
            leoIndexs[SubTaskSet[i].kindLEO - 1].append(SubTaskSet[i].index)
    
    for leoi in leoIndexs:
        # Assign for each CubeSat 
        for cu in range(1, 6):
            cubeInputIndex = []
            for i in leoi:
                if SubTaskSet[i].kindCube == cu and SubTaskSet[i].X == 0:
                    cubeInputIndex.append(SubTaskSet[i].index)

            if len(cubeInputIndex) == 0: break
            
            # y*
            o = []
            for i in cubeInputIndex:
                o.append(math.sqrt((alpha1*SubTaskSet[i].memory)/(tB[0] * math.log2(1+SNR))))
            for i in range(len(cubeInputIndex)):
                SubTaskSet[cubeInputIndex[i]].y = o[i] / (sum(o) - o[i])

            # beta*
            eta = 0
            for i in cubeInputIndex:
                bandwidth = SubTaskSet[i].y * tB[0]
                power = SubTaskSet[i].beta * tP[0]
                Ttran = SubTaskSet[i].memory / (bandwidth * math.log2(1 + SNR));
                Ptran = Xtran[0] * bandwidth
                Tcomp = SubTaskSet[i].power / power
                Pcomp = Xcomp[0] * power
                eta += alpha1 * (Ttran + Tcomp) + alpha2 * (Ptran + Pcomp)
            eta /= (len(cubeInputIndex)*balance)
            for i in cubeInputIndex:
                Phi = (alpha1 * SubTaskSet[i].power)/(eta * len(cubeInputIndex))
                Lambda = (alpha2 * Xcomp[0] * eta)/(len(cubeInputIndex))
                SubTaskSet[i].beta = math.sqrt(Phi/(Lambda + iota))

        # Assign for LEOSat
        leoInputIndex = []
        for i in leoi:
            if SubTaskSet[i].X == 1:
                leoInputIndex.append(SubTaskSet[i].index)
        
        if len(leoInputIndex) == 0: break
        
        # y*
        o = []
        for i in leoInputIndex:
            o.append(math.sqrt((alpha1*SubTaskSet[i].memory)/(tB[1] * math.log2(1+SNR))))
        for i in range(len(leoInputIndex)):
            SubTaskSet[leoInputIndex[i]].y = o[i] / (sum(o) - o[i])

        # beta*
        eta = 0
        for i in leoInputIndex:
            bandwidth = SubTaskSet[i].y * tB[1]
            power = SubTaskSet[i].beta * tP[1]
            Ttran = SubTaskSet[i].memory / (bandwidth * math.log2(1 + SNR));
            Ptran = Xtran[1] * bandwidth
            Tcomp = SubTaskSet[i].power / power
            Pcomp = Xcomp[1] * power
            eta += alpha1 * (Ttran + Tcomp) + alpha2 * (Ptran + Pcomp)
        eta /= (len(leoInputIndex) * balance)
        for i in leoInputIndex:
            Phi = (alpha1 * SubTaskSet[i].power)/(eta * len(leoInputIndex))
            Lambda = (alpha2 * Xcomp[1] * eta)/(len(leoInputIndex))
            SubTaskSet[i].beta = math.sqrt(Phi/(Lambda + iota))
    
    # Assign for GEOSat
    geoInputIndex = []
    for i in SubTaskSet:
        if i.X == 2:
            geoInputIndex.append(i.index)
    
    # y*
    o = []
    for i in geoInputIndex:
        o.append(math.sqrt((alpha1*SubTaskSet[i].memory)/(tB[2] * math.log2(1+SNR))))
    for i in range(len(geoInputIndex)):
        SubTaskSet[geoInputIndex[i]].y = o[i] / (sum(o) - o[i])

    # beta*
    eta = 0
    for i in geoInputIndex:
        bandwidth = SubTaskSet[i].y * tB[2]
        power = SubTaskSet[i].beta * tP[2]
        Ttran = SubTaskSet[i].memory / (bandwidth * math.log2(1 + SNR));
        Ptran = Xtran[2] * bandwidth
        Tcomp = SubTaskSet[i].power / power
        Pcomp = Xcomp[2] * power
        eta += alpha1 * (Ttran + Tcomp) + alpha2 * (Ptran + Pcomp)
    eta /= (len(geoInputIndex) * balance)
    for i in geoInputIndex:
        Phi = (alpha1 * SubTaskSet[i].power)/(eta * len(geoInputIndex))
        Lambda = (alpha2 * Xcomp[2] * eta)/(len(geoInputIndex))
        SubTaskSet[i].beta = math.sqrt(Phi/(Lambda + iota))

    return

def EqualResource():
    global SubTaskSet
    for task in SubTaskSet:
        if task.X == 0:
            task.y = 1/20
            task.beta = 1/20
        elif task.X == 1 :
            task.y = 1/100
            task.beta = 1/100
        else:
            task.y = 1/(SubTaskNum - 1000)
            task.beta = 1/(SubTaskNum - 1000)
    return

def GetUtility():
    result = 0
    for i in range(SubTaskNum):
        bandwidth = SubTaskSet[i].y * tB[SubTaskSet[i].X]
        power = SubTaskSet[i].beta * tP[SubTaskSet[i].X]
        Ttran = SubTaskSet[i].memory / (bandwidth * math.log2(1 + SNR));
        Ptran = Xtran[SubTaskSet[i].X] * bandwidth
        Tcomp = SubTaskSet[i].power / power
        Pcomp = Xcomp[SubTaskSet[i].X] * power
        result += alpha1 * (Ttran + Tcomp) + alpha2 * (Ptran + Pcomp)
    return result/1e+3

def DataDifferentPower(str):
    df = [[0, 0, 0],[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in SubTaskSet:
        if i.power <= 25:
            df[0][i.X] += 1
        elif i.power <= 35:
            df[1][i.X] += 1
        elif i.power <= 45:
            df[2][i.X] += 1
        elif i.power <= 55:
            df[3][i.X] += 1
        elif i.power <= 70:
            df[4][i.X] += 1
    pd.DataFrame(df).to_excel(str)
def DataDifferentMemory(str):
    df = [[0, 0, 0],[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in SubTaskSet:
        if i.memory <= 20:
            df[0][i.X] += 1
        elif i.memory <= 40:
            df[1][i.X] += 1
        elif i.memory <= 60:
            df[2][i.X] += 1
        elif i.memory <= 80:
            df[3][i.X] += 1
        elif i.memory <= 90:
            df[4][i.X] += 1
    pd.DataFrame(df).to_excel(str)
def DataTotalState():
    df = []
    for i in SubTaskSet:
        data = [i.memory, i.power, i.kindCube, i.kindLEO, i.X,  i.y, i.beta]
        df.append(data)
    
    pd.DataFrame(df).to_excel('export_sample.xlsx', sheet_name='new_name')

def initResource():
    for i in SubTaskSet:
        i.beta = 0.01
        i.y = 0.01

if __name__ == '__main__':
    ccc = 1450
    for _ in range(5):
        ccc+=50
        print("==================================" + str(ccc))
        SubTaskNum = rand.randint(ccc, ccc)
        SubTaskSet = [SubTask(i) for i in range(SubTaskNum)]

        optiRe = []

        RandomX()
        initResource()
        ProposedResource()
        optiRe.append(GetUtility())

        CCppoX()
        initResource()
        ProposedResource()
        optiRe.append(GetUtility())

        ProposedX()
        initResource()
        ProposedResource()
        optiRe.append(GetUtility())

        print(optiRe)


    #nums = [0, 0, 0]
    #for i in SubTaskSet:
    #    nums[i.X] += 1
    #print(nums)
    
    #sums = [0, 0]
    #for i in SubTaskSet:
    #    sums[0] += i.y
    #    sums[1] += i.beta
    #print(sums)