# Throughput Predicting Attempt
#     Created by Zheng Weijia (William)
#     The Chinese University of Hong Kong
#     Nov. 7, 2021

import csv
from math import exp, floor, pi, sin, sqrt
from operator import index, indexOf
import numpy as np
from numpy.core.numeric import False_
from scipy.stats import laplace
from scipy.stats import laplace_asymmetric
from scipy.stats.morestats import boxcox_normmax
from sklearn.preprocessing import MinMaxScaler
from numpy.core.fromnumeric import argmax, mean, size, var
from numpy.lib.function_base import append
import utils as utils
from torch.autograd import Variable
import torch
import time as tm
import os
import matplotlib.pyplot as pyplot
from numpy.core.fromnumeric import argmax, mean, size, var



def probest(C_iMinus1, binsMe, probModel, marginal):
    # print("C-i-1 "+ str(C_iMinus1) )
    past = -1

    predictedValue = C_iMinus1

    for indexPast in range(len(binsMe)-1):
        if (binsMe[indexPast] <= C_iMinus1 and binsMe[indexPast+1] >C_iMinus1):
            past = indexPast

    if (sum(probModel[past]) >= 50): 
        # print("***********")
        # predictedValue = utils.medianFunction(binsMe=binsMe, probability=probModel, past=past)[0]
        # predictedValue =  max(utils.medianFunction(binsMe=binsMe, probability=probModel, past=past)[0], utils.expectationFunction(binsMe=binsMe, probability=probModel, past=past)[0] )
        predictedValue =  utils.expectationFunction(binsMe=binsMe, probability=probModel, past=past, marginal=marginal)[0]
        # predictedValue = utils.veryConfidentFunction(binsMe=binsMe, probability=probModel, past=past,quant=0.1)[0]
        # predictedValue = utils.mleFunction(binsMe=binsMe, probability=probModel, past=past)[0]

    else: 
        predictedValue = -1
    
    return [predictedValue]


def confidenceSuggest(C_iMinus1, binsMe, probModel, z, FPS):
    # print("C-i-1 "+ str(C_iMinus1) )
    past = -1

    predictedValue = C_iMinus1

    for indexPast in range(len(binsMe)-1):
        if (binsMe[indexPast] <= C_iMinus1 and binsMe[indexPast+1] >C_iMinus1):
            past = indexPast

    if (sum(probModel[past]) >= 50): 
        predictedValue = (1+ 0.1/(z*FPS)) * utils.veryConfidentFunction(binsMe=binsMe,probability=probModel,past= indexPast ,quant=0.05)[0]

    else: 
        predictedValue = -1
    
    return [predictedValue]