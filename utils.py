from gettext import find
import math
from unittest import result
from warnings import catch_warnings
from matplotlib.pyplot import summer
from numpy.core.fromnumeric import argmax, partition
from numpy.lib.function_base import median
from scipy.stats import norm
from scipy.stats import laplace
import numpy as np
from numpy.core.fromnumeric import argmax, mean, size, var
import math
import random
import bisect

from sklearn.metrics import top_k_accuracy_score

B_IN_MB = 1000.0*1000.0

def index_of_first(cdn_arrive_time, playingTime, pRealFrameNo ):
    for i,v in enumerate(cdn_arrive_time):
        if ( cdn_arrive_time[i+1] > playingTime and  cdn_arrive_time[i] <= playingTime    ):
            return i
    return None


def index_first_equal(number,list ):
    for i,v in enumerate(list):
        if (  v == number  ):
            return i
    return None


def frame_upload_done_time( runningTime, networkEnvBW, size, networkSamplingInterval ):
    shift = math.floor( runningTime/networkSamplingInterval )
    i = 0 

    while (size > 0): 
        if (i == 0):
            i = 1
            s_temp = size - networkEnvBW[shift]  *( networkSamplingInterval*(shift+1) - runningTime )
            # print(s_temp)
            if (s_temp<=0):
                t_out = (size / (networkEnvBW[shift] ) ) + runningTime
                return t_out
            size = s_temp

        else: 
            # print(shift)
            s_temp = size - networkEnvBW[shift]*networkSamplingInterval
            # print(s_temp)
            if (s_temp<=0): 
                t_out = networkSamplingInterval*shift + size / (networkEnvBW[shift] )
                return t_out
            size = s_temp

        shift = shift +1
    
    if (size<=0): 
        return runningTime
    print(size)


def packet_level_frame_upload_finish_time( runningTime, packet_level_data, packet_level_timestamp, framesize,
                                            packet_level_integral_C, packet_level_time, toUsePacketRecords ):
    shift = find_gt_index(a= packet_level_timestamp, x= runningTime)
    i = 0 

    while (framesize > 0): 
        if (i == 0):
            i = 1
            s_temp = framesize - packet_level_data[shift]
            if (toUsePacketRecords):
                packet_level_integral_C.append(packet_level_integral_C[-1]+ packet_level_data[shift])
                packet_level_time.append(packet_level_timestamp[shift])
            if (s_temp<=0):
                t_out = packet_level_timestamp[shift]
                return [t_out, packet_level_integral_C, packet_level_time]
            framesize = s_temp
        else: 
            s_temp = framesize - packet_level_data[shift]
            if (toUsePacketRecords):
                packet_level_integral_C.append(packet_level_integral_C[-1]+ packet_level_data[shift])
                packet_level_time.append(packet_level_timestamp[shift])
            if (s_temp<=0): 
                t_out = packet_level_timestamp[shift]
                return [t_out,packet_level_integral_C, packet_level_time]
            framesize = s_temp
        shift = shift +1

def expectationFunction(binsMe,probability,past,marginal):
    result = 0
    p1 = random.uniform(0, 1)
    p2 = 1-p1
    occur = sum(probability[past])
    prob = [ x/occur for x in probability[past] ]
    for p,index in zip(prob, range(len(prob)-1) ):
        # if (index >= marginal and index < len(binsMe)-marginal):
            result = result + p*(p1*binsMe[index]+p2*binsMe[index+1])
    
    return [result]

def expectationFunction2Past(binsMe,probability,past1, past2,marginal):
    result = 0
    p1 = random.uniform(0, 1)
    p2 = 1-p1
    occur = sum(probability[past2][past1])
    prob = [ x/occur for x in probability[past2][past1] ]
    for p,index in zip(prob, range(len(prob)-1) ):
        # if (index >= marginal and index < len(binsMe)-marginal):
            result = result + p*( p1 * binsMe[index] + p2 * binsMe[index+1])
    
    return [result]

def medianFunction(binsMe,probability,past):
    histogram_establish = []
    for ppValue, index in zip(probability[past], range(len(probability[past]) -1 )  ):
        counter = 0
        while (counter < ppValue):
            counter = counter + 1
            histogram_establish.append(0.5*binsMe[index]+0.5*binsMe[index+1])
    
    giveBackValue = laplace.fit(histogram_establish)[0]

    return [ giveBackValue, histogram_establish]


def mleFunction(binsMe,probability,past):
    histogram_establish = []
    for ppValue, index in zip(probability[past], range(len(probability[past]) -1 )  ):
        counter = 0
        while (counter < ppValue):
            counter = counter + 1
            histogram_establish.append(0.5*binsMe[index]+0.5*binsMe[index+1])
    
    giveBackValue = binsMe[np.argmax(probability[past])]

    return [ giveBackValue, histogram_establish]

def mleFunction2Past(binsMe,probability,past1,past2):
    histogram_establish = []
    p1 = random.uniform(0, 1)
    p2 = 1-p1
    for ppValue, index in zip(probability[past2][past1], range(len(probability[past2][past1]) -1 )  ):
        counter = 0
        while (counter < ppValue):
            counter = counter + 1
            histogram_establish.append(p1*binsMe[index]+p2*binsMe[index+1])
    
    giveBackValue = binsMe[np.argmax(probability[past2][past1])]

    return [ giveBackValue, histogram_establish]


def veryConfidentFunction(binsMe,probability,C_iMinus1, quant):
    histogram_establish = []

    past = -np.Infinity

    for indexPast in range(len(binsMe)-1):
        if (binsMe[indexPast] <= C_iMinus1 and binsMe[indexPast+1] >=C_iMinus1):
            past = indexPast

    if (past == -1): return [-1, [] ]
    try:
        for ppValue, index in zip(probability[past], range(len(probability[past])-1)  ):
            counter = 0
            while (counter < ppValue):
                counter = counter + 1
                # histogram_establish.append(0.5*binsMe[index]+0.5*binsMe[index+1])
                histogram_establish.append( binsMe[index] + (binsMe[index+1]-binsMe[index])*counter/ppValue )
        
        valueGiven = np.quantile(histogram_establish,quant)
    except: 
        valueGiven = -1

    return [ valueGiven, histogram_establish]


def veryConfidentFunction2Past(binsMe,probability,past1,past2, quant):
    histogram_establish = []
    for ppValue, index in zip(probability[past2][past1], range(len(probability[past2][past1])-1)  ):
        counter = 0
        while (counter < ppValue):
            counter = counter + 1
            # histogram_establish.append(0.5*binsMe[index]+0.5*binsMe[index+1])
            histogram_establish.append( binsMe[index] + (binsMe[index+1]-binsMe[index])*counter/ppValue )
    
    valueGiven = np.quantile(histogram_establish,quant)

    return [ valueGiven, histogram_establish]



def constructProbabilityModel(networkEnvBW, binsMe, networkSampleFreq, traceDataSampleFreq, threshold): 
        if (len(networkEnvBW)>threshold):
            networkEnvBW = networkEnvBW[len(networkEnvBW)-1 - threshold :len(networkEnvBW)-1]
        tobeDeleted = []
        probability  = [ [0] * len(binsMe)  for _ in range(len(binsMe))]
        marginalProbability  = [ 0 for  _ in  range(len(binsMe)) ]

        for j in range( 1,  len(networkEnvBW) ):
            if (j%1000 == 0) : print( str( (j/ len(networkEnvBW))*100 ) + "%, please wait."  )
            
            if (j % math.floor(traceDataSampleFreq/networkSampleFreq) == 0 ):
                self = 10
                past = 10

                if (j == 0 ):
                    continue

                for indexSelf in range(len(binsMe)-1): 
                    if (binsMe[indexSelf] <= networkEnvBW[j] and binsMe[indexSelf+1] > networkEnvBW[j]):
                        self = indexSelf
                
                for indexPast in range(len(binsMe)-1):
                    if (binsMe[indexPast] <= networkEnvBW[j-1] and binsMe[indexPast+1] > networkEnvBW[j-1]):
                        past = indexPast

                probability[past][self] += 1
                marginalProbability[self] += 1
                tobeDeleted.append(past)
                tobeDeleted.append(self)

        model = probability
    
        return [model,tobeDeleted]

def generatingBackwardHistogram(FPS,int_C, timeSeq, currentTime, lenLimit):
    pilot = timeSeq[-1]
    result = []
    while (len(result)< lenLimit and pilot - 1/FPS>timeSeq[0]):
        result.insert(0, (F(pilotTime=pilot, int_C=int_C, timeSeq=timeSeq)-F(pilotTime =pilot-1/FPS, int_C=int_C, timeSeq=timeSeq))*FPS )
        pilot = pilot - 1/FPS
    
    return result

def find_lt_index(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    if i:
        return i-1
    raise ValueError

def find_le_index(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return i-1
    raise ValueError

def find_gt_index(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return i
    raise ValueError

def find_ge_index(a, x):
    'Find leftmost item greater than or equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError

def F(pilotTime, int_C, timeSeq):
    try:
        ge_index = find_ge_index(timeSeq,pilotTime)
    except: 
        ge_index = len(timeSeq) - 1
    lt_index = find_le_index(timeSeq,pilotTime)

    if (int_C[lt_index] == int_C[ge_index] ==0 or timeSeq[ge_index]-timeSeq[lt_index] == 0):
        FValue = int_C[lt_index]
    else:
        FValue = int_C[lt_index] + (int_C[ge_index]-int_C[lt_index])/(timeSeq[ge_index]-timeSeq[lt_index]) * (pilotTime-timeSeq[lt_index])
    return FValue

def generatingBackwardHistogramS(backwardTime, int_C, timeSeq, currentTime, lenLimit):
    shift = find_le_index(a= timeSeq, x= currentTime)
    # print(shift)
    pilot = timeSeq[shift]
    result = []
    while (len(result)< lenLimit and pilot-backwardTime>timeSeq[0]):
        F_big = F(pilotTime=pilot, int_C=int_C, timeSeq=timeSeq)
        F_small = F(pilotTime =pilot - backwardTime, int_C=int_C, timeSeq=timeSeq)
        total_size_sent_interval = F_big - F_small
        if (total_size_sent_interval <0): print(str(currentTime) + " is the current time and " + str(pilot))
        result.insert(0, total_size_sent_interval)
        pilot = pilot - backwardTime
    
    return result