# Throughput Predicting Attempt
#     Created by Zheng Weijia (William)
#     The Chinese University of Hong Kong
#     Oct. 13 , 2021

import csv
from math import e, exp, floor, pi, sin, sqrt
from operator import index, indexOf
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import argmax, argmin, mean, size, var
import utils as utils
import matplotlib.pyplot as pyplot
import multiLinreg as MLR
from scipy.stats import laplace



MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000.0*1000.0

network_trace_dir = './dataset/network_trace/1h_less/'

count = 0
intake = 90000
testingSize = 5000
threshold = 50000

samplePoints = 40
marginalSample = 3

# initialize the sampling interval
samplingInterval = 0

for suffixNum in range(12,13):
    samplingInterval = 0.04

    networkEnvBW = []
    with open( network_trace_dir+str(suffixNum) + ".csv" ) as file1:
        for line in file1:
            if (count < intake):
                count = count  +1 
                parse = line.split()
                # the [0] is time, and the [1] is the available bw at that time
                networkEnvBW.append(float(parse[0]) / (B_IN_MB)) 



def uploadProcess(user_id, pTR):
        train_rate = pTR

        startPoint = np.quantile(networkEnvBW, 0.0005)
        endPoint = np.quantile(networkEnvBW, 0.9995)

        MIN_TP = min(networkEnvBW)
        MAX_TP = max(networkEnvBW)
        print("Max: "+ str(MAX_TP))
        print(endPoint)
        
        if (startPoint!=0):
            binsMe = np.concatenate(( np.linspace( MIN_TP,startPoint, marginalSample, endpoint=False) , np.linspace( startPoint, endPoint, samplePoints, endpoint=False) ,  np.linspace( endPoint, MAX_TP, marginalSample, endpoint=True)  ), axis=0)
        else:
            binsMe = np.concatenate(( np.linspace( startPoint, endPoint, samplePoints, endpoint=False) ,  np.linspace( endPoint, MAX_TP, marginalSample, endpoint=True)  ), axis=0)

        probability  = [ [0] * len(binsMe)  for _ in range(len(binsMe))]
        probability3D = np.zeros( (len(binsMe),len(binsMe),len(binsMe)) )
        marginalProbability  = [ 0 for  _ in  range(len(binsMe)) ]

        toBeDeleted = []

        for j in range( len(networkEnvBW) -1 - floor(len(networkEnvBW)*train_rate) - testingSize, len(networkEnvBW) -1 - testingSize ):
            
            if (j % 1 == 0):
                self = 10
                past1 = 10
                past2 = 10

                if (j <2 ):
                    continue

                for indexSelf in range(len(binsMe)-1): 
                    if (binsMe[indexSelf] <= networkEnvBW[j] and binsMe[indexSelf+1] > networkEnvBW[j]):
                        self = indexSelf
                
                for indexPast in range(len(binsMe)-1):
                    if (binsMe[indexPast] <= networkEnvBW[j-1] and binsMe[indexPast+1] > networkEnvBW[j-1]):
                        past1 = indexPast

                for indexSelf in range(len(binsMe)-1): 
                    if (binsMe[indexSelf] <= networkEnvBW[j-2] and binsMe[indexSelf+1] > networkEnvBW[j-2]):
                        past2 = indexSelf

                probability[past1][self] += 1
                probability3D[past2][past1][self] += 1
                marginalProbability[self] += 1

                toBeDeleted.append(past2)
                toBeDeleted.append(past1)
                toBeDeleted.append(self)

                while (len(toBeDeleted) >  threshold*3 ):
                    probability[toBeDeleted[1]][toBeDeleted[2]] -= 1
                    probability3D[toBeDeleted[0]][toBeDeleted[1]][toBeDeleted[2]] -= 1
                    toBeDeleted = toBeDeleted[3:]

                # probability[past1][self] += 1
                # probability3D[past2][past1][self] += 1



        trueThroughput = []
        predictedThroughput = []
        predictedThroughput3D = []
        lastMeasuredValueResult = []
        m16ValueResult = []
        m128ValueResult = []
        MLRValueResult = []

        mleThroughput = []
        ci95lb = []
        ci95ub = []
        
        countAll = 0
        countInCI = 0
        
        # This is the predicting part.
        for k in range( len(networkEnvBW)-1 - testingSize  , len(networkEnvBW)-1):
            trueThroughput.append(networkEnvBW[k])
            countAll += 1
            lastMeasuredValueResult.append( mean(networkEnvBW[max(k-1,0):k]) )
            m16ValueResult.append( mean(networkEnvBW[max(k-16,0):k]) )
            m128ValueResult.append( mean(networkEnvBW[max(k-128,0):k]) )

            predictorsNum= 10
            coef = MLR.MultipleLinearRegression(timeSeries= networkEnvBW[0:k], predictorsNum= predictorsNum, miniTrainingSize= 16)
            # if (k % 500 == 0): 
            #     print(coef)
            explantoryRVs = networkEnvBW[ k-predictorsNum : k][::-1]
            explantoryRVs[0] = 1
            MLRValueResult.append(  np.array(explantoryRVs).dot(coef)  )

            past2 = -1
            past1 = -1
            self = -1

            cilb = -1
            ciub = -1

            for indexPast in range(len(binsMe)-1):
                if (binsMe[indexPast] <= networkEnvBW[k-2] and binsMe[indexPast+1] > networkEnvBW[k-2]):
                    past2 = indexPast

            for indexPast in range(len(binsMe)-1):
                if (binsMe[indexPast] <= networkEnvBW[k-1] and binsMe[indexPast+1] > networkEnvBW[k-1]):
                    past1 = indexPast
            
            # self will not be used to predict, otherwise it is nonsense
            for indexSelf in range(len(binsMe)-1): 
                if (binsMe[indexSelf] <= networkEnvBW[k] and binsMe[indexSelf+1] > networkEnvBW[k]):
                    self = indexSelf

            if ( sum(probability[past1])!=0 and sum(probability3D[past2][past1])!=0 ):
                # expectation, 數學期望方法：
                predictedValue = utils.expectationFunction(binsMe,probability,past1, marginal=marginalSample)[0]
                predictedValue3D = utils.expectationFunction2Past(binsMe,probability3D,past1,past2, marginal=marginalSample)[0]
                
                # MLE maximum likelohood estimator方法: 
                mlePredictReturn = utils.mleFunction(binsMe=binsMe, probability=probability, past=past1)
                mlePredictValue = mlePredictReturn[0]
                # z_0.975 = 1.96

                # empirical Confidence Interval
                cilb = utils.veryConfidentFunction(binsMe=binsMe, probability=probability, past=past1, quant=0.1)[0]
                ciub = utils.veryConfidentFunction(binsMe=binsMe, probability=probability, past=past1, quant=0.9)[0]
                # print("LB:" + str(cilb) + " | UB:" + str(ciub) )

            else: 
                predictedValue = mean(networkEnvBW[max(k - 16,0):k]) 
                mlePredictValue = mean(networkEnvBW[max(k - 16,0):k])
                predictedValue3D = mean(networkEnvBW[max(k - 16,0):k]) 
            
            toBeDeleted.append(past2)
            toBeDeleted.append(past1)
            toBeDeleted.append(self)

            while (len(toBeDeleted) >  threshold*3 ):
                probability[toBeDeleted[1]][toBeDeleted[2]] -= 1
                probability3D[toBeDeleted[0]][toBeDeleted[1]][toBeDeleted[2]] -= 1
                toBeDeleted = toBeDeleted[3:]

            probability[past1][self] += 1
            probability3D[past2][past1][self] += 1

            predictedThroughput.append( predictedValue )
            mleThroughput.append( mlePredictValue)
            predictedThroughput3D.append( predictedValue3D )

            if (ciub!= -1 and cilb !=-1):
                ci95lb.append(cilb)
                ci95ub.append(ciub)

                if ( networkEnvBW[k]>= cilb and networkEnvBW[k] <= ciub  ):
                    countInCI += 1

            else: 
                ci95lb.append(mean(networkEnvBW[k]))
                ci95ub.append(mean(networkEnvBW[k]))


        # Divisor is set to be the expected value of C_i, forall i \in \mathbb{N}.
        divisor = mean(networkEnvBW)

        diffProbEst = [ (x  - y) for x,y in zip(predictedThroughput,  trueThroughput)]
        diffLastMeasure = [ (x  - y) for x,y in zip( lastMeasuredValueResult,  trueThroughput)]
        diffM16Measure = [ (x  - y) for x,y in zip( m16ValueResult,  trueThroughput)]
        diffM128Measure = [ (x  - y) for x,y in zip( m128ValueResult,  trueThroughput)]
        diffMLE = [ (x  - y) for x,y in zip(mleThroughput,  trueThroughput)]
        diffECM3d = [ (x  - y) for x,y in zip(predictedThroughput3D,  trueThroughput)]
        diffMLR = [ (x  - y) for x,y in zip(MLRValueResult,  trueThroughput)]


        # Scores:
        # I am using the RMSE model developed by http://www.mclab.info/Globecom2015.pdf
        diffScoreProbEst = sqrt( sum( [ (u**2)/len(trueThroughput)  for u in diffProbEst]  )) / divisor
        diffScoreLastMeasure = sqrt( sum( [ (u**2)/len(trueThroughput)  for u in diffLastMeasure]  ) ) / divisor
        diffScoreMLE = sqrt( sum( [ (u**2)/len(trueThroughput)  for u in diffMLE ] ) ) / divisor
        diffScoreM16 = sqrt( sum( [ (u**2)/len(trueThroughput)  for u in diffM16Measure ] ) ) / divisor
        diffScoreM128 = sqrt( sum( [ (u**2)/len(trueThroughput)  for u in diffM128Measure ] ) ) / divisor
        diffScoreECM3d = sqrt( sum( [ (u**2)/len(trueThroughput)  for u in diffECM3d ] ) ) / divisor
        diffScoreMLR = sqrt( sum( [ (u**2)/len(trueThroughput)  for u in diffMLR ] ) ) / divisor

        print("Difference Metric of ECM Algorithm:" + str(diffScoreProbEst) )
        print("Difference Metric of M=1 Algorithm:" + str(diffScoreLastMeasure) )
        print("Difference Metric of M=16 Algorithm:" + str(diffScoreM16) )
        print("Difference Metric of M=128 Algorithm:" + str(diffScoreM128) )
        print("Difference Metric of ECM3d Algorithm:" + str(diffScoreECM3d) )
        print("Difference Metric of MLR Algorithm:" + str(diffScoreMLR) )
        print("Difference Metric of MLE Algorithm:" + str(diffScoreMLE) + "\n" )
        containingRate = countInCI / countAll
        print("Being in CI rate: " + str(containingRate) + "\n" )


        # legend = []
        # pyplot.plot(trueThroughput[320:500], alpha = 1, color= "red")
        # legend.append("Real")
        # pyplot.plot(predictedThroughput[320:500], color="blue", alpha= 0.8)
        # legend.append("ECM Predict")
        # pyplot.plot(m16ValueResult[320:500], color="green",alpha= 0.8)
        # legend.append("AM(M=16)")
        # pyplot.plot(MLRValueResult[320:500], color="purple",alpha= 0.8)
        # legend.append("MLR")
        # pyplot.plot(np.array(ci95lb[320:500]), color="black",alpha= 0.8,linewidth= 0.8 )
        # legend.append("CI lower bound")
        # pyplot.plot(np.array(ci95ub[320:500]), color="black",alpha= 0.8,linewidth= 0.8 )
        # legend.append("CI upper bound")

        # pyplot.legend( legend, loc=2)
        # pyplot.ylabel("Throughput Magnitude (MB/s)")
        # pyplot.xlabel("Time " + "(per " + str(samplingInterval) +" second)" )
        # pyplot.show()

        model = probability
    



        return [diffScoreProbEst, 
                diffScoreLastMeasure, 
                diffScoreM16, 
                diffScoreM128,
                diffScoreMLE, 
                diffScoreECM3d,
                diffScoreMLR,
                model, 
                binsMe]




trackUse = [1,16,128]

xAxis = np.linspace(0.03,0.85, num=3)
yECMAxis = []
y1Axis = []
y16Axis = []
y128Axis = []
yECMLEAxis = []
yECM3d = []
yMLR = []

for ratio in xAxis:
    a = uploadProcess('dummyUsername', ratio)
    yECMAxis.append(a[0])
    y1Axis.append(a[1])
    y16Axis.append(a[2])
    y128Axis.append(a[3])
    yECMLEAxis.append(a[4])
    yECM3d.append(a[5])
    yMLR.append(a[6])

    model = a[len(a) - 2]
    
    # df = pd.DataFrame(model).to_csv("sampleMatrix.csv",header=False,index=False)

    binsMe = a[-1]    
    binPlot = binsMe[0:len(binsMe)-1]

    if (ratio == xAxis[-1]):
        for i in range(floor(samplePoints/3),floor(samplePoints/3) +5):
            origData = utils.mleFunction(binsMe=binsMe , probability=model, past= i)
            y =  origData[-1]
            ag, bg = laplace.fit( y )

            pyplot.hist(y,bins=binsMe,density=False)
            binUsed = [0] + binsMe
            pyplot.plot(binPlot, [ len(y)*( laplace.cdf(binUsed[v+1], ag, bg) -laplace.cdf(binUsed[v], ag, bg) )  for x,v in zip(binUsed[0:len(binUsed)-1],range(len(binUsed)))], '--', color ='black')
            pyplot.xlabel("Sampled Ci's magnitude, ag: " + str(ag) + " bg: "+ str(bg))
            pyplot.ylabel("# of occurs")
            pyplot.show()


print(var(networkEnvBW))


pyplot.xlabel("Proportion of training data used out of " + str(intake - testingSize))
pyplot.ylabel("Normalized RMSE")
pyplot.title("Diff score against training rate \n"  + "# Quantizer="+str(samplePoints) +", # Marginal Quantizer="+str(marginalSample) )

# ECM
pyplot.plot(xAxis, yECMAxis, color='blue',)
            # markersize=1, linewidth=0.5)

# AM M =1
pyplot.plot(xAxis, y1Axis,  color='red',)
            # markersize=1, linewidth=0.5)

# AM M =16
pyplot.plot(xAxis, y16Axis,  color='purple',)
            # markersize=1, linewidth=0.5)

# AM M=128
pyplot.plot(xAxis, y128Axis,  color='green',)
            # markersize=1, linewidth=0.5)

# MLE
pyplot.plot(xAxis, yECMLEAxis, color='orange')

# ECM (3D)
pyplot.plot(xAxis, yECM3d, color='black')

# MLR
pyplot.plot(xAxis, yMLR, color='darkseagreen')

# In total 3x3 lines have been plotted
pyplot.legend( ["ECM",  "AM (M=1)","AM (M=16)", "AM (M=128)", "ECMle", "ECM3d", "MLR"], loc=2)
pyplot.show()

