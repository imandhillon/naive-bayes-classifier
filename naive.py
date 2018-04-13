import sys
import numpy as np


def getdata(filestring):
    strings = filestring.split('\n')
    mylabels = []

    maxindex = 0
    split = filestring.split()

    for i in split:
        if i[0].isdigit():
            myint = int(i[0])
            if myint > int(maxindex):
                maxindex = i[0]

    numlines = len(strings)-2
    values = np.zeros([int(numlines), int(maxindex)]) 

    # -2 because there appear to be two extra newlines at end of file
    for i in range(len(strings)-2):
        line = strings[i]
        mylabels.append(line[0:2])

        for c in range(len(strings[i])):
            if strings[i][c] is ':':
                values[i,int(strings[i][c-1])-1] = strings[i][c+1]

    return mylabels, values

def probability(labels, values):
    numsamples = len(labels)
    maxindex = len(values[0])
    numplus1 = 0
    numminus1 = 0

    numattribvalues = int(np.amax(values))
    if numattribvalues is 1:
        numattribvalues = 2
    attribcount = np.transpose(np.zeros((maxindex, numattribvalues)))
    plus1attribcount = np.transpose(np.zeros([maxindex, numattribvalues]))
    minus1attribcount = np.transpose(np.zeros([maxindex, numattribvalues]))

    for i in labels:
        if int(i) is 1:
            numplus1 += 1
        else:
            numminus1 += 1

    probplus1 = numplus1/numsamples
    probminus1 = numminus1/numsamples

    for i in range(numsamples):
        for j in range(maxindex):
            v = int(values[i,j])
            attribcount[v-1, j] += 1
            if int(labels[i]) is 1:
                plus1attribcount[v-1, j] += 1
            else:
                minus1attribcount[v-1, j] += 1

    #attprobplus1 = np.divide(plus1attribcount, attribcount, out=np.zeros_like(plus1attribcount), where=attribcount!=0)
    #attprobminus1 = np.divide(minus1attribcount, attribcount, out=np.zeros_like(minus1attribcount), where=attribcount!=0)
    #print(attprobplus1)

    attprobplus1 = plus1attribcount / numplus1
    attprobminus1 = minus1attribcount / numminus1
    attribprob = attribcount / numsamples

    return probplus1, probminus1, attprobplus1, attprobminus1, attribprob

def predict(labels, values, probplus1, probminus1, attprobplus1, attprobminus1, attribprob):
    truepos = 0
    falsepos = 0
    trueneg = 0
    falseneg = 0
    # posterior = (prior * expectation) / evidence
    expectvectorp = np.zeros(len(values[0]))
    expectvectorm = np.zeros(len(values[0]))
    evidencevector = np.zeros(len(values[0]))
    for i in range(len(values)):
        for j in range(len(values[0])):
            expectvectorp[j] = attprobplus1[int(values[i,j])-1, j]
            expectvectorm[j] = attprobminus1[int(values[i,j])-1, j]
            evidencevector[j] = attribprob[int(values[i,j])-1, j]
        
        plusposterior = (probplus1 * np.prod(expectvectorp)) / np.prod(evidencevector)
        minusposterior = (probminus1 * np.prod(expectvectorm)) / np.prod(evidencevector)

        if plusposterior > minusposterior and int(labels[i]) == 1:
            truepos += 1
        if plusposterior > minusposterior and int(labels[i]) == -1:
            falsepos += 1
        if plusposterior < minusposterior and int(labels[i]) == -1:
            trueneg += 1
        if plusposterior < minusposterior and int(labels[i]) == 1:
            falseneg += 1

    print(truepos, end=' ')
    print(trueneg, end=' ')
    print(falsepos, end=' ')
    print(falseneg)
    
        

with open(sys.argv[1], 'r') as f:
    trainfile = f.read()

with open(sys.argv[2], 'r') as f:
    testfile = f.read()

trainlabels, trainvalues = getdata(trainfile)
testlabels, testvalues = getdata(testfile)

probplus1, probminus1, attprobplus1, attprobminus1, attribprob = probability(trainlabels, trainvalues)

predict(trainlabels, trainvalues, probplus1, probminus1, attprobplus1, attprobminus1, attribprob)
predict(testlabels, testvalues, probplus1, probminus1, attprobplus1, attprobminus1, attribprob)
