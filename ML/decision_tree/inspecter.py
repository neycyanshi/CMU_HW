import math
import numpy as np
import sys
import csv


def readTSV():
    infile = sys.argv[1]
    # outfile = sys.argv[2]
    # the input file is arg1
    # the output file is arg2
    numDataTrain = 0
    labels = []

    with open(infile) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        # reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            # print(row)
            numDataTrain = numDataTrain + 1
            labels.append(row[-1])
            # print(labels)
            # for each attribute and party,
            # create a dictionary, key = number of politician, value = 1('y') and 0('n')
        del labels[0]
        # print(labels)
    return labels
    # this way it returns a list


# def entropy(num1, num2):
#     numAll = num1 + num2
#     p1 = num1 / numAll
#     p2 = num2 / numAll
#     entropy = -(p1 * math.log2(p1) + p2 * math.log2(p2))
#     return entropy

# use numpy to calculate entropy
def entropyCal(labels, base=None):
    numLabels = len(labels)

    # if there is only one kind of label then entropy = 0;
    if numLabels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    # print(value)
    # print(counts)
    # print(counts[0])
    # print(counts[1])
    probs = counts / numLabels
    n_classes = np.count_nonzero(probs)
    # calculates the number of nonzero types of labels
    if n_classes <= 1:
        return 0
    entropy = 0.
    for prob in probs:
        entropy -= prob * math.log2(prob)
    return entropy


def errorRate(labels):
    # calculate error rate, majority vote
    # find the majority
    numLabels = len(labels)
    if numLabels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    maxC = np.amax(counts)
    # print("maxC = " + str(maxC))
    error = (numLabels - maxC) / numLabels
    return error


# make an output .txt file
def outTxt(entropy, error):
    outfile = sys.argv[2]
    outputf = open(outfile, "w+")
    entropyOut = "entropy: " + str(entropy)
    errorOut = "error: " + str(error)
    out = entropyOut + '\n' + errorOut
    outputf.write("%s" % out)
    outputf.close()


if __name__ == "__main__":
    entropy = entropyCal(readTSV())
    error = errorRate(readTSV())
    # print("entropy = " + str(entropy))
    # print("error = " + str(error))
    outTxt(entropy, error)
