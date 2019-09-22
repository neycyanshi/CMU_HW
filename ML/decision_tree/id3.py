from math import log
import operator


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # count class and #label of class
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # single class entropy
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet1():  # exmaple data
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    features = ['头发', '声音']  # 2 features
    return dataSet, features


def splitDataSet(dataSet, axis, value):  # chop axis if feature[axis] == value
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # entropy after classified using the ith feature
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


class Node:
    def __init__(self, depth, val, counts):
        self.d = depth
        self.v = val  # str. feature name of this treeNode
        self.children = {}  # dict. key is edge(yes or no, or multiple choices), value is next treeNode along this edge
        self.counts = counts  # list. statistics of this node given data


class Tree:
    def __init__(self, dataSet, featureNames, maxDepth):
        self.maxDepth = maxDepth
        self.labels = [example[-1] for example in dataSet]
        self.labelNames = sorted(list(set(self.labels)))  # sort to keep order
        self.featureNames = featureNames
        self.root = self.stump(dataSet, featureNames, 1)

    def stump(self, dataSet, featureNames, depth):
        labels = [example[-1] for example in dataSet]
        statistics = [labels.count(lbName) for lbName in self.labelNames]
        if depth == self.maxDepth or len(featureNames) == 0:
            return Node(depth, majorityCnt(labels), statistics)  # now Node.val is final label when predicted
        if labels.count(labels[0]) == len(labels):
            return Node(depth, labels[0], statistics)  # now Node.val is final label
        # if len(dataSet[0]) == 1:  # only label column, len(featureNames) == 0
        #     return majorityCnt(labels)

        bestFeat = chooseBestFeatureToSplit(dataSet)
        node = Node(depth, featureNames[bestFeat], statistics)
        del (featureNames[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]

        # Binary tree if only 2 uniqueVals
        for value in set(featValues):
            visibleFeats = featureNames.copy()
            node.children[value] = self.stump(splitDataSet(dataSet, bestFeat, value), visibleFeats, depth + 1)
        return node

    def predict(self, example):
        node = self.root
        while node.children:
            selectedFeatName = node.v
            selectedFeatVal = example[selectedFeatName]  # get the corresponding feature val from example
            node = node.children[selectedFeatVal]
        return node.v

    def predictDataset(self, testSet, featureNames):
        for example in testSet:
            predictedLabel = self.predict(example)
        # write to output files here

    def prettyPrint(self):
        # total statistics
        line = "["
        for lbName in self.labelNames:
            line += ("{} {} /".format(self.labels.count(lbName), lbName))
        print(line[:-2] + "]")

        # DFS traverse tree
        def printHelper(node):
            if node is None:
                return

            # each edge/child
            for edge, cdNode in node.children.items():
                line = '| ' * node.d
                line += "{} = {}: [".format(node.v, edge)
                # statistics of this node
                for i, lbName in enumerate(self.labelNames):
                    line += ("{} {} /".format(cdNode.counts[i], lbName))
                line = line[:-2] + "]"
                print(line)  # print this node
                printHelper(cdNode)  # dfs print children nodes

        printHelper(self.root)


if __name__ == '__main__':
    dataSet, featureNames = createDataSet1()

    myTree = Tree(dataSet, featureNames, 3)
    myTree.prettyPrint()
