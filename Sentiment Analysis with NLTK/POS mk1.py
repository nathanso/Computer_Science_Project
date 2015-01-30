import collections
from collections import Counter
from nltk import UnigramTagger as ut
from nltk import BigramTagger as bt
from pickle import dump,load
import re
import math
import random
import csv


def loadCsv(filename):
        with open(filename, "r") as O:
                lines = csv.reader(O)
                dataset = list(lines)
                for i in range(len(dataset)):
                        dataset[i] = [float(x) for x in dataset[i]]
                return dataset

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iter():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    filename = 'pima-indians-diabetes.data.csv'
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split '+str(len(dataset))+" rows into train= "+str(len(trainingSet))+' and test '+str(len(testSet))+ 'rows')
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy:'+str(accuracy))

def loadtagger(taggerfilename):
        infile = open(taggerfilename,'rb')
        tagger = load(infile); infile.close()
        return tagger

def traintag(corpusname, corpus):
        # Function to save tagger.
        def savetagger(tagfilename,tagger):
                outfile = open(tagfilename, 'wb')
                dump(tagger,outfile,-1); outfile.close()
                return
        # Training UnigramTagger.
        uni_tag = ut(corpus)
        savetagger(corpusname+'_unigram.tagger',uni_tag)
        # Training BigramTagger.
        bi_tag = bt(corpus)
        savetagger(corpusname+'_bigram.tagger',bi_tag)
        print("Tagger trained with",corpusname,"using" +\
                                "UnigramTagger and BigramTagger.")
        return

# Function to unchunk corpus.
def unchunk(corpus):
        nomwe_corpus = []
        for i in corpus:
                nomwe = " ".join([j[0].replace("_"," ") for j in i])
                nomwe_corpus.append(nomwe.split())
        return nomwe_corpus

class browntag():
        def __init__(self,mwe=True):
                self.mwe = mwe
                # Train tagger if it's used for the first time.
                try:
                        loadtagger('brown_unigram.tagger').tag(['estoy'])
                        loadtagger('brown_bigram.tagger').tag(['estoy'])
                except IOError:
                        print("*** First-time use of brown tagger ***")
                        print("Training tagger ...")
                        from nltk.corpus import brown as brown
                        brown_sents = brown.tagged_sents()
                        traintag('brown',brown_sents)
                        # Trains the tagger with no MWE.
                        brown_nomwe = unchunk(brown.tagged_sents())
                        tagged_brown_nomwe = batch_pos_tag(brown_nomwe)
                        traintag('brown_nomwe',tagged_brown_nomwe)
                        print
                # Load tagger.
                if self.mwe == True:
                        self.uni = loadtagger('brown_unigram.tagger')
                        self.bi = loadtagger('brown_bigram.tagger')
                elif self.mwe == False:
                        self.uni = loadtagger('brown_nomwe_unigram.tagger')
                        self.bi = loadtagger('brown_nomwe_bigram.tagger')

def pos_tag(tokens, mmwe=True):
        tagger = browntag(mmwe)
        return tagger.uni.tag(tokens)

def batch_pos_tag(sentences, mmwe=True):
        tagger = browntag(mmwe)
        return tagger.uni.tag_sents(sentences)

def biGrams():
    print('Enter a sentence')
    sSent = input()
    lBigrams = []
    lSent =re.findall(r"[\w']+|[.,!?;]",sSent)
    for i in range(len(lSent)-1):
        lBigrams.append((lSent[i]+" "+lSent[i+1]))
    return lBigrams

def tokenizer ():
    print('Enter Some sentences')
    sSent = input()
    #lTokens = sSent.split(".")
    #for x in range (0, len(lTokens)):
        #sSent = lTokens[x]
        #del lTokens[x]

    lTokens = re.findall(r"[\w']+|[.,!?;]",sSent)
    #lTokens.remove([])
    print(lTokens)
    return lTokens

main()
tagger = browntag()
print (tagger.uni.tag(biGrams()))
