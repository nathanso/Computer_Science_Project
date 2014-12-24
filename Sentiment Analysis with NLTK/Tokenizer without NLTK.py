import collections
from collections import Counter
import nltk
from nltk.corpus import brown
import re

def tokenizer ():
    print('Enter Some sentences')
    sSent = input()
    lTokens = sSent.split(".")
    for x in range (0, len(lTokens)):
        sSent = lTokens[x]
        del lTokens[x]
        lTokens.insert(x,re.findall(r"[\w']+|[.,!?;]",sSent))
        ##lTokens[x].extend(".")
    ##lTokens.remove(["."])
    lTokens.remove([])
    print(lTokens)
    return lTokens

def wordSentListFreq ():
    lTokens = tokenizer()
    lWordFreq = []
    for x in range (0, len(lTokens)):
        lTemp = Counter(lTokens[x])
        lWordFreq.append(lTemp)
    return lSentWordFreq

def wordListFreq():
    print('Enter Some sentences')
    sSent = input()
    lTokens.insert(x,re.findall(r"[\w']+|[.,!?;]",sSent))
    for x in range (0, len(lTokens)):
        lWordFreq = Counter(lTokens)
    return lWordFreq

def biGrams():
    print('Enter a sentence')
    sSent = input()
    lBigrams = []
    lSent =re.findall(r"[\w']+|[.,!?;]",sSent)
    for i in range(len(lSent)-1):
        lBigrams.append((lSent[i], lSent[i+1]))
    return lBigrams

def nGrams():
    print('Enter a sentence')
    sSent = input()
    print('state your n')
    sNum = input()
    nNum = int(sNum)
    lNgrams = []
    lSent =re.findall(r"[\w']+|[.,!?;]",sSent)
    for i in range(len(lSent)-nNum+1):
        lNgrams.append(lSent[i:i+nNum])
    return(lNgrams)

def 
