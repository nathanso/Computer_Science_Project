import collections
from collections import Counter
import nltk
from nltk.corpus import brown
import re

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
"""
        for z in range (0, len(lTokens[x])):
            sToken = lTokens[x][z]
            if sToken[len(sToken)-3:len(sToken)] == "n't":
                lTokens[x].remove(sToken)
                lTokens[x].insert((sToken[0,len(sToken)-4],"n't"),z)

        ##lTokens[x].extend(".")
    ##lTokens.remove(["."])
"""

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

def nGrams(nInt):
    print('Enter a sentence')
    sSent = input()
    nNum = nInt
    lNgrams = []
    lTokens =re.findall(r"[\w']+|[.,!?;]",sSent)
    for i in range(len(lTokens)-nNum+1):
        lNgrams.append(lTokens[i:i+nNum])
    return(lNgrams)
tokenizer()
