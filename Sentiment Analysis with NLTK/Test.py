import re
import collections
from collections import Counter


def biGrams():
    print('Enter a sentence')
    sSent = input()
    lBigrams = []
    lSent =re.findall(r"[\w']+|[.,!?;]",sSent)
    for i in range(len(lSent)-1):
        lBigrams.append((lSent[i]+" "+lSent[i+1]))
    print(lBigrams)
    return lBigrams

biGrams()
