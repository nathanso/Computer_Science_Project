import re

def tokenizer():
    print("Please Enter your Sentence")
    sSentence = input()
    
    if sSentence[len(sSentence):len(sSentence)+1] != ".":
        sSentence = sSentence +"."
    
    lSen = []
    nStor1 = 0
    nStor2 = 0
    nLetter = ""
    sWord = ""
    sNegate = "n't"
    if sSentence[len(sSentence)-1:len(sSentence)]!= ".":
        sSentence = sSentence +"."
    for i in range (0,  len(sSentence)):
        nLetter = ord(sSentence[i:i+1])
        nStor2 = i
        if nLetter == 32:
            sWord = sSentence[nStor1:nStor2]
            if sWord[len(sWord)-3:len(sWord)] == "n't":
                sWord = sWord[0:len(sWord)-3]
                lSen.append(sWord)
                lSen.append(sNegate)
            else:
                lSen.append(sSentence[nStor1:nStor2])
            nStor1 = nStor2+1
        else:
            if nLetter == 44 or nLetter == 46:
                lSen.append(sSentence[nStor1:nStor2])
                nStor1 = nStor2+1
    lSen= [item.lower() for item in lSen]
    sEmpty = ""
    sEmpty in (
    print(lSen)
    return lSen

def sentiment(tokenizer):
    lSen = tokenizer()
    num_pos_tweets = 0
    num_neg_tweets = 0
    positive_words = []
    with open('positive-words.txt') as inputfile:
        for line in inputfile:
            positive_words.append(line.strip())

    negative_words = []
    with open('negative-words.txt') as inputfile:
        for line in inputfile:
            positive_words.append(line.strip())

    for x in range (0, len(lSen)):
        for i in range(0, len(positive_words)):
            if lSen[x] == positive_words[i]:
                num_pos_tweets = num_pos_tweets + 1

    for z in range (0, len(lSen)):
        for y in range(0, len(negative_words)):
            if lSen[x] == negative_words[i]:
                num_neg_tweets = num_neg_tweets + 1
    print(str(num_pos_tweets))
    print(str(num_neg_tweets))
    
    if num_neg_tweets >= num_pos_tweets:
        print("This tweet is negative")
    else:
        print("this tweet is Positive")
tokenizer()
sentiment(tokenizer)

