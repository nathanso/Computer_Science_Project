def tokenizer():
    print("Please Enter your Sentence")
    sSentence = input()
    lSen = []
    nStor1 = 0
    nStor2 = 0
    nLetter = ""
    for i in range (0,  len(sSentence)):
        nLetter = ord(sSentence[i:i+1])
        nStor2 = i
        if nLetter == 32:
            lSen.append(sSentence[nStor1:nStor2])
            nStor1 = nStor2+1
        else:
            if nLetter == (44 or 46):
                lSen.append(sSentence[nStor1:nStor2])
                nStor1 = nStor2+1

def sentiment(lSen):
    positive_words = []
    negative_words = []
    
