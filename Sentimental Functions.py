import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import nltk



consumer_key = 'KG2opAGBUTm7WOh4kxRH4hkv0'
consumer_secret = 'yMnAZmUJNgPMG52aJppODuCmumiLJAKaofRdOnTc3fmD2BQxmV'
access_token = '2797292249-DpQSv3x81hHuPQCSgyyowyWiQk5nWFQUHz94quG'
access_secret = '0kK2PLwxdiXG7PAyZgBGptvqUO9Aiz5hPsQZHVecqdh8Q'


class listener(StreamListener):

    def on_data(self, data):
        tweet = data.split(',"text":"')[1].split('","source')[0]
        tweetData = str(time.time()) + '::' + tweet
        saveFile = open('TwitterDB.csv','a')
        saveFile.write(tweetData)
        saveFile.write('\n')
        saveFile.close()

    def on_error(self, status):
        print (status) 

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
    
    print(lSen)
    return lSen

def sentiment(tokenizer):
    lSen = tokenizer()
    num_pos_tweets = 0
    num_neg_tweets = 0
    pos_position_list = []
    neg_position_list = []
    nSen = len(lSen)
    
    positive_words = []
    with open('positive-words.txt') as inputfile:
        for line in inputfile:
            positive_words.append(line.strip())

    negative_words = []
    with open('negative-words.txt') as inputfile:
        for line in inputfile:
            positive_words.append(line.strip())

    for x in range (0, nSen):
        for i in range(0, len(positive_words)):
            if lSen[x] == positive_words[i]:
                num_pos_tweets = num_pos_tweets + 1
                pos_position_list.append(x)    

    for z in range (0, nSen):
        for y in range(0, len(negative_words)):
            if lSen[z] == negative_words[i]:
                num_neg_tweets = num_neg_tweets + 1
                neg_position_list.append(z)
    
    for g in range ( 0, len(pos_position_list)):
        for h in range ( 0, nSen):
            if lSen(pos_position_list(g)-1) == "n't":
                num_neg_tweets = num_neg_tweets + 1
                num_pos_tweets = num_pos_tweets-1
    
    if num_neg_tweets >= num_pos_tweets:
        print("This tweet is negative")
    else:
        print("this tweet is Positive")

print('What is your Movie')
sUser = input()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
twitterStream = Stream(auth, listener())
twitterStream.filter(track = [sUser])
print('finish')
