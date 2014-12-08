import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time



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

print('What is your Movie')
sUser = input()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
twitterStream = Stream(auth, listener())
twitterStream.filter(track = [sUser])
print('finish')
