import nltk
import time
import urllib2
from urllib2 import urlopen
import cookielib
from cookielib import CookieJar
import datetime
import re
import sqlite3

cJ = CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cJ))
opener.addheaders = [("User-agents","Google Chrome/39.0")]

conn = sqlite3.connect("knowledgeBase.db")
cur = conn.cursor()
cur.execute("CREATE TABLE RssFeed(Id INT, Title TEXT, Link TEXT)")
table = []

def processor(data):
    try:
        tokenized = nltk.word_tokenize(data)
        tagged = nltk.pos_tag(tokenized)
        namedEnt = nltk.ne_chunk(tagged, binary = True)

        entities = re.findall(r'NE\s(.*?)/', str(namedEnt))
        descriptives = re.findall(r'\(\'(\w*)\',\s\'JJ\w?\'', str(tagged))
        if len(entities) > 1:
            pass
        elif len(entities) == 0:
            pass
        else:
            print( '_____________________')
            print ('Named:', entities[0])
            print ('Descriptions:')
            for eachDesc in descriptives:
                print (eachDesc)                

    except Exception as e:
        print"error in the main Proccessor loop"
        print str(e)

def database():
    try:
        page = 'http://www.huffingtonpost.co.uk/feeds/verticals/uk-politics/index.xml'
        sourceCode = opener.open(page).read()
        try:
            x = 1
            Titles = re.findall(r'<title>(.*?)</title>',sourceCode)
            #descriptions = re.findall(r'<description>(.*?)</description>',sourceCode)
            links = re.findall(r'<link>(.*?)</link>',sourceCode)
            for eachTitle in Titles:
                cur.execute("INSERT INTO RssFeed(Id, Title, Link) VALUES(?,?,?)",(x, Titles[x-1],links[x-1]))
            time.sleep(555)
            print ('finish')

        except Exception as e:
            print"Failed in the 2nd Loop of Main function"
            print (str(e))

    except Exception as e:
        print"Failed in the 1st Loop of Main function"
        print (str(e))

database()
