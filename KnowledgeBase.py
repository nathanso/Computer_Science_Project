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
cur.execute("CREATE TABLE RssFeed(Id INT, Title TEXT, Description TEXT, Link TEXT)")

def database():
    try:
        page = 'http://www.huffingtonpost.co.uk/feeds/verticals/uk-politics/index.xml'
        sourceCode = opener.open(page).read()
        try:
            titles = re.findall(r'<title>(.*?)</title>',sourceCode)
            descriptions = re.findall(r'<description>(.*?)</description>',sourceCode)
            links = re.findall(r'<link>(.*?)</link>',sourceCode)

            for x in titles:
<<<<<<< HEAD
                cur.execute('INSERT INTO RssFeed VALUES(x,titles[x], description[x], links[x])')
=======
                cur.execute("INSERT INTO RssFeed VALUES(?, ?, ?, ?)"(x,titles[x], description[x], links[x]))
>>>>>>> origin/master


        except Exception, e:
            print"Failed in the 2nd Loop of Main function"
            print str(e)

    except Exception, e:
        print"Failed in the 1st Loop of Main function"
        print str(e)

database()
