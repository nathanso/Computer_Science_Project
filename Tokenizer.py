import nltk
import re
import time

content_array = ['Nathan is an awesome man.']

def tokenizer():
    try:
        for item in content_array:
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)
            print tagged

            wordSubject = nltk.ne_chunk(tagged)
            wordSubject.draw()
    except Exception, e:
        print "There is an error in the first loop"
        print str(e)

tokenizer()
