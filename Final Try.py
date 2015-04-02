class AnalyzeTweets(object):

    def __init__(self, tArray):
        self.tArray = tArray #tArray stands for the tokenized tweet array that will be fed into it

        self.positive_words = [] #imports positive words list into the self.positive_words array
        with open('positive-words.txt') as inputfile:
            for line in inputfile:
                self.positive_words.append(line.strip())

        self.negative_words = [] #imports negative words list into the self.negative_words array
        with open('negative-words.txt') as inputfile:
            for line in inputfile:
                self.negative_words.append(line.strip())

        self.stop_words = [] #imports stop words list into the self.stop_words array
        with open('stop-words.txt') as inputfile:
            for line in inputfile:
                self.stop_words.append(line.strip())


    def sentiment(self):
        num_pos_tweets = 0 #used to count the amount of positive tokens in the tweet arrary in tArray
        num_neg_tweets = 0 #used to count the amount of negative tokens in the tweet arrary in tArray
        global overall_ratings #overall rating is global variable and is counts the overall sentiment of the tweets, as per the functional requirement

        for Tokens in self.tArray: #for loop in variables in tArray
            for sWords in self.stop_words: #nested for loop in variables in self.stop_words
                if sWords == Tokens: #if condition to find matches in both arrays, if found it is removed
                    self.tArray.remove(sWords)


        for Tokens in self.tArray: #for loop in variables in tArray
            for pWords in self.positive_words: #nested for loop in variables in self.positive_words
                if pWords == Tokens: #if condition to find matches in both arrays, if found the num_pos_tweets counter is incremented
                    num_pos_tweets = num_pos_tweets + 1


        for Tokens in self.tArray: #for loop in variables in tArray
            for nWords in self.negative_words: #nested for loop in variables in self.negative_words
                if nWords == Tokens: #if condition to find matches in both arrays, if found the num_neg_tweets counter is incremented
                    num_neg_tweets = num_neg_tweets + 1

        if num_neg_tweets < num_pos_tweets: #if condition to compare the value of num_neg_tweets and num_pos_tweets, if num_pos_tweets is bigger overall_rating increments
                overall_ratings = overall_ratings + 1


        return overall_ratings

        def Most_Informative_Features (self): #stores the most informative features of each tweets
            pos_word_list = []
            neg_word_list = []
            global Features #Features is a global array

            for Tokens in lSen:
                for pWords in self.positive_words:
                    if Tokens == pWords: #When a match is found, it is appended to the pos_word_list
                        pos_word_list.append(Tokens)
            for Tokens in lSen:
                for nWords in self.negative_words:
                    if Tokens == nWords:#When a match is found, it is appended to the neg_word_list
                        neg_word_list.append(Tokens)

            features.append([pos_word_list,neg_word_list]) #appends the pos_word_list array and neg_word_list array to the feature array as one single array

            return features
