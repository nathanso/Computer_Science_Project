import re, math, itertools, pickle, collections, os
from collections import defaultdict, Counter
#import nltk.util, nltk.collocations, nltk.metrics

POLARITY_DATA_DIR = os.path.join('D:\\', 'rt-polaritydata') # File path of the Test Data

RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt') #Name of the positive test data
RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-neg.txt') #Name of the negative test data

_ADD_LOGS_MAX_DIFF = math.log(1e-30, 2)
_NINF = float('-1e300')


def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        try:
            initializer = next(it)
        except StopIteration:
            raise TypeError('reduce() of empty sequence with no initial value')
    accum_value = initializer
    for x in it:
        accum_value = function(accum_value, x)
    return accum_value

def add_logs(logx, logy):
    if (logx < logy + _ADD_LOGS_MAX_DIFF):
        return logy
    if (logy < logx + _ADD_LOGS_MAX_DIFF):
        return logx
    base = min(logx, logy)
    return base + math.log(2 ** (logx - base) + 2 ** (logy - base), 2)

def sum_logs(logs):
    return reduce(add_logs, logs[1:], logs[0])

class FDist(Counter): # A frequency Distribution class, with attributes from the Counter Library module.

    def __init__(self, sampleData=None): #Has sampleData being the
        Counter.__init__(self, sampleData) # Uses Counter class's instantiation

    def nTotal(self): #Return the total number of sample outcomes
        return sum(self.values())

    def wMore(self): #Returns the total amount of samples that is greater than 0
        return len(self)

    def wSingle(self): #Returns samples which only occurred once in the sampleData
        return [item for item in self if self[item] == 1]

    def wordFreq(self): # Returns a dictionary of samples and the frequency
        wordFreq = defaultdict(int)
        for value in self.values():
            wordFreq[value] += 1
        return wordFreq

    def cFreq(self, sampleData):# Returns the cumulative frequency of a sample
        cFreq = 0.0
        for sample in sampleData:
            cFreq += self[sample]
        return cFreq

    def maxWord(self):# Returns the most common sample in the SampleData
        return self.most_common(1)[0][0]

class PDist(): # A Super-Class the calculates the probability distribution of the sample

    SUM_TO_ONE = True # A boolean that displays that sum of all probability is one.

    def __init__(self): # As it is a Super-Class, it is suppose to be a framework for the Sub-Classes below this
        if self.__class__ == PDist:
            raise NotImplementedError("Interfaces can't be instantiated")

    def prob(self, sample): #Returns the probability of a sample
        raise NotImplementedError()

    def lProb(self, sample): # returns of the probability of a sample in log base 2
        prob = self.prob(sample)
        return (math.log(prob, 2)
                if prob != 0 else _NINF)

    def maxWords(self): # Returns the sample with the biggest probability
        raise NotImplementedError()

    def sampleData(self):
        raise NotImplementedError()

    def discount(self):
        return 0.0


class DictPDist(PDist): # A Sub-Class that calculates the probability of samples based on the dictionary produced

    def __init__(self, PDict=None, log=False, normalise=False):
        self.PDict = (PDict.copy() if PDict is not None else {}) # PDict is probability dictionary if there is already there
        self.log = log # Attribute that stores a boolean, based on the instantiated

        if normalise: # If normalise is true, then the log attribute is evaluated
            if log: # Evaluate log's value (whether is true or false)
                value_sum = sum_logs(list(self.PDict.values()))
                if value_sum <= _NINF:
                    logp = math.log(1.0 / len(PDict), 2)
                    for x in PDict:
                        self.PDict[x] = logp
                else:
                    for (x, p) in self.PDict.items():
                        self.PDict[x] -= value_sum
            else:
                value_sum = sum(self.PDict.values())
                if value_sum == 0:
                    p = 1.0 / len(PDict)
                    for x in PDict:
                        self.PDict[x] = p
                else:
                    norm_factor = 1.0 / value_sum
                    for (x, p) in self.PDict.items():
                        self.PDict[x] *= norm_factor

    def prob(self, sample):
        if self.log:
            return (2 ** (self.PDict[sample]) if sample in self.PDict else 0)
        else:
            return self.PDict.get(sample, 0)

    def lProb(self, sample):
        if self.log:
            return self.PDict.get(sample, _NINF)
        else:
            if (sample not in self.PDict) or (self.PDict[sample] == 0):
                return _NINF
            else:
                return math.log(self.PDict[sample], 2)

    def max(self):
        if hasattr(self, "max"):
            self.max = max((x, y) for (y, x) in self.PDict.items())[1]
        return self.max

    def samples(self):
        return self.PDict.keys()


class LidstoneProbDist(PDist):
    SUM_TO_ONE = False

    def __init__(self, FDist, gamma, bins=None):

        if (bins == 0) or (bins is None and FDist.nTotal() == 0):
            name = self.__class__.__name__[:-8]

        if (bins is not None) and (bins < FDist.wMore()):
            name = self.__class__.__name__[:-8]
        self.freqdist = FDist
        self.gamma = float(gamma)
        self.N = self.freqdist.nTotal()

        if bins is None:
            bins = FDist.wMore()
        self._bins = bins

        self.divisor = self.N + bins * gamma
        if self.divisor == 0.0:
            self.gamma = 0
            self.divisor = 1

    def freqdist(self):
        return self.freqdist

    def prob(self, sample):
        c = self.freqdist[sample]
        return (c + self.gamma) / self.divisor

    def max(self):
        return self.FDist.max()

    def samples(self):
        return self.freqdist.keys()

    def discount(self):
        gb = self.gamma * self.bins
        return gb / (self.N + gb)


class ELEProbDist(LidstoneProbDist):
    def __init__(self, FDist, bins=None):
        LidstoneProbDist.__init__(self, FDist, 0.5, bins)


class CondFDist(defaultdict):
    def __init__(self, cond_samples=None):
        defaultdict.__init__(self, FDist)
        if cond_samples:
            for (cond, sample) in cond_samples:
                self[cond][sample] += 1

    def __reduce__(self):
        kv_pairs = ((cond, self[cond]) for cond in self.conditions())
        return (self.__class__, (), None, None, kv_pairs)

    def conditions(self):
        return list(self.keys())
"""
    def N(self):
        return sum(FDist.nTotal() for FDist in compat.itervalues(self))
"""

class NaiveBayesClassifier():
    def __init__(self, label_probdist, feature_probdist):
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self.labels = list(label_probdist.samples())

    def labels(self):
        return self.labels

    def prob_classify(self, featureset):
        featureset = featureset.copy()
        for fname in list(featureset.keys()):
            for label in self.labels:
                if (label, fname) in self._feature_probdist:
                    break
            else:
                del featureset[fname]

        logprob = {}

        for label in self.labels:
            logprob[label] = self._label_probdist.lProb(label)

            for     label in self.labels:
                for (fname, fval) in featureset.items():
                    if (label, fname) in self._feature_probdist:
                        feature_probs = self._feature_probdist[label, fname]
                        if label not in logprob:
                            for label in self.labels:
                                logprob[label] = 0
                        logprob[label] = logprob[label] + feature_probs.lProb(fval)
                    else:
                        logprob[label] += sum_logs([])
        return DictPDist(logprob, log=True, normalise=True)

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def classify_many(self, featuresets):
        return [self.classify(fs) for fs in featuresets]

    def show_most_informative_features(self, n=10):
        # Determine the most relevant features, and display them.
        cpdist = self._feature_probdist
        print('Most Informative Features')

        for (fname, fval) in self.most_informative_features(n):
            def labelprob(l):
                return cpdist[l, fname].prob(fval)

            labels = sorted([l for l in self.labels
                             if fval in cpdist[l, fname].samples()],
                            key=labelprob)
            if len(labels) == 1:
                continue
            l0 = labels[0]
            l1 = labels[-1]
            if cpdist[l0, fname].prob(fval) == 0:
                ratio = 'INF'
            else:
                ratio = '%8.1f' % (cpdist[l1, fname].prob(fval) /
                                   cpdist[l0, fname].prob(fval))
            print(('%24s = %-14r %6s : %-6s = %s : 1.0' %
                   (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))

    def most_informative_features(self, n=100):


        features = set()
        maxprob = defaultdict(lambda: 0.0)
        minprob = defaultdict(lambda: 1.0)

        for (label, fname), probdist in self._feature_probdist.items():
            for fval in probdist.samples():
                feature = (fname, fval)
                features.add(feature)
                p = probdist.prob(fval)
                maxprob[feature] = max(p, maxprob[feature])
                minprob[feature] = min(p, minprob[feature])
                if minprob[feature] == 0:
                    features.discard(feature)

        features = sorted(features,
                          key=lambda feature_:
                          minprob[feature_] / maxprob[feature_])
        return features[:n]

    @staticmethod
    def train(labeled_featuresets, estimator=ELEProbDist):
        label_freqdist = FDist()
        feature_freqdist = defaultdict(FDist)
        feature_values = defaultdict(set)
        fnames = set()

        for featureset, label in labeled_featuresets:
            label_freqdist[label] += 1
            for fname, fval in featureset.items():
                # Increment freq(fval|label, fname)
                feature_freqdist[label, fname][fval] += 1
                # Record that fname can take the value fval.
                feature_values[fname].add(fval)
                # Keep a list of all feature names.
                fnames.add(fname)

        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].nTotal()
                if num_samples - count > 0:
                    feature_freqdist[label, fname][None] += num_samples - count
                    feature_values[fname].add(None)
        label_probdist = estimator(label_freqdist)
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label, fname] = probdist

        return NaiveBayesClassifier(label_probdist, feature_probdist)

def save_classifier(classifier):
    f = open('my_classifier.pickle', 'wb')
    pickle.dump(classifier, f, -1)
    f.close()

def save_word_scores(word_score):
    f = open('word_scores.pickle', 'wb')
    pickle.dump(ws, f, -1)
    f.close()

def load_classifier():
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier

def load_word_scores():
    f = open('word_scores.pickle', 'rb')
    word_scores = pickle.load(f)
    f.close()
    return word_scores

def evaluate_features(feature_select):
    posFeatures = []
    negFeatures = []
    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords = [feature_select(posWords), 'pos']
            posFeatures.append(posWords)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords = [feature_select(negWords), 'neg']
            negFeatures.append(negWords)
    posCutoff = int(math.floor(len(posFeatures) * 3 / 4))
    negCutoff = int(math.floor(len(negFeatures) * 3 / 4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
    classifier = NaiveBayesClassifier.train(trainFeatures)
    classifier1 = save_classifier(classifier)
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)
    """
    print('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
    print('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
    print('pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos']))
    print('pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos']))
    print('neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg']))
    print('neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg']))
    classifier.show_most_informative_features(10)
    """
def _contingency(n_ii, n_ix_xi_tuple, n_xx):
    (n_ix, n_xi) = n_ix_xi_tuple
    n_oi = n_xi - n_ii
    n_io = n_ix - n_ii
    return (n_ii, n_oi, n_io, n_xx - n_ii - n_oi - n_io)

def _marginals(n_ii, n_oi, n_io, n_oo):
    return (n_ii, (n_oi + n_ii, n_io + n_ii), n_oo + n_oi + n_io + n_ii)

def phi_sq(*marginals):
    n_ii, n_io, n_oi, n_oo = _contingency(*marginals)
    return float((n_ii*n_oo - n_io*n_oi)**2) /((n_ii + n_io) * (n_ii + n_oi) * (n_io + n_oo) * (n_oi + n_oo))

def chi_sq(n_ii, n_ix_xi_tuple, n_xx):
    (n_ix, n_xi) = n_ix_xi_tuple
    return n_xx *phi_sq(n_ii, (n_ix, n_xi), n_xx)


def make_full_dict(words):
    return dict([(word, True) for word in words])

def create_word_scores():
    posWords = []
    negWords = []
    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords.append(posWord)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords.append(negWord)
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))
    word_fd = FDist()
    cond_word_fd = CondFDist()
    for word in posWords:
        word_fd[word.lower()] += 1
        cond_word_fd['pos'][word.lower()] += 1
    for word in negWords:
        word_fd[word.lower()] += 1
        cond_word_fd['neg'][word.lower()] += 1
    pos_word_count = cond_word_fd['pos'].nTotal()
    neg_word_count = cond_word_fd['neg'].nTotal()
    total_word_count = pos_word_count + neg_word_count
    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    return word_scores

def find_best_words(word_scores, number):
    best_vals = sorted(word_scores, key=lambda s: s[1], reverse=True)[:number]
    best_words = set([s[0] for s in best_vals])
    return best_words

def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

print("what is your review?")
review = input()

if os.path.exists("word_scores.pickle"):
    classifier2 = load_classifier()
    word_scores = load_word_scores().items()
    best_words = find_best_words(word_scores, 15000)
    print(classifier2.classify(best_word_features(review.split())))

else:

    ws = create_word_scores()
    word_scores = save_word_scores(ws)
    word_score = create_word_scores().items()
    numbers_to_test = [15000]
    for num in numbers_to_test:
        print('evaluating best %d word features' % (num))
        best_words = find_best_words(word_score, num)

        evaluate_features(best_word_features)
    classifier2 = load_classifier()
    print(classifier2.classify(best_word_features(review.split())))
