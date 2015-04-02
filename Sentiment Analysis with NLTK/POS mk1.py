import collections
from collections import Counter
from pickle import dump,load
import re
import math
import random
import csv
from nltk.corpus import names


class ProbDistI(object):
   
    SUM_TO_ONE = True
    """True if the probabilities of the samples in this probability
       distribution will always sum to one."""

    def __init__(self):
        if self.__class__ == ProbDistI:
            raise NotImplementedError("Interfaces can't be instantiated")

    def prob(self, sample):
        """
        Return the probability for a given sample.  Probabilities
        are always real numbers in the range [0, 1].

        :param sample: The sample whose probability
               should be returned.
        :type sample: any
        :rtype: float
        """
        raise NotImplementedError()

    def logprob(self, sample):
        """
        Return the base 2 logarithm of the probability for a given sample.

        :param sample: The sample whose probability
               should be returned.
        :type sample: any
        :rtype: float
        """
        # Default definition, in terms of prob()
        p = self.prob(sample)
        return (math.log(p, 2) if p != 0 else _NINF)

    def max(self):
        """
        Return the sample with the greatest probability.  If two or
        more samples have the same probability, return one of them;
        which sample is returned is undefined.

        :rtype: any
        """
        raise NotImplementedError()

    def samples(self):
        """
        Return a list of all samples that have nonzero probabilities.
        Use ``prob`` to find the probability of each sample.

        :rtype: list
        """
        raise NotImplementedError()

    # cf self.SUM_TO_ONE
    def discount(self):
        """
        Return the ratio by which counts are discounted on average: c*/c

        :rtype: float
        """
        return 0.0

    # Subclasses should define more efficient implementations of this,
    # where possible.
    def generate(self):
        """
        Return a randomly selected sample from this probability distribution.
        The probability of returning each sample ``samp`` is equal to
        ``self.prob(samp)``.
        """
        p = random.random()
        p_init = p
        for sample in self.samples():
            p -= self.prob(sample)
            if p <= 0: return sample
        # allow for some rounding error:
        if p < .0001:
            return sample
        # we *should* never get here
        if self.SUM_TO_ONE:
            warnings.warn("Probability distribution %r sums to %r; generate()"
                          " is returning an arbitrary sample." % (self, p_init-p))
        return random.choice(list(self.samples()))

class LidstoneProbDist(ProbDistI):
    """
    The Lidstone estimate for the probability distribution of the
    experiment used to generate a frequency distribution.  The
    "Lidstone estimate" is parameterized by a real number *gamma*,
    which typically ranges from 0 to 1.  The Lidstone estimate
    approximates the probability of a sample with count *c* from an
    experiment with *N* outcomes and *B* bins as
    ``c+gamma)/(N+B*gamma)``.  This is equivalent to adding
    *gamma* to the count for each bin, and taking the maximum
    likelihood estimate of the resulting frequency distribution.
    """
    SUM_TO_ONE = False
    def __init__(self, freqdist, gamma, bins=None):
        """
        Use the Lidstone estimate to create a probability distribution
        for the experiment used to generate ``freqdist``.

        :type freqdist: FreqDist
        :param freqdist: The frequency distribution that the
            probability estimates should be based on.
        :type gamma: float
        :param gamma: A real number used to parameterize the
            estimate.  The Lidstone estimate is equivalent to adding
            *gamma* to the count for each bin, and taking the
            maximum likelihood estimate of the resulting frequency
            distribution.
        :type bins: int
        :param bins: The number of sample values that can be generated
            by the experiment that is described by the probability
            distribution.  This value must be correctly set for the
            probabilities of the sample values to sum to one.  If
            ``bins`` is not specified, it defaults to ``freqdist.B()``.
        """
        if (bins == 0) or (bins is None and freqdist.N() == 0):
            name = self.__class__.__name__[:-8]
            raise ValueError('A %s probability distribution ' % name +
                             'must have at least one bin.')
        if (bins is not None) and (bins < freqdist.B()):
            name = self.__class__.__name__[:-8]
            raise ValueError('\nThe number of bins in a %s distribution ' % name +
                             '(%d) must be greater than or equal to\n' % bins +
                             'the number of bins in the FreqDist used ' +
                             'to create it (%d).' % freqdist.B())

        self._freqdist = freqdist
        self._gamma = float(gamma)
        self._N = self._freqdist.N()

        if bins is None:
            bins = freqdist.B()
        self._bins = bins

        self._divisor = self._N + bins * gamma
        if self._divisor == 0.0:
            # In extreme cases we force the probability to be 0,
            # which it will be, since the count will be 0:
            self._gamma = 0
            self._divisor = 1

    def freqdist(self):
        """
        Return the frequency distribution that this probability
        distribution is based on.

        :rtype: FreqDist
        """
        return self._freqdist

    def prob(self, sample):
        c = self._freqdist[sample]
        return (c + self._gamma) / self._divisor

    def max(self):
        # For Lidstone distributions, probability is monotonic with
        # frequency, so the most probable sample is the one that
        # occurs most frequently.
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def discount(self):
        gb = self._gamma * self._bins
        return gb / (self._N + gb)

    def __repr__(self):
        """
        Return a string representation of this ``ProbDist``.

        :rtype: str
        """
        return '<LidstoneProbDist based on %d samples>' % self._freqdist.N()


class ELEProbDist(LidstoneProbDist):
    """
    The expected likelihood estimate for the probability distribution
    of the experiment used to generate a frequency distribution.  The
    "expected likelihood estimate" approximates the probability of a
    sample with count *c* from an experiment with *N* outcomes and
    *B* bins as *(c+0.5)/(N+B/2)*.  This is equivalent to adding 0.5
    to the count for each bin, and taking the maximum likelihood
    estimate of the resulting frequency distribution.
    """
    def __init__(self, freqdist, bins=None):
        """
        Use the expected likelihood estimate to create a probability
        distribution for the experiment used to generate ``freqdist``.

        :type freqdist: FreqDist
        :param freqdist: The frequency distribution that the
            probability estimates should be based on.
        :type bins: int
        :param bins: The number of sample values that can be generated
            by the experiment that is described by the probability
            distribution.  This value must be correctly set for the
            probabilities of the sample values to sum to one.  If
            ``bins`` is not specified, it defaults to ``freqdist.B()``.
        """
        LidstoneProbDist.__init__(self, freqdist, 0.5, bins)

    def __repr__(self):
        """
        Return a string representation of this ``ProbDist``.

        :rtype: str
        """
        return '<ELEProbDist based on %d samples>' % self._freqdist.N()



class ClassifierI(object):

    def labels(self):
        """
        :return: the list of category labels used by this classifier.
        :rtype: list of (immutable)
        """
        raise NotImplementedError()

    def classify(self, featureset):
        """
        :return: the most appropriate label for the given featureset.
        :rtype: label
        """
        if overridden(self.classify_many):
            return self.classify_many([featureset])[0]
        else:
            raise NotImplementedError()

    def prob_classify(self, featureset):
        """
        :return: a probability distribution over labels for the given
            featureset.
        :rtype: ProbDistI
        """
        if overridden(self.prob_classify_many):
            return self.prob_classify_many([featureset])[0]
        else:
            raise NotImplementedError()

    def classify_many(self, featuresets):
        """
        Apply ``self.classify()`` to each element of ``featuresets``.  I.e.:

            return [self.classify(fs) for fs in featuresets]

        :rtype: list(label)
        """
        return [self.classify(fs) for fs in featuresets]

    def prob_classify_many(self, featuresets):
        """
        Apply ``self.prob_classify()`` to each element of ``featuresets``.  I.e.:

            return [self.prob_classify(fs) for fs in featuresets]

        :rtype: list(ProbDistI)
        """
        return [self.prob_classify(fs) for fs in featuresets]


class NaiveBayesClassifier(ClassifierI):
    """
    A Naive Bayes classifier.  Naive Bayes classifiers are
    paramaterized by two probability distributions:

      - P(label) gives the probability that an input will receive each
        label, given no information about the input's features.

      - P(fname=fval|label) gives the probability that a given feature
        (fname) will receive a given value (fval), given that the
        label (label).

    If the classifier encounters an input with a feature that has
    never been seen with any label, then rather than assigning a
    probability of 0 to all labels, it will ignore that feature.

    The feature value 'None' is reserved for unseen feature values;
    you generally should not use 'None' as a feature value for one of
    your own features.
    """
    def __init__(self, label_probdist, feature_probdist):
        """
        :param label_probdist: P(label), the probability distribution
            over labels.  It is expressed as a ``ProbDistI`` whose
            samples are labels.  I.e., P(label) =
            ``label_probdist.prob(label)``.

        :param feature_probdist: P(fname=fval|label), the probability
            distribution for feature values, given labels.  It is
            expressed as a dictionary whose keys are ``(label, fname)``
            pairs and whose values are ``ProbDistI`` objects over feature
            values.  I.e., P(fname=fval|label) =
            ``feature_probdist[label,fname].prob(fval)``.  If a given
            ``(label,fname)`` is not a key in ``feature_probdist``, then
            it is assumed that the corresponding P(fname=fval|label)
            is 0 for all values of ``fval``.
        """
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self._labels = list(label_probdist.samples())

    def labels(self):
        return self._labels

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def prob_classify(self, featureset):
        featureset = featureset.copy()
        for fname in list(featureset.keys()):
            for label in self._labels:
                if (label, fname) in self._feature_probdist:
                    break
            else:
                del featureset[fname]

        logprob = {}
        for label in self._labels:
            logprob[label] = self._label_probdist.logprob(label)

        for label in self._labels:
            for (fname, fval) in featureset.items():
                if (label, fname) in self._feature_probdist:
                    feature_probs = self._feature_probdist[label,fname]
                    logprob[label] += feature_probs.logprob(fval)
                else:
                    logprob[label] += sum_logs([]) # = -INF.

        return DictionaryProbDist(logprob, normalize=True, log=True)

    def show_most_informative_features(self, n=10):
        
        cpdist = self._feature_probdist
        print('Most Informative Features')

        for (fname, fval) in self.most_informative_features(n):
            def labelprob(l):
                return cpdist[l,fname].prob(fval)

            labels = sorted([l for l in self._labels
                             if fval in cpdist[l,fname].samples()],
                            key=labelprob)
            if len(labels) == 1: continue
            l0 = labels[0]
            l1 = labels[-1]
            if cpdist[l0,fname].prob(fval) == 0:
                ratio = 'INF'
            else:
                ratio = '%8.1f' % (cpdist[l1,fname].prob(fval) /
                                  cpdist[l0,fname].prob(fval))
            print(('%24s = %-14r %6s : %-6s = %s : 1.0' %
                   (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))

    def most_informative_features(self, n=100):
               
        features = set()
        maxprob = defaultdict(lambda: 0.0)
        minprob = defaultdict(lambda: 1.0)

        for (label, fname), probdist in self._feature_probdist.items():
            for fval in probdist.samples():
                feature = (fname, fval)
                features.add( feature )
                p = probdist.prob(fval)
                maxprob[feature] = max(p, maxprob[feature])
                minprob[feature] = min(p, minprob[feature])
                if minprob[feature] == 0:
                    features.discard(feature)

        features = sorted(features,
            key=lambda feature_: minprob[feature_]/maxprob[feature_])
        return features[:n]

    @staticmethod
    def train(labeled_featuresets, estimator=ELEProbDist):
         label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)
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
                count = feature_freqdist[label, fname].N()
                # Only add a None key when necessary, i.e. if there are
                # any samples with feature 'fname' missing.
                if num_samples - count > 0:
                    feature_freqdist[label, fname][None] += num_samples - count
                    feature_values[fname].add(None)

        # Create the P(label) distribution
        label_probdist = estimator(label_freqdist)

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label,fname] = probdist

        return NaiveBayesClassifier(label_probdist, feature_probdist)


class ProbDistI(object):
    SUM_TO_ONE = True
    """True if the probabilities of the samples in this probability
       distribution will always sum to one."""

    def prob(self, sample):
        raise NotImplementedError()

    def logprob(self, sample):
        # Default definition, in terms of prob()
        p = self.prob(sample)
        return (math.log(p, 2) if p != 0 else _NINF)

    def max(self):
        raise NotImplementedError()

    def samples(self):
        raise NotImplementedError()

    # cf self.SUM_TO_ONE
    def discount(self):
        return 0.0

    # Subclasses should define more efficient implementations of this,
    def generate(self):
        p = random.random()
        p_init = p
        for sample in self.samples():
            p -= self.prob(sample)
            if p <= 0: return sample
        # allow for some rounding error:
        if p < .0001:
            return sample
        # we *should* never get here
        if self.SUM_TO_ONE:
            warnings.warn("Probability distribution %r sums to %r; generate()"
                          " is returning an arbitrary sample." % (self, p_init-p))
        return random.choice(list(self.samples()))



class DictionaryProbDist(ProbDistI):
    def __init__(self, prob_dict=None, log=False, normalize=False):
        self._prob_dict = (prob_dict.copy() if prob_dict is not None else {})
        self._log = log

        # Normalize the distribution, if requested.
        if normalize:
            if len(prob_dict) == 0:
                raise ValueError('A DictionaryProbDist must have at least one sample ' +
                             'before it can be normalized.')
            if log:
                value_sum = sum_logs(list(self._prob_dict.values()))
                if value_sum <= _NINF:
                    logp = math.log(1.0/len(prob_dict), 2)
                    for x in prob_dict:
                        self._prob_dict[x] = logp
                else:
                    for (x, p) in self._prob_dict.items():
                        self._prob_dict[x] -= value_sum
            else:
                value_sum = sum(self._prob_dict.values())
                if value_sum == 0:
                    p = 1.0/len(prob_dict)
                    for x in prob_dict:
                        self._prob_dict[x] = p
                else:
                    norm_factor = 1.0/value_sum
                    for (x, p) in self._prob_dict.items():
                        self._prob_dict[x] *= norm_factor


    def prob(self, sample):
        if self._log:
            return (2**(self._prob_dict[sample]) if sample in self._prob_dict else 0)
        else:
            return self._prob_dict.get(sample, 0)

    def logprob(self, sample):
        if self._log:
            return self._prob_dict.get(sample, _NINF)
        else:
            if sample not in self._prob_dict: return _NINF
            elif self._prob_dict[sample] == 0: return _NINF
            else: return math.log(self._prob_dict[sample], 2)

    def max(self):
        if not hasattr(self, '_max'):
            self._max = max((p,v) for (v,p) in self._prob_dict.items())[1]
        return self._max
    def samples(self):
        return self._prob_dict.keys()
    def __repr__(self):
        return '<ProbDist with %d samples>' % len(self._prob_dict)

def names_demo_features(name):
            features = {}
            features['alwayson'] = True
            features['startswith'] = name[0].lower()
            features['endswith'] = name[-1].lower()
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                features['count(%s)' % letter] = name.lower().count(letter)
                features['has(%s)' % letter] = letter in name.lower()
            return features


def names_demo(trainer, features=names_demo_features(name)):
    
    # Construct a list of classified names, using the names corpus.
    namelist = ([(name, 'male') for name in names.words('male.txt')] +
                [(name, 'female') for name in names.words('female.txt')])

    # Randomly split the names into a test & train set.
    random.seed(123456)
    random.shuffle(namelist)
    train = namelist[:5000]
    test = namelist[5000:5500]

    # Train up a classifier.
    print('Training classifier...')
    classifier = trainer( [(features(n), g) for (n,g) in train] )

    # Run the classifier on the test data.
    print('Testing classifier...')
    acc = accuracy(classifier, [(features(n),g) for (n,g) in test])
    print('Accuracy: %6.4f' % acc)

    # For classifiers that can find probabilities, show the log
    # likelihood and some sample probability distributions.
    try:
        test_featuresets = [features(n) for (n,g) in test]
        pdists = classifier.prob_classify_many(test_featuresets)
        ll = [pdist.logprob(gold)
              for ((name, gold), pdist) in zip(test, pdists)]
        print('Avg. log likelihood: %6.4f' % (sum(ll)/len(test)))
        print()
        print('Unseen Names      P(Male)  P(Female)\n'+'-'*40)
        for ((name, gender), pdist) in list(zip(test, pdists))[:5]:
            if gender == 'male':
                fmt = '  %-15s *%6.4f   %6.4f'
            else:
                fmt = '  %-15s  %6.4f  *%6.4f'
            print(fmt % (name, pdist.prob('male'), pdist.prob('female')))
    except NotImplementedError:
        pass

    # Return the classifier
    return classifier


    

def main():
    filename = 'pima-indians-diabetes.data.csv'
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split '+str(len(dataset))+" rows into train= "+str(len(trainingSet))+' and test '+str(len(testSet))+ 'rows')
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy:'+str(accuracy))

def loadtagger(taggerfilename):
        infile = open(taggerfilename,'rb')
        tagger = load(infile); infile.close()
        return tagger

def traintag(corpusname, corpus):
        # Function to save tagger.
        def savetagger(tagfilename,tagger):
                outfile = open(tagfilename, 'wb')
                dump(tagger,outfile,-1); outfile.close()
                return
        # Training UnigramTagger.
        uni_tag = ut(corpus)
        savetagger(corpusname+'_unigram.tagger',uni_tag)
        # Training BigramTagger.
        bi_tag = bt(corpus)
        savetagger(corpusname+'_bigram.tagger',bi_tag)
        print("Tagger trained with",corpusname,"using" +\
                                "UnigramTagger and BigramTagger.")
        return

# Function to unchunk corpus.
def unchunk(corpus):
        nomwe_corpus = []
        for i in corpus:
                nomwe = " ".join([j[0].replace("_"," ") for j in i])
                nomwe_corpus.append(nomwe.split())
        return nomwe_corpus

class browntag():
        def __init__(self,mwe=True):
                self.mwe = mwe
                # Train tagger if it's used for the first time.
                try:
                        loadtagger('brown_unigram.tagger').tag(['estoy'])
                        loadtagger('brown_bigram.tagger').tag(['estoy'])
                except IOError:
                        print("*** First-time use of brown tagger ***")
                        print("Training tagger ...")
                        from nltk.corpus import brown as brown
                        brown_sents = brown.tagged_sents()
                        traintag('brown',brown_sents)
                        # Trains the tagger with no MWE.
                        brown_nomwe = unchunk(brown.tagged_sents())
                        tagged_brown_nomwe = batch_pos_tag(brown_nomwe)
                        traintag('brown_nomwe',tagged_brown_nomwe)
                        print
                # Load tagger.
                if self.mwe == True:
                        self.uni = loadtagger('brown_unigram.tagger')
                        self.bi = loadtagger('brown_bigram.tagger')
                elif self.mwe == False:
                        self.uni = loadtagger('brown_nomwe_unigram.tagger')
                        self.bi = loadtagger('brown_nomwe_bigram.tagger')

def pos_tag(tokens, mmwe=True):
        tagger = browntag(mmwe)
        return tagger.uni.tag(tokens)

def batch_pos_tag(sentences, mmwe=True):
        tagger = browntag(mmwe)
        return tagger.uni.tag_sents(sentences)

def biGrams():
    print('Enter a sentence')
    sSent = input()
    lBigrams = []
    lSent =re.findall(r"[\w']+|[.,!?;]",sSent)
    for i in range(len(lSent)-1):
        lBigrams.append((lSent[i]+" "+lSent[i+1]))
    return lBigrams

def tokenizer ():
    print('Enter Some sentences')
    sSent = input()
    #lTokens = sSent.split(".")
    #for x in range (0, len(lTokens)):
        #sSent = lTokens[x]
        #del lTokens[x]

    lTokens = re.findall(r"[\w']+|[.,!?;]",sSent)
    #lTokens.remove([])
    print(lTokens)
    return lTokens

main()
tagger = browntag()
print (tagger.uni.tag(biGrams()))
'''
