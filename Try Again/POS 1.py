import collections, itertools
from collections import Counter
import math as math
import nltk.classify.util, nltk.metrics
from nltk.corpus import movie_reviews, stopwords
from functools import reduce
import locale
import re
import types
import textwrap
import pydoc
import bisect
import os
import sys
import types
from functools import wraps
from itertools import islice, chain
from pprint import pprint
from collections import defaultdict, deque
from sys import version_info

from nltk.internals import slice_bounds, raise_unorderable_types
from nltk.compat import (class_types, text_type, string_types, total_ordering,
                         python_2_unicode_compatible, getproxies,
			 ProxyHandler, build_opener, install_opener,
			 HTTPPasswordMgrWithDefaultRealm,
			 ProxyBasicAuthHandler, ProxyDigestAuthHandler)


def ngrams(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n-1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n-1))

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def _rank_dists(ranks1, ranks2):
    ranks1 = dict(ranks1)
    ranks2 = dict(ranks2)
    for k in ranks1:
        try:
            yield k, ranks1[k] - ranks2[k]
        except KeyError:
            pass


def spearman_correlation(ranks1, ranks2):
    n = 0
    res = 0
    for k, d in _rank_dists(ranks1, ranks2):
        res += d * d
        n += 1
    try:
        return 1 - (6 * float(res) / (n * (n*n - 1)))
    except ZeroDivisionError:
        # Result is undefined if only one item is ranked
        return 0.0


def ranks_from_sequence(seq):
    return ((k, i) for i, k in enumerate(seq))


def ranks_from_scores(scores, rank_gap=1e-15):
    """Given a sequence of (key, score) tuples, yields each key with an
    increasing rank, tying with previous key's rank if the difference between
    their scores is less than rank_gap. Suitable for use as an argument to
    ``spearman_correlation``.
    """
    prev_score = None
    rank = 0
    for i, (key, score) in enumerate(scores):
        try:
            if abs(score - prev_score) > rank_gap:
                rank = i
        except TypeError:
            pass

        yield key, rank
        prev_score = score

_log2 = lambda x: _math.log(x, 2.0)
_ln = math.log

_product = lambda s: reduce(lambda x, y: x * y, s)

_SMALL = 1e-20

try:
    from scipy.stats import fisher_exact
except ImportError:
    def fisher_exact(*_args, **_kwargs):
        raise NotImplementedError

### Indices to marginals arguments:

NGRAM = 0
"""Marginals index for the ngram count"""

UNIGRAMS = -2
"""Marginals index for a tuple of each unigram count"""

TOTAL = -1
"""Marginals index for the number of words in the data"""


class NgramAssocMeasures(object):
    _n = 0

    @staticmethod
    def _contingency(*marginals):
        raise NotImplementedError("The contingency table is not available"
                                    "in the general ngram case")

    @staticmethod
    def _marginals(*contingency):
        """Calculates values of contingency table marginals from its values."""
        raise NotImplementedError("The contingency table is not available"
                                    "in the general ngram case")

    @classmethod
    def _expected_values(cls, cont):
        n_all = sum(cont)
        bits = [1 << i for i in range(cls._n)]

        # For each contingency table cell
        for i in range(len(cont)):
            # Yield the expected value
            yield (_product(sum(cont[x] for x in range(2 ** cls._n)
                                if (x & j) == (i & j))
                            for j in bits) /
                   float(n_all ** (cls._n - 1)))

    @staticmethod
    def raw_freq(*marginals):
        """Scores ngrams by their frequency"""
        return float(marginals[NGRAM]) / marginals[TOTAL]

    @classmethod
    def student_t(cls, *marginals):
        """Scores ngrams using Student's t test with independence hypothesis
        for unigrams, as in Manning and Schutze 5.3.1.
        """
        return ((marginals[NGRAM] -
                  _product(marginals[UNIGRAMS]) /
                  float(marginals[TOTAL] ** (cls._n - 1))) /
                (marginals[NGRAM] + _SMALL) ** .5)

    @classmethod
    def chi_sq(cls, *marginals):
        cont = cls._contingency(*marginals)
        exps = cls._expected_values(cont)
        return sum((obs - exp) ** 2 / (exp + _SMALL)
                   for obs, exp in zip(cont, exps))

    @staticmethod
    def mi_like(*marginals, **kwargs):
        return (marginals[NGRAM] ** kwargs.get('power', 3) /
                float(_product(marginals[UNIGRAMS])))

    @classmethod
    def pmi(cls, *marginals):
        """Scores ngrams by pointwise mutual information, as in Manning and
        Schutze 5.4.
        """
        return (_log2(marginals[NGRAM] * marginals[TOTAL] ** (cls._n - 1)) -
                _log2(_product(marginals[UNIGRAMS])))

    @classmethod
    def likelihood_ratio(cls, *marginals):
        """Scores ngrams using likelihood ratios as in Manning and Schutze 5.3.4.
        """
        cont = cls._contingency(*marginals)
        return (cls._n *
                sum(obs * _ln(float(obs) / (exp + _SMALL) + _SMALL)
                    for obs, exp in zip(cont, cls._expected_values(cont))))

    @classmethod
    def poisson_stirling(cls, *marginals):
        """Scores ngrams using the Poisson-Stirling measure."""
        exp = (_product(marginals[UNIGRAMS]) /
               float(marginals[TOTAL] ** (cls._n - 1)))
        return marginals[NGRAM] * (_log2(marginals[NGRAM] / exp) - 1)

    @classmethod
    def jaccard(cls, *marginals):
        """Scores ngrams using the Jaccard index."""
        cont = cls._contingency(*marginals)
        return float(cont[0]) / sum(cont[:-1])


class BigramAssocMeasures(NgramAssocMeasures):
    _n = 2

    @staticmethod
    def _contingency(n_ii, n_ix_xi_tuple, n_xx):
        (n_ix, n_xi) = n_ix_xi_tuple
        n_oi = n_xi - n_ii
        n_io = n_ix - n_ii
        return (n_ii, n_oi, n_io, n_xx - n_ii - n_oi - n_io)

    @staticmethod
    def _marginals(n_ii, n_oi, n_io, n_oo):
        return (n_ii, (n_oi + n_ii, n_io + n_ii), n_oo + n_oi + n_io + n_ii)

    @staticmethod
    def _expected_values(cont):
        """Calculates expected values for a contingency table."""
        n_xx = sum(cont)
        # For each contingency table cell
        for i in range(4):
            yield (cont[i] + cont[i ^ 1]) * (cont[i] + cont[i ^ 2]) / float(n_xx)

    @classmethod
    def phi_sq(cls, *marginals):
        n_ii, n_io, n_oi, n_oo = cls._contingency(*marginals)

        return (float((n_ii*n_oo - n_io*n_oi)**2) /
                ((n_ii + n_io) * (n_ii + n_oi) * (n_io + n_oo) * (n_oi + n_oo)))

    @classmethod
    def chi_sq(cls, n_ii, n_ix_xi_tuple, n_xx):

        (n_ix, n_xi) = n_ix_xi_tuple
        return n_xx * cls.phi_sq(n_ii, (n_ix, n_xi), n_xx)

    @classmethod
    def fisher(cls, *marginals):
        n_ii, n_io, n_oi, n_oo = cls._contingency(*marginals)

        (odds, pvalue) = fisher_exact([[n_ii, n_io], [n_oi, n_oo]], alternative='less')
        return pvalue

    @staticmethod
    def dice(n_ii, n_ix_xi_tuple, n_xx):
        """Scores bigrams using Dice's coefficient."""
        (n_ix, n_xi) = n_ix_xi_tuple
        return 2 * float(n_ii) / (n_ix + n_xi)


class TrigramAssocMeasures(NgramAssocMeasures):
    _n = 3

    @staticmethod
    def _contingency(n_iii, n_iix_tuple, n_ixx_tuple, n_xxx):
        (n_iix, n_ixi, n_xii) = n_iix_tuple
        (n_ixx, n_xix, n_xxi) = n_ixx_tuple
        n_oii = n_xii - n_iii
        n_ioi = n_ixi - n_iii
        n_iio = n_iix - n_iii
        n_ooi = n_xxi - n_iii - n_oii - n_ioi
        n_oio = n_xix - n_iii - n_oii - n_iio
        n_ioo = n_ixx - n_iii - n_ioi - n_iio
        n_ooo = n_xxx - n_iii - n_oii - n_ioi - n_iio - n_ooi - n_oio - n_ioo

        return (n_iii, n_oii, n_ioi, n_ooi,
                n_iio, n_oio, n_ioo, n_ooo)

class ClassifierI(object):
    def labels(self):
        raise NotImplementedError()

    def classify(self, featureset):
        if overridden(self.classify_many):
            return self.classify_many([featureset])[0]
        else:
            raise NotImplementedError()

    def prob_classify(self, featureset):
        if overridden(self.prob_classify_many):
            return self.prob_classify_many([featureset])[0]
        else:
            raise NotImplementedError()

    def classify_many(self, featuresets):
        return [self.classify(fs) for fs in featuresets]

    def prob_classify_many(self, featuresets):
        return [self.prob_classify(fs) for fs in featuresets]

class FreqDist(Counter):
    def __init__(self, samples=None):
        Counter.__init__(self, samples)

    def N(self):
        return sum(self.values())

    def B(self):
        return len(self)

    def hapaxes(self):
        return [item for item in self if self[item] == 1]


    def Nr(self, r, bins=None):
        return self.r_Nr(bins)[r]

    def r_Nr(self, bins=None):
        _r_Nr = defaultdict(int)
        for count in self.values():
            _r_Nr[count] += 1

        # Special case for Nr[0]:
        _r_Nr[0] = bins - self.B() if bins is not None else 0

        return _r_Nr

    def _cumulative_frequencies(self, samples):
        cf = 0.0
        for sample in samples:
            cf += self[sample]
            yield cf

    # slightly odd nomenclature freq() if FreqDist does counts and ProbDist does probs,
    # here, freq() does probs
    def freq(self, sample):
        if self.N() == 0:
            return 0
        return float(self[sample]) / self.N()

    def max(self):
        if len(self) == 0:
            raise ValueError('A FreqDist must have at least one sample before max is defined.')
        return self.most_common(1)[0][0]

    def plot(self, *args, **kwargs):
        try:
            import pylab
        except ImportError:
            raise ValueError('The plot function requires the matplotlib package (aka pylab). '
                         'See http://matplotlib.sourceforge.net/')

        if len(args) == 0:
            args = [len(self)]
        samples = [item for item, _ in self.most_common(*args)]

        cumulative = _get_kwarg(kwargs, 'cumulative', False)
        if cumulative:
            freqs = list(self._cumulative_frequencies(samples))
            ylabel = "Cumulative Counts"
        else:
            freqs = [self[sample] for sample in samples]
            ylabel = "Counts"
        # percents = [f * 100 for f in freqs]  only in ProbDist?

        pylab.grid(True, color="silver")
        if not "linewidth" in kwargs:
            kwargs["linewidth"] = 2
        if "title" in kwargs:
            pylab.title(kwargs["title"])
            del kwargs["title"]
        pylab.plot(freqs, **kwargs)
        pylab.xticks(range(len(samples)), [compat.text_type(s) for s in samples], rotation=90)
        pylab.xlabel("Samples")
        pylab.ylabel(ylabel)
        pylab.show()

    def tabulate(self, *args, **kwargs):
        if len(args) == 0:
            args = [len(self)]
        samples = [item for item, _ in self.most_common(*args)]

        cumulative = _get_kwarg(kwargs, 'cumulative', False)
        if cumulative:
            freqs = list(self._cumulative_frequencies(samples))
        else:
            freqs = [self[sample] for sample in samples]
        # percents = [f * 100 for f in freqs]  only in ProbDist?

        for i in range(len(samples)):
            print("%4s" % samples[i], end=' ')
        print()
        for i in range(len(samples)):
            print("%4d" % freqs[i], end=' ')
        print()

    def copy(self):
        return self.__class__(self)

    def __le__(self, other):
        if not isinstance(other, FreqDist):
            raise_unorderable_types("<=", self, other)
        return set(self).issubset(other) and all(self[key] <= other[key] for key in self)

    # @total_ordering doesn't work here, since the class inherits from a builtin class
    __ge__ = lambda self, other: not self <= other or self == other
    __lt__ = lambda self, other: self <= other and not self == other
    __gt__ = lambda self, other: not self <= other

    def __repr__(self):
        return self.pprint()

    def pprint(self, maxlen=10):
        items = ['{0!r}: {1!r}'.format(*item) for item in self.most_common(maxlen)]
        if len(self) > maxlen:
            items.append('...')
        return 'FreqDist({{{0}}})'.format(', '.join(items))

    def __str__(self):
        return '<FreqDist with %d samples and %d outcomes>' % (len(self), self.N())

class ProbDistI(object):
    SUM_TO_ONE = True
    """True if the probabilities of the samples in this probability
       distribution will always sum to one."""

    def __init__(self):
        if self.__class__ == ProbDistI:
            raise NotImplementedError("Interfaces can't be instantiated")

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
    # where possible.
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

class LidstoneProbDist(ProbDistI):
    SUM_TO_ONE = False
    def __init__(self, freqdist, gamma, bins=None):
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

def sum_logs(logs):
    return (reduce(add_logs, logs[1:], logs[0]) if len(logs) != 0 else _NINF)

class NaiveBayesClassifier(ClassifierI):
    def __init__(self, label_probdist, feature_probdist):
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self._labels = list(label_probdist.samples())

    def labels(self):
        return self._labels

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def prob_classify(self, featureset):
        # Discard any feature names that we've never seen before.
        # Otherwise, we'll just assign a probability of 0 to
        # everything.
        featureset = featureset.copy()
        for fname in list(featureset.keys()):
            for label in self._labels:
                if (label, fname) in self._feature_probdist:
                    break
            else:
                #print 'Ignoring unseen feature %s' % fname
                del featureset[fname]

        # Find the log probabilty of each label, given the features.
        # Start with the log probability of the label itself.
        logprob = {}
        for label in self._labels:
            logprob[label] = self._label_probdist.logprob(label)

        # Then add in the log probability of features given labels.
        for label in self._labels:
            for (fname, fval) in featureset.items():
                if (label, fname) in self._feature_probdist:
                    feature_probs = self._feature_probdist[label,fname]
                    logprob[label] += feature_probs.logprob(fval)
                else:
                    logprob[label] += sum_logs([])

        return DictionaryProbDist(logprob, normalize=True, log=True)

    def show_most_informative_features(self, n=10):
        # Determine the most relevant features, and display them.
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

def evaluate_classifier(featx):
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')

    negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    negcutoff = int(math.floor(len(negfeats)*3/4))
    poscutoff = int(math.floor(len(posfeats)*3/4))

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)

    print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    print ('pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos']))
    print ('pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos']))
    print ('neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg']))
    print ('neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg']))
    classifier.show_most_informative_features()

def word_feats(words):
    return dict([(word, True) for word in words])

print ('evaluating single word features')
evaluate_classifier(word_feats)

word_fd = FreqDist()
cond_word_fd = ConditionalFreqDist()

for word in movie_reviews.words(categories=['pos']):
    word_fd[word.lower()] += 1
    cond_word_fd['pos'][word.lower()] += 1

for word in movie_reviews.words(categories=['neg']):
    word_fd[word.lower()] += 1
    cond_word_fd['neg'][word.lower()] += 1

pos_word_count = cond_word_fd['pos'].N()
neg_word_count = cond_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count

word_scores = {}

for word, freq in word_fd.items():
    pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word],
        (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word],
        (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score

best = sorted(word_scores.items(), key=lambda s: s, reverse=True)[:10000]
bestwords = set([s for s in best])

def best_word_feats(words):
    return dict([(word, True) for word in words if word in bestwords])

print ('evaluating best word features')
evaluate_classifier(best_word_feats)

def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d

print ('evaluating best words + bigram chi_sq word features')
evaluate_classifier(best_bigram_word_feats)
