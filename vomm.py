"""This is a module for implementing variable order markov model
algorithms as described in the paper "On Prediction Using Variable
Order Markov Models" by Ron Begleiter, Ran El-Yaniv, and Golan Yona in
the Journal of Artificial Intelligence Research 22 (2004) 385-421.  """

import numpy as np
import scipy.stats as stats
from collections import Counter, defaultdict

def find_elbow(S):
    """Finds the elbow in the graph S. Make sure S has been
    presorted before calling this function and that it is
    a numpy array"""

    Sp = S - S[0]
    N = Sp.shape[0]-1
    SN = Sp[-1]
    M = SN/N
    R = (SN**2 + N**2)**0.5
    sintheta = SN/R
    costheta = N/R
    x = np.arange(Sp.shape[0])
    y = Sp
    # xp = x*costheta + y*sintheta # Don't need it.
    yp = y*costheta - x*sintheta
    yp = np.abs(yp)
    elbow = yp.argmax()
    return elbow

def JSD(P, Q):
    """Computes the Jensen-Shannon Divergence between distributions P and Q. """
    M = 0.5 * (P + Q)
    return 0.5 * (stats.entropy(P, M) + stats.entropy(Q, M))

def find_contexts(training_data, d= 4):
    """
    Takes a sequence of observed symbols and finds all contexts of
    length at most d.

    training_data represents the sequence of observed symbols as a
    tuple of integers x where 0 <= x < alphabet size.

    Contexts of length k are represented as tuples of integers of length k.

    This function returns the observed contexts as a set.
    """

    contexts = set()

    N = len(training_data)

    for k in range(1,d+1):
        contexts = contexts.union([training_data[t:t+k] for t in range(N-k+1)])

    return contexts

def count_occurrences(training_data, d=4, alphabet_size = None):
    """
    Counts the number of occurrences of s\sigma where s is a context
    and \sigma is the symbol that immediately follows s in the
    training data.

    training_data represents the sequence of observed symbols as a
    tuple of integers x where 0 <= x < alphabet size.

    d determines the longest context to consider.

    alphabet_size determines the number of possible observable
    distinct symbols. If not set, the function determines it from the
    training data.

    The function returns the counts as a dictionary with key context s
    and value the counts array.

    """

    contexts = find_contexts(training_data, d = d)

    if alphabet_size == None:
        alphabet_size = max(training_data) + 1

    counts = dict([(x, np.zeros(alphabet_size,dtype=np.int)) for x in contexts])

    # Include the null context as well.
    counts[()] = np.bincount(training_data,minlength = alphabet_size)

    N = len(training_data)
    for k in range(1,d+1):
        for t in range(N-k):
            s = training_data[t:t+k]
            sigma = training_data[t+k]
            counts[s][sigma]  += 1

    return counts

def compute_ppm_probability(counts):
    """
    Takes the counts dictionary and generates the Pr(sigma| s)
    probabilities recursively. It returns a dictionary whose key is
    context and value is the probability distribution on the successive symbol.
    """

    # figure out d and alphabet_size from the counts dictionary.
    d = max( [len(x) for x in counts.keys() ])
    alphabet_size = counts[()].shape[0]

    pdf = dict([(x, np.zeros(alphabet_size)) for x in counts.keys()])

    # partition the contexts by size.
    byk = [ [] for k in range(d+1) ]
    for x in counts.keys():
        byk[len(x)].append(x)

    # Now recursively define pdfs starting with the shortest context
    # to the largest.

    pdf[()] = (counts[()] +1.0)/(counts[()].sum() + alphabet_size)

    for k in range(1,d+1):
        for x in byk[k]:
            sigma_observed = np.argwhere(counts[x] > 0).reshape(-1)
            alphabet_obs_size = len(sigma_observed)
            sigma_escaped = np.argwhere(counts[x] == 0).reshape(-1)
            denominator = alphabet_obs_size + counts[x].sum()
            x_1 = x[1:] # sub context if needed.

            if alphabet_obs_size > 0:
                escape_factor = alphabet_obs_size*1.0/denominator
            else:
                escape_factor = 1.0

            pdf[x][sigma_observed] = counts[x][sigma_observed]*1.0/denominator

            if len(sigma_escaped) > 0:
                pdf[x][sigma_escaped] = escape_factor*pdf[x_1][sigma_escaped]/pdf[x_1][sigma_escaped].sum()

            # Normalize (needed in the case that all symbols are observed)
            pdf[x] = pdf[x]/pdf[x].sum()

    return pdf

def find_largest_context(chunk,fast_lookup_table,d):
    """Find the largest context that matches the observed chunk of symbols
    and returns it.

    chunk is a sequence of symbols represented as a tuple of integers 0 <= x < alphabet size.

    fast_lookup_table is a dictionary with key context s and value the
    list of contexts which are of the form xs.

    d is the size of the largest context in the set of contexts.
    """

    if len(chunk) == 0:
        return ()

    current_context = ()
    end = len(chunk)
    start = end

    while chunk[start:end] == current_context:
        start -= 1
        if start < 0 or start < end - d:
            break

        if chunk[start:end] in fast_lookup_table[current_context]:
            current_context = chunk[start:end]
        else:
            break

    return current_context

class ppm:
    """ This class implements the "predict by partial match" algorithm. """

    def __init__(self):
        """ Not much to do here. """

    def generate_fast_lookup(self):
        """Takes the pdf_dict dictionary and computes a faster lookup
        dictionary mapping suffix s to its longer contexts xs.  I need
        this to speed up computing the probability of logpdf for an
        observed sequence
        """

        # I want to create a fast look up of context s -> xs
        # So scoring a sequence is faster.

        context_by_length  = dict([(k,[]) for k in range(self.d+1) ])

        for x in self.pdf_dict.keys():
            context_by_length[len(x)].append(x)

        # Now lets generate a dictionary look up context s -> possible context xs.
        self.context_child = {}

        for k in range(self.d):
            for x in context_by_length[k]:
                self.context_child[x] = [ y for y in context_by_length[k+1] if y[1:] == x ]

        for x in context_by_length[self.d]:
            self.context_child[x] = []

    def fit(self,training_data, d=4, alphabet_size = None):
        """
        This is the method to call to fit the model to the data.
        training_data should be a sequence of symbols represented by
        integers 0 <= x < alphabet size.

        d specifies the largest context sequence we're willing to consider.

        alphabet_size specifies the number of distinct symbols that
        can be possibly observed. If not specified, the alphabet_size
        will be inferred from the training data.
        """

        if alphabet_size == None:
            alphabet_size = max(training_data) + 1

        self.alphabet_size = alphabet_size
        self.d = d

        counts = count_occurrences(tuple(training_data),d=self.d,
                                   alphabet_size = self.alphabet_size)

        self.pdf_dict = compute_ppm_probability(counts)

        self.logpdf_dict = dict([(x,np.log(self.pdf_dict[x])) for x in self.pdf_dict.keys()])

        # For faster look up  when computing logpdf(observed data).
        self.generate_fast_lookup()

        return

    def logpdf(self,observed_data):
        """Call this method after using fitting the model to compute the log of
        the probability of an observed sequence of data.

        observed_data should be a sequence of symbols represented by
        integers 0 <= x < alphabet_size. """

        temp = tuple(observed_data)
        # start with the null context and work my way forward.

        logprob = 0.0
        for t in range(len(temp)):
            chunk = temp[max(t-self.d,0):t]
            sigma = temp[t]
            context = find_largest_context(chunk,self.context_child,self.d)
            logprob += self.logpdf_dict[context][sigma]

        return logprob

    def generate_data(self,prefix=None, length=200):
        """Generates data from the fitted model.

        The length parameter determines how many symbols to generate.

        prefix is an optional sequence of symbols to be appended to,
        in other words, the prefix sequence is treated as a set of
        symbols that were previously "generated" that are going to be
        appended by an additional "length" number of symbols.

        The default value of None indicates that no such prefix
        exists. We're going to be generating symbols starting from the
        null context.

        It returns the generated data as an array of symbols
        represented as integers 0 <=x < alphabet_size.

        """

        if prefix != None:
            new_data = np.zeros(len(prefix) + length,dtype=np.int)
            new_data[:len(prefix)] = prefix
            start = len(prefix)
        else:
            new_data = np.zeros(length,dtype=np.int)
            start = 0

        for t in range(start,len(new_data)):
            chunk = tuple(new_data[max(t-self.d,0):t])
            context = find_largest_context(chunk,self.context_child,self.d)
            new_symbol = np.random.choice(self.alphabet_size,p=self.pdf_dict[context])
            new_data[t] = new_symbol

        return new_data[start:]

    def __str__(self):
        """ Implements a string representation to return the parameters of this model. """

        return "\n".join(["alphabet size: %d" % self.alphabet_size,
                          "context length d: %d" % self.d,
                          "Size of model: %d" % len(self.pdf_dict)])

def kullback_leibler_test(pdfs,s,threshold):
    """Takes a dictionary pdfs where key is context s and value is the
    probability distribution Pr( | s), a context, and a
    Kullback-Leibler threshold value and returns True (passed) or
    False (failed).
    """

    if s == ():
        # Null context always passes.
        return True

    p = pdfs[s]
    q = pdfs[s[1:]]

    logpq = np.log(p/q)
    logpq[q == 0.0] = 1000
    logpq[p == 0.0] = 0.0

    kl_value = (p*logpq).sum()
    return kl_value >= threshold

def jensen_shannon_test(pdfs,s,threshold):
    """Takes a dictionary pdfs where key is context s and value is the
    probability distribution Pr( | s), a context, and a
    Jensen-Shannon divergence threshold value and returns True (passed) or
    False (failed).
    """

    if s == ():
        # Null context always passes.
        return True

    p = pdfs[s]
    q = pdfs[s[1:]]
    js_value = JSD(p,q)
    return js_value >= threshold


class pst(ppm):
    """This is the class to implement the probabilistic suffix tree
    algorithm.  What distinguishes PST from PPM is that we prune
    contexts that don't carry sufficient information.

    There are 3 criteria for pruning:

    1. frequency threshold -- if a context does not occur a sufficient number of times in
    the training data then it's pruned.

    2. meaning_threshold -- if a context's sample probability
    distribution does not contain a probability that's sufficiently
    large enough, then the context is pruned.

    3. Kullback-Leibler threshold -- if a context s probability
    distribution is not sufficiently different enough from the
    probability distribution of parent suffix s' then the context is
    pruned.

    """

    def fit(self, training_data, d=4, alphabet_size = None,
            freq_threshold = None, meaning_threshold = None,
            kl_threshold = 0.01):
        """This is the method to call to fit the model to the data.
        training_data should be a sequence of symbols represented by
        integers 0 <= x < alphabet size.

        d specifies the largest context sequence we're willing to consider.

        alphabet_size specifies the number of distinct symbols that
        can be possibly observed. If not specified, the alphabet_size
        will be inferred from the training data.

        freq_threshold determines the minimum number of times a
        context has to occur for it to be kept and not pruned. If not
        set, then we use the value freq_threshold = .1*alphabet_size

        meaning_threshold sets the minimum value for max{ Pr(sigma|s)|
        sigma)} we're willing to accept for a context s. If not set we
        use the value 2/alphabet size.

        kl_threshold sets the minimum distance between the probability
        distributions for child s and suffix parent s' contexts we
        require for a context s to be kept. If not set, we use the value 0.1

        """

        if alphabet_size == None:
            alphabet_size = max(training_data) + 1

        self.alphabet_size = alphabet_size
        self.d = d
        self.kl_threshold = kl_threshold

        if freq_threshold == None:
            freq_threshold = 0.1*alphabet_size

        self.freq_threshold = freq_threshold

        if meaning_threshold == None:
            meaning_threshold = 2.0/alphabet_size

        self.meaning_threshold = meaning_threshold

        counts = count_occurrences(tuple(training_data),d=self.d,
                                   alphabet_size = self.alphabet_size)

        # Kill all contexts that fail freq_threshold.
        counts = dict([ (s,counts[s]) for s in counts if counts[s].sum() >= self.freq_threshold])

        self.pdf_dict = compute_ppm_probability(counts)

        # K-L is a harsher threshold than meaning threshold. We'll do it first.

        # Kill all context that fail Kullback-Leibler divergence
        passed = dict([ (s,kullback_leibler_test(self.pdf_dict,s,self.kl_threshold))
                        for s in self.pdf_dict ])
        # propagate failures from parent to children.
        for k in range(self.d+1):
            for s in passed:
                if s == ():
                    continue
                if not passed[s[1:]]:
                    passed[s] = False
        self.pdf_dict = dict([(s,self.pdf_dict[s]) for s in self.pdf_dict if passed[s] ])

        # Kill all context that fail meaning threshold.
        passed = dict([ (s,self.pdf_dict[s].max() >= self.meaning_threshold) for s in self.pdf_dict])

        # propagate failures from parent to children.
        for k in range(self.d+1):
            for s in passed:
                if s == ():
                    continue
                if not passed[s[1:]]:
                    passed[s] = False
        self.pdf_dict = dict([(s,self.pdf_dict[s]) for s in self.pdf_dict if passed[s] ])


        self.logpdf_dict = dict([(x,np.log(self.pdf_dict[x])) for x in self.pdf_dict.keys()])

        # For faster look up  when computing logpdf(observed data).
        self.generate_fast_lookup()

        return

    def __str__(self):
        """ Implements a string representation to return the parameters of this model. """

        return "\n".join(["alphabet size: %d" % self.alphabet_size,
                          "context length d: %d" % self.d,
                          "Size of model: %d" % len(self.pdf_dict),
                          "Frequency threshold: %f" % self.freq_threshold,
                          "Meaning threshold: %f" % self.meaning_threshold,
                          "Kullback-Leibler threshold: %f" % self.kl_threshold])


class pst_JS(ppm):
    """This is the class to implement the probabilistic suffix tree
    algorithm.  What distinguishes PST from PPM is that we prune
    contexts that don't carry sufficient information.

    There is 1 criteria for pruning:

    Jensen-Shannon threshold -- if a context s probability
    distribution is not sufficiently different enough from the
    probability distribution of parent suffix s' then the context is
    pruned.

    """

    def fit(self, training_data, d=4, alphabet_size = None,
            js_threshold = None):
        """This is the method to call to fit the model to the data.
        training_data should be a sequence of symbols represented by
        integers 0 <= x < alphabet size.

        d specifies the largest context sequence we're willing to consider.

        alphabet_size specifies the number of distinct symbols that
        can be possibly observed. If not specified, the alphabet_size
        will be inferred from the training data.

        js_threshold sets the minimum distance between the probability
        distributions for child s and suffix parent s' contexts we
        require for a context s to be kept. If not set, we autodetermine it.

        """

        if alphabet_size == None:
            alphabet_size = max(training_data) + 1

        self.alphabet_size = alphabet_size
        self.d = d

        counts = count_occurrences(tuple(training_data),d=self.d,
                                   alphabet_size = self.alphabet_size)

        self.pdf_dict = compute_ppm_probability(counts)

        if js_threshold == None:
            # we need to determine it
            contexts = list(counts.keys())
            scores = {}
            for child in contexts:
                if len(child) > 1:
                    parent = child[1:]
                    scores[(child,parent)] = JSD(self.pdf_dict[parent],self.pdf_dict[child])

            sorted_scores = np.sort(np.array(list(scores.values())))
            elbow = find_elbow(sorted_scores)
            js_threshold = sorted_scores[elbow]

        self.js_threshold = js_threshold

        # JS threshold
        passed = dict([ (s,jensen_shannon_test(self.pdf_dict,s,self.js_threshold))
                        for s in self.pdf_dict ])
        # propagate failures from parent to children.
        for k in range(self.d+1):
            for s in passed:
                if s == ():
                    continue
                if not passed[s[1:]]:
                    passed[s] = False
        self.pdf_dict = dict([(s,self.pdf_dict[s]) for s in self.pdf_dict if passed[s] ])

        self.logpdf_dict = dict([(x,np.log(self.pdf_dict[x])) for x in self.pdf_dict.keys()])

        # For faster look up  when computing logpdf(observed data).
        self.generate_fast_lookup()

        return

    def __str__(self):
        """ Implements a string representation to return the parameters of this model. """

        return "\n".join(["alphabet size: %d" % self.alphabet_size,
                          "context length d: %d" % self.d,
                          "Size of model: %d" % len(self.pdf_dict),
                          "Jensen Shannon threshold: %f" % self.js_threshold])


def construct_counts_dictionary(observed_sequence,d=4):
    """Constructs the dictionary of observed counts of words following 
    contexts at most d words long. 
    
    observed_sequence should be a sequence of type integer of non-negative integers.
    Each integer corresponds to some word in the vocabulary.
    """
    
    counts = defaultdict(Counter)
    N = len(observed_sequence)
    
    for t in range(N):
        obs = observed_sequence[t]
        for l in range(0,d+1):
            if t - l < 0:
                break
            context = observed_sequence[t-l:t]
            counts[context][obs] +=1
    
    return counts
            
def construct_prob_dictionary(counts_dictionary,vocabulary_size):
    """Computes the memory efficient dictionary word| context probabilities"""

    V = vocabulary_size
    
    probs_dictionary = dict()
    for context in counts_dictionary:
        local_list = counts_dictionary[context]
        total_count = sum(list(local_list.values()))
        
        R = len(local_list)
        if R == V:
            omega = 1.0
        else:
            omega = (1+ total_count)/(2 + total_count)
            
        probs_dictionary[context] = defaultdict(float)
        
        local_probs = defaultdict(float)
        for word in local_list:
            c_i = local_list[word]
            pr = c_i/total_count * omega
            local_probs[word] = pr
        
        if R < V:
            local_probs[None] = (1.0 - omega)/(V-R)
        else:
            local_probs[None] = 0.0
        
        probs_dictionary[context] = local_probs
    
    return probs_dictionary

def generate_log_pdf_dict(probs_dictionary):
    """Computes the log likelihood dictionary
    log (Pr(symbol|context)) from the probs_dictionary"""
    
    log_pdf_dict = {}
    for context in probs_dictionary:
        local_pdf_list = probs_dictionary[context]
        local_log_pdf_list = {}
        for asymbol in local_pdf_list:
            local_log_pdf_list[asymbol] = np.log(local_pdf_list[asymbol])
            
        log_pdf_dict[context] = local_log_pdf_list
    
    return log_pdf_dict

class PPM_words:
    """This class is designed to be memory efficient in the case that the
    size of the vocabulary is large (>= 2**15 for example).

    """
    
    def __init__(self):
        """ Not much to do here. """


    def fit(self,training_data, d=4, alphabet_size = None,verbose=False):
        """
        This is the method to call to fit the model to the data.
        training_data should be a sequence of symbols represented by
        integers 0 <= x < alphabet size.

        d specifies the largest context sequence we're willing to consider.

        alphabet_size specifies the number of distinct symbols that
        can be possibly observed. If not specified, the alphabet_size
        will be inferred from the training data.
        """

        if alphabet_size == None:
            alphabet_size = max(training_data) + 1

        self.alphabet_size = alphabet_size
        self.d = d

        training_data = tuple(training_data) # ensure the data are type tuple.
        counts = construct_counts_dictionary(training_data,d=d)
        self.pdf_dict = construct_prob_dictionary(counts,alphabet_size)

        self.logpdf_dict = generate_log_pdf_dict(self.pdf_dict)

        # For faster look up  when computing logpdf(observed data).
        # self.generate_fast_lookup(verbose=verbose)

        return

    def logpdf(self,observed_data,verbose=False):
        """Call this method after using fitting the model to compute the log of
        the probability of an observed sequence of data.

        observed_data should be a sequence of symbols represented by
        integers 0 <= x < alphabet_size. """

        if not verbose:
            def use_me(x,desc=None):
                return x
        else:
            use_me = tqdm

        temp = tuple(observed_data)
        # start with the null context and work my way forward.

        logprob = 0.0
        for t in use_me(range(len(temp)),desc='computing log pdf'):
            chunk = temp[max(t-self.d,0):t]
            sigma = temp[t]
            while len(chunk) > 0:
                if chunk in self.logpdf_dict:
                    break
                else:
                    chunk = chunk[1:]
            context = chunk
            if sigma in self.logpdf_dict[context]:
                logprob += self.logpdf_dict[context][sigma]
            else:
                logprob += self.logpdf_dict[context][None] # Didn't see this symbol with the context
        return logprob

    def generate_data(self,prefix=None, length=200,verbose=False):
        """Generates data from the fitted model.

        The length parameter determines how many symbols to generate.

        prefix is an optional sequence of symbols to be appended to,
        in other words, the prefix sequence is treated as a set of
        symbols that were previously "generated" that are going to be
        appended by an additional "length" number of symbols.

        The default value of None indicates that no such prefix
        exists. We're going to be generating symbols starting from the
        null context.

        It returns the generated data as an array of symbols
        represented as integers 0 <=x < alphabet_size.

        """

        if not verbose:
            def use_me(x):
                return x
        else:
            use_me = tqdm

        if prefix != None:
            new_data = np.zeros(len(prefix) + length,dtype=np.int)
            new_data[:len(prefix)] = prefix
            start = len(prefix)
        else:
            new_data = np.zeros(length,dtype=np.int)
            start = 0

        scratch_pdf = np.zeros(self.alphabet_size)
        for t in use_me(range(start,len(new_data))):
            chunk = tuple(new_data[max(t-self.d,0):t])
            while len(chunk) > 0:
                if chunk in self.logpdf_dict:
                    break
                else:
                    chunk = chunk[1:]
            context = chunk
            
            scratch_pdf[:] = self.pdf_dict[context][None] # the symbols we didn't see
            for symbol in self.pdf_dict[context]:
                if symbol  == None:
                    continue
                scratch_pdf[symbol] = self.pdf_dict[context][symbol]
                
            new_symbol = np.random.choice(self.alphabet_size,p=scratch_pdf)
            new_data[t] = new_symbol

        return new_data[start:]

    def __str__(self):
        """ Implements a string representation to return the parameters of this model. """

        return "\n".join(["alphabet size: %d" % self.alphabet_size,
                          "context length d: %d" % self.d,
                          "Size of model: %d" % len(self.pdf_dict)])
