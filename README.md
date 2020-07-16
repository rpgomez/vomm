Introduction
===========

This project implements in python 2 algorithms for variable order
markov models called Predict by Partial Match (PPM) and Probabilistic
Suffix Tree (PST). This code is based on the paper *"On Prediction Using Variable
Order Markov Models"* by Ron Begleiter, Ran El-Yaniv, and Golan Yona in
the Journal of Artificial Intelligence Research 22 (2004) 385-421.

Details
=======
The 2 algorithms (PPM, PST) are implemented in a python module called vomm.py
(Variable Order Markov Models) with 2 variant classes for PPM and and 2 variant classes for PST.

Using either class to learn a model consists of several steps:

1. Acquire training data. The data needs to be represented as an array of integers $(x_t)$ with the constraint  $0 <=  x_t < \text{alphabet size}$. An example:
```{python}
# Take a string and convert each character to its ordinal value.
training_data = [ord(x) for x in "abracadabra" ]
```
2. Instantiate an object of the appropriate class. Example:
```{python}
import vomm
my_model = vomm.ppm()
```
3. Learn a model from the data.
```{python}
import vomm
my_model  = vomm.ppm()
my_model.fit(training_data, d=2)
```

Once you have learned a model from training you can do 2 things directly:

1. Compute the log of the probability of an observed sequence based on
model's parameters. The observed sequence has to be a sequence of integers $(x_t)$ just like the training data with the same constraints on $x_t$. For example:
```{python}
my_model.logpdf(observed_sequence)
```
2. Generate new data from the model of a specified length. For example
```{python}
my_model.generate_data(length=300)
```

To print out the numerical parameters of the learned model, just use the print statement/function like so:
```{python}
print(my_model)
```

Variants
========

There are 2 variants of the PPM algorithm:

1. ppm -- this is an appropriate class to use when the size of the alphabet is relatively small, for example alphabet size < 256, i.e. an alphabet of bytes.

2. PPM_words -- this is an appropriate class to use when the size of the alphabet is large, for example a vocabulary of words  ~ 2**15. This class implements a memory efficient implementation of the look up table for Pr(symbol|context).

There are 2 variants of the PST algorithm:

1. pst -- a class to use in situations similar to ppm.
2. pst_JS -- this class is for the same regime as pst, but it implements a different pruning solution for contexts: it makes use of Jensen-Shannon Entropy to decide if the probability distribution of a longer context is sufficiently different than the probability distribution of its shorter context parent. One advantage of this class is that it autodetermine an appropriate threshold to use if the user doesn't know what to to use. 

Internals
=========
Learning a model consists of

1. determining a set of contexts (a sequence of symbols of length no
   larger than $d$) based on the length constraint $d$ (and other
   possible constraints.)
2. Estimating the probability distribution $Pr(\sigma|s)$ for each
   context $s$ and symbol $\sigma$ in the alphabet.

After creating an object $x$ of whichever class you chose and *fitted*
to the training data, the object $x$ will have several attributes of
which 3 are important:

* pdf_dict -- this is a dictionary with key context $s$ and value the probability distribution $Pr(|s)$.
* logpdf_dict -- it's similar to pdf_dict, but the value is the log of the probability distribution.
* context_child -- this is a dictionary with key context $s$ and value
  the set of possible longer children contexts $\{ xs \}$ which are in
  the set of contexts recovered in step 1. This dictionary speeds up
  finding the largest context $s$ which matches the tail end of a
  sequence of symbols.

Performance
===========

* On a training set of ascii text (man bash | strings) of 338383 symbols  converted to their ordinal values with an alphabet size of 127 it takes
  * approximately 56.5 seconds to learn a ppm model with d=4 generating 33764 contexts;
  * approximately 20.3 seconds to learn a pst model with d=10 and default values for the other parameters generating 3834 contexts.
* Using the same data and computing the log probability it takes
  * approximately 3 seconds with ppm model;
  * approximately 2 seconds with pst model.

Installation
===========

This is a python module which is installed by using a distutils based
install script setup.py.  Installation consists of either:

```
python setup.py install
```

or

```
pip install .
```
