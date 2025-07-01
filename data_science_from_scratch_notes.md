# Chapter 2: A Crash Course in Python

pythonic code - there should be one and preferably only one obvious way to do something  

## Whitespace Formatting

Python uses indentation to delimit blocks of code, whitespace is ignored in parentheses and brackets  
can also use backslash to indicate that statement continues to the next line  

## Modules
certain features of Python (part of the language and third-party ones)  are not loaded by default  

```python
import numpy  #importing a module
import numpy as np #importing and aliasing a module
from collections import Counter  #import a few specific values from module
```

## Arithmetic

python uses integer division  
```python
from __future__ import division #changes to floating-point division
```
can use integer division by // instead  

## Functions
rule for taking zero or more inputs and returning a corresponding output

```python
def double(x):
    """ docstring that describes the function
    multiplies the input by 2"""
    return x * 2
```
## Strings

strings can be delimited using single or double quotations marks  
special characters encoded using back slash \    
raw string r"\t" (represents the characters \t)  
multiline strings using triple double quotes  

## Exceptions

try: ... except: ... 
not bad to throw exceptions in python  

## Lists
ordered collection of objects, similar to array in other languages  

```python
integer_list = [1,2,3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [integer_list, hetereogeneous_list, []]

list_length = len(integer_list) # equals 3
list_sum = sum(integer_list) # equals 6

x = range(10) # is the list [0,1,...,9]
nth_element = x[n] # nth element of zero-indexed array  
nine = x[-1] # last element of x

# using brackets to slice lists
first_three = x[:3] 
three_to_end = x[3:]
one_to_four = x[1:5]
lst three = x[-3:]
without_first_and_last = x[1:-1]
copy_of_x = x[:]

# in operator to check for list membership
1 in [1,2,3]

# concantenate lists together
x = [1,2,3]
x.extend([4,5,6]) # x is now [1,2,3,4,5,6]
x.append(0) # 0 is added to the end of x

# list addition
y = x + [4,5,6] # y is [1,2,3,4,5,6], x is unchanged

#unpacking lists

x, y = [1,2] # number of elements must match on both sides
_, y = [1,2] # don't care about first value
```

## Tuples
lists but immutable (cannot be modified)

```python
my_list = [1,2]
my_tuple = (1,2)
other_tuple = 3,4

#tuples and lists can be used for multipl assignment
x,y = 1,2
x,y = y,x
```
## Dictionaries
associates values with keys and allows quick retrieval of a value corresponding to a key  

```python
empty_dict = {} # Pythonic
empty_dict2 = dict() # less Pythonic
grades = { "Joel" : 80, "Tim" : 95} # dictionary literal
#lookup the value for a key using square brackets
joels_grade = grades["Joel"] # equals 80

#KeyError if asking for a key not in the dict
joel_has_grade = "Joel" in grades #check for existence of key

#.get() method returns a default value instead of raising an exception for keys not in dict
kates_grade = grades.get("Kate", 0) # 0 is the default value

#assign key-value pairs using square brackets
grades["Kate"] = 100
num_students = len(grades)

grades.keys() # list of keys
grades.values() # list of values
grades.items() # list of (key, value) tuples
```
dictionary keys are immutable  
cannot use lists as keys  
if you need a multipart key, you should use a tuple or figure out a way to turn the key to a string  

**defaultdict**
acts like a regular dictionary, but when you try to look up a key that doesn't exist, it first adds a value for the key using the the type that you choose

```python
from collections import defaultdict

word_counts = defaultdict(int) # int() produces 0
for word in document:
    word_counts[word] += 1
```

**counter**
A counter turns a sequence of values into a defauldict(int)-like object  
maps keys to counts  

```python
from collections import Counter
c = Counter([0,1,2,0]) # c = {0 : 2, 1 : 1, 2 : 2}

word_counts = Counter(document)

#prints the 10 most common words and their counts
for word, count in word_counts.most_common(10):
    print word, count
```

## Sets
a collection of distinct elements
```python
s = set()
s.add(1) # s is now { 1 }
s.add(2) # s is now {1 , 2}
s.add(2) # s is now {1, 2}
x = len(s) # equals 2
y = 2 in s # True
z = 3 in s # False
```
'in' function to check membership is very fast on sets (better than lists)  
easy to find distinct items in a collection  

## Control Flow
```python
# if's
if 1 > 2:
    ...
elif 1 > 3:
    ...
else:
    ...

# one line if-then-else
parity = "even" if x % 2 == 0 else "odd"

# while loop
x = 0
while x < 10:
    print x, "is less than 10"
    x += 1

#for loop
for x in range(10):
    print x, "is less than 10"

# continue and break
for x in range(10):
    if x == 3:
        continue # go immediately to the next iteration
    if x == 5:
        break # quit the loop entirely
    print x # prints 0, 1, 2, 4
```

## Truthiness
booleans are capitalized in Python  
```python
# uses None to indicate nonexistent value
x = None
print x is None
# the following are all "Falsy", and everything else is True
False
None
[]
{}
""
set()
0
0.0
```
`first_char = s and s[0]` returns the second value when the first is "truthy", returns the first value if it's not  
`safe_x = x or 0` is definitely a number  
`all()` returns True when every element is Truthy  
`any()` returns True when at least one element is Truthy

## The Not-So-Basics

### Sorting

`sorted(list())` returns a new sorted list
`list().sort()` sorts the list

`sorted(list(), key, reverse)` takes in a list, sorts according to the key, and in reverse if selected  

### List Comprehensions
when you want to transform a list into another list, by choosing only certain elements, by transforming elements, or both

```python
even_numbers = [x for x in range(5) if x % 2 == 0] # [0, 2, 4]
squares = [x * x for x in range(5)] # [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers] # [0, 4, 16]

# turn lists into dictionaries or sets
square_dict = {x : x * x for x in range(5)} # { 0:0, 1:1, 2:4, 3:9, 4:16 }
square_set = { x * x for x in [1, -1] }

# when you don't need the value from list
zeroes = [0 for _ in even_numbers]

# can use multiple for loops
increasing_pairs = [(x, y)
                    for x in range(10)
                    for y in range(x+1, 10)]
```

### Generators and Iterators
a problem with lists is that they can grow very big  
-> not efficient with memory and calculating efficiency  
generator is something you iterate over but whose values are produced lazily (as needed)  

```python
def lazy_range(n):
    """ a lazy version of range"""
    i = 0
    while i < n:
        yield i
        i += 1

# the following loop will consume the yielded values until none are left
for i in lazy_range(10):
    do_something_with(i)

# python comes with lazy_range function xrange
# range in Python 3 is lazy already

# you can only iterate through lazy generators once

lazy_evens_below_20 = (i for i in lazy_range(20) if i % 2 == 0)

# iteritems() lazily iterates through key-value pairs for dicts
```

### Randomness
generating random numbers
```python
import random

# produces four numbers in a list that are uniformly distributed between 0 and 1
four_uniform_randoms = [random.random() for _ in range(4)]

# python uses pseudorandomness, where you could set a seed and get reproducible results"

random.randrange(10) # choose randomly from range(10)

up_to_ten = range(10)
random.shuffle(up_to_ten) # shuffles the order

my_best_friend = random.choice(["Alice", "Bob", "Charlie"]) # chooses a random element

#random sample without replacement
lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6) # [16, 36, 10, 6, 25, 9]

#random sample with replacement
four_with_replacement = [random.choice(range(10))
                        for _ in range(4)]
```

### Regular Expressions
provide a way of searching text  
```python
import re

print all([                     # all of these are true
    not re.match("a", "cat"), # cat doesn't start with a
    re.search("a", "cat"),    # cat has an a in it
    not re.search('c', 'dog'), # cat doesn't have c
    3 == len(re.split('[ab]', 'carbs')), # split on a or b to [c, r, s]
    'R-D-' == re.sub('[0-9]', '-', 'R2D2') # replace digits with dashes
    ])
```

### Object-Oriented Programming
Python allows you to define classes that encapsulate data and functions  

```python
# implementing Set in python

# name classes by PascalCase
class Set:

    # member functions
    # every one takes a first parameter "self"
    # that referes to the particular Set object being used
    def __init__(self, values=None):
        """ This is the constructor.
        It gets called when you create a new Set
        s1 = Set()
        s2 = Set([1,2,3,4])
        """
        self.dict = {} # each instance of Set has its own dict property

        if values is not None:
            for value in values:
                self.add(value)

    def __repr__(self):
        """ this is the string representation of a Set object
        if you type it at the Python prompt or pass it to str()"""
        return "Set: " + str(self.dict.key())

    # represent membership by being a key in self.dict with value True
    def add(self, value):
        self.dict[value] = True
    
    # value is in the Set if it's a key in the dictionary
    def contains(self, value):
        return value in self.dict
    
    def remove(self, value):
        del self.dict[value]
```

### Functional Tools
sometimes we'll want to partially apply (or curry) functions to create new functions

```python
def exp(base, power):
    return base ** power
```

```python
# if we want to make a two_to_the function
from functools import partial
two_to_the = partial(exp, power=2)
print two_to_the(3)

# starts getting messy if we curry arguments in the middle of the function
```

`map`, `reduce`, and `filter` provide function alternatives to list comprehensions

```python
def double(x):
    return 2 * x

xs = [1,2,3,4]
twice_xs = [double(x) for x in xs]  # [2,4,6,8]
twice_xs = map(double, xs)  # same as above
list_doubler = partial(map, double) # *function* that doubles as list
twice_xs = list_doubler(xs)
```

```python
def multiply(x, y): return x * y

products = map(multiply, [1,2], [4,5]) # [4, 10]
```

```python
def is_even(x):
    """True if x is even, False, is x is odd"""
    return x % 2 == 0

x_evens = [x for x in xs if is_even(x)] # [2,4]
x_evens = filter(is_even, xs) # same as above
list_evener = partial(filter, is_even) # *function* that filters a list
x_evens = list_evener(xs) # again [2, 4]
```

```python
x_product = reduce(multiply, xs) # 1*2*3*4 = 24
list_product = partial(reduce, multiply) # *function* that reduces a list
x_product = list_product(xs) # = 24
```

### enumerate
iterate over a list and use both the element and their indexes  

```python
# not Pythonic
for i in range(len(document)):
    document = documents[i]
    do_something(i, document)

# also not Pythonic
i = 0
for document in documents:
    do_something(i, document)
    i += 1

# Pythonic
for i, document in enumerate(documents):
    do_something(i, document)
```

### zipp and Argument Unpacking
need to zip two or more lists together  
zip transforms multiple lists into a single list of tuples  

```python
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
zip(list1, list2) # is [('a', 1), ('b', 2), ('c', 3)]
# if lists are of different lengths, zip stops as soon as the first list ends

# unzip a list
pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)
# asterisk performs argument unpacking
```

### args and kwargs
specifying a function that takes arbitray arguments  

```python
def doubler(f):
    def g(x):
        return 2 * f(x)
    return g
```
this breaks down when we have function with more than one argument  
`def f2(x,y): return x + y`

```python
def magic(*args, **kwargs):
    print "unnamed args:", args
    print "keyword args:", kwargs

magic(1, 2, key="word", key2="word2")

# prints
# unnamed args: (1,2)
# keyword args: {'key2': 'word2', 'key': 'word'}
```

`args` is a tuple of unnamed (positional) arguments  
`kwargs` is a dict of named (keyword)what arguments

# Chapter 3 Visualizing Data

`import matplotlib.pyplot as plt`

## Bar charts
## Line charts
## Scatterplots

# Chapter 4 Linear Algebra

## Vectors
## Matrices

# Chapter 5 Statistics

## Describing a single set of data

### Central tendencies

`mean` is sum of data divided by count  
`median` is middle most or average of middle most values  

there are some nonobvious ways to find the median without sorting the data, but we have to for now  

`mode` is most common value  
`quantile` is the percentage of the values that are below a threshold  

### Dispersion
how spread out the data is  

`range` is the difference between largest and smallest elements  
`variance` is the sum of the squares of deviations from the mean, divided by the number of elements minus one  
on average this definition of variation is an underestimate of the population variance
`standard deviation` is the square root of variance  
`interquartile range` is the difference between the 75th and 25th quartile values  

### Correlation

`covariance` is the dot product of the deviations of two variables, divided by the number of elements minus one  
a large positive covariance means that x tends to be large when y is large and small when y is small  
a large negative covariance means the other way around   
`correlation` is the covariance divided by the standard deviations of the variables   
-1 is perfect anti-correlation, 1 is perfect correlation, 0 is no correlation  

correlation is sensitive to outliers because standard deviation is sensitive to outliers  

## Simpson's Paradox
correlation measures the relationship between the two variables assuming everything else is equal  
can lead to misleading data once you account for more variables  
can assume "all else being equal" for random experiments  
bad assumption when there are deeper pattern to class assignments  

## Some Other Correlational Caveats
correlation looks for how x_i compares to mean(x) affects y_i compares to mean(y)  
there could be other types of relationships between variables  
correlation tells nothing about the scale of the relationship  

## Correlation and Causation
correlation is not causation  
randomized trials makes claims about causation stronger  

# Chapter 6 Probability
probability is the way of quantifying the uncertainty associated with events chosen from some universe of events 

## Dependence and Independence
two events E and F are dependent if knowing information about E gives information about whether F happens  
otherwise they are independent  

two events E and F is independent if $P(E,F) = P(E)P(F)$

## Conditional Probability
if two events are not necessarily independent, we have the probability of E given F  
$P(E|F) = \frac{P(E,F)}{P(F)}$  
when E and F are independent, this gives  
$P(E|F) = P(E)$

## Bayes's Theorem
$$P(E|F) = \frac{P(E,F)}{P(F)} = \frac{P(F|E)P(E)}{P(F)}$$

$$P(E|F) = \frac{P(F|E)P(E)}{P(F|E)P(E)+P(F|\neg E)P(\neg E)}$$

## Random Variables
random variable is a variable whose possible values have an associated probability distribution  
the expected value is the average of the values weighted by their probabilities  
can be conditioned on other events  

## Continuous Distribution
discrete distributions associates positive probabilities with discrete outcomes  
continuous distributions are modeled across a continuum of outcomes  
the probability of a specific event happening is zero because there are infinitely many events  
`probability density function (pdf)` is used to represent a continuous distribution such that the probability of seeing a value in an interval is equal to the integral of the pdf  
`cumulatie distribution function (cdf)` gives the probability that a random variable is less than or equal to a certain value  

## The Normal Distribution
completely characterized by mean (where its centered) and standard deviation (spread)  

$$f(x|\mu, \sigma) = \frac{1}{\sqrt{2 \pi} \sigma}\exp(-\frac{(x-\mu)^2}{2 \sigma^2})$$

`standard normal distribution` when $\mu=0$ and $\sigma=1$  
if Z is a standard normal random variable, then $X = \sigma Z + \mu$ is also normal, and  
$Z=\frac{X-\mu}{\sigma}$ is a standard normal variable  

sometimes we invert normal_cdf to find the value corresponding to a specified probability  

## The Central Limit Theorem
a random variable defined as the average of a large number of independent and identically distributed random variables is itself approximately normally distributed  

if x_i are random variables with mean $\mu$ and standard deviation $\sigma$, and n is large, then  
$\frac{1}{n}(x_1 + ... + x_n)$ is approximately normally distributed with mean $\mu$ and standard deviation $\frac{\sigma}{\sqrt{n}}$  

$\frac{(x_1+...+x_n)-\mu n}{\sigma \sqrt{n}}$ is approximately normally distributed with mean 0 and standard deviation 1

# Chapter 7 Hypothesis and Inference
## Statistical Hypothesis Testing
null hypothesis $H_0$ represents the default position  
alternative hypothesis $H_1$ represents a position of interest  
we use statistics to decide whether we can reject the null hypothesis or not  

`significance` is how willing we are to make a type 1 error  
`type 1 error` is a false positive (reject null hypothesis when its true)   
`power` is the probability of not making a type 2 error  
`type 2 error` is when we fail to reject null when it's false  
p-values is the probability of an event happening assuming that $H_0$ is true  

## Confidence Intervals
how confident we are that the real parameter lies within a set of bounds using the normal distribution  
`normal_two_sided_bounds(0.95, mu, sigma)`

## P-hacking
if you are setting out to find "significant" results, you usually can  
test enough hypotheses against the dataset, and one of them will appear significant  

good science involves determining the hypotheses before looking at the data and cleaning the data without the hypotheses in mind.  

## Bayesian Inference
start with a prior distribution for the parameters and use the observed data and Bayes' Theorem to get an updated posterior distribution for the parameters  

rather than making probability judgement about the tests, you make probability judgements about the parameters themselves  

if we flipped a coin more and more times, the prior would matter less and less until eventually we would have the same posterior distribution no matter which prior  

# Chapter 8 Gradient Descent

## The Idea Behind Gradient Descent

`gradient` is the vector of partial derivatives that gives the direction in which the function increases most rapidly  
we can minimize or maximize a function by continuously taking steps in the direction of the gradient  

## Estimating the Gradient
we can estimate the ith partial derivative by treating it as a function of just the ith variable, holding the other variables fixed   
this way is extremely computationally expensive  

## Choosing the Right Step Size
choosing the step size is more of an art than a science  
* using a fixed step size
* gradually shrinking the step size over time
* at each step, choose the step size that minimizes the value of objective function  

## Stochastic Gradient Descent 
computes the gradient and takes a step for only one point at a time  
during each cycle, iterate through the data in a random order, taking a gradient step for each data point  

# Chapter 9 Getting Data


## stdin and stdout
if running Python scripts at command line, you can pipe data through them using `sys.stdin` and `sys.stdout`  

```python
# reads in lines of text and spit out the ones that match an expression
import sys, re

regex = sys.argv[1]

for line in sys.stdin:
    if re.search(regex, line):
        sys.stdout.write(line)
```

```python
# prints the number of lines it receives
import sys

count = 0
for line in sys.stdin:
    count += 1

print count
```
`|` is the pipe character, which means use the output of the left command as the input of the right command

```python
# counts the words in the input and writes out the most common ones
import sys
from collections import Counter

try:
    num_words = int(sys.argv[1])
except:
    print "usage: most_common_word.py num_words"
    sys.exit(1)

counter = Counter(word.lower() # lowercase words
                for line in sys.stdin
                for word in line.strip().split() # split on spaces
                if word) # skip empty words
for word, count in counter.most_common(num_words):
    sys.stdout.write(str(count))
    sys.stdout.write("\t")
    sys.stdout.write(word)
    sys.stdout.write("\n")
```

## Reading Files
can also explicitly read from and write to files directly in code  

### The Basics of Text Files
first step is to obtain a file object using `open`  
`file = open('reading_file.txt', 'r')`  
should always open files with `with` block, which closes the file at the end automatically  

### Delimited Files
files often contain comma-separated or tab-separated fields  
if file has no headers, we probably want each row as a list  
```python
import csv

with open('tab_delimited_stock_prices.txt', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        process(data, symbol, closing_price)
```

if a file does have headers  
we can either skip the header row or get each row as a dict  

```python
with open('colon_delimited_stock_prices.txt', 'rb') as f:
    reader = csv.DictReader(f, delimiter=':')
    for row in reader:
        date = row['data']
        symbol = row['symbol']
        closing_price = float(row['closing_price'])
        process(date, symbol, closing_price)
```

## Scraping the Web
fetching web pages is easy, getting meaningful structured info is not  

### HTML and the Parsing Thereof

```html
<html>
    <head>
        <title>A web page</title>
    </head>
    <body>
        <p id="author">Joel Grus</p>
        <p id="subject">Data Science</p>
    </body>
</html>
```

in a perfect world, all web pages are marked up semantically, but in general HTML is not well-formed or annotated  
we can use the BeautifulSoup library  

## Using APIs
many websites and web services provide application programming interfaces (APIs) which allows you to request data in a structured format.  

### JSON (and XML)
HTTP is a protocol for transferring text, the data you request through a web API needs to be serialized into string format.  
this serialization often uses JavaScript Object Notation (JSON), which looks quite similar to Python dicts  

we can parse JSON using Python's `json` module  

### Using an Unauthenticated API
boring and technical stuff, could search up on google later  


# Chapter 10 Working with Data
first step with data should be to explore the data, not immediately building models and looking for answers  

## Exploring One-Dimensional Data
the simplest case is 1D dataset, which is a collection of numbers  
we can first compute a few summary statistics (mean, median, range, etc.)  
a good next step is to create a histogram  

## Two Dimensions
a scatter plot is probably useful to view the joint relationship between fields  

## Many Dimensions
to figure out how all dimensions relate to one another, we can create a correlation matrix,   
in which the ith and jth entry is the correlation between the ith dimension and jth dimension of the data   
another option if we don't have too many dimensions is a scatter plot matrix   

## Cleaning and Munging
figure out how to clean munge (transform data into a useful form) data lol

## Manipulating Data
specific to the task, figure out on your own lol

## Rescaling
many techniques are sensitive to the scale and units of the data (such as calculating distances)  
we can fix this by rescaling data so that each dimension has mean 0 and standard deviation 1, effectively getting rid of all units  

## Dimensionality Reduction
sometimes the "actual" (or useful) dimensions of the data might not correspond to the dimensions we have  
when most of the variation seems to be along a single dimension that doesn't correspond with any of the existing axes, we can use  
principal component analysis to extract one or more dimensions that capture as much of the variation in the data as possible  

1. convert each dimension to have mean zero
1. find the direction which captures the greatest variance in the data
1. project the data onto the direction to find the values of that component
1. for further components, remove the projects from the data and repeat

PCA can clean data by eliminating noise dimensions and consolidating dimensions that are highly correlated  
after extracting low-dimensional representation of data, we can use a variety of techniques that don't work as well on high-dimensional data  
while PCA can help you build better models, it can make those models harder to interpret  

# Chapter 11 Machine Learning


a `model` is simply a specification of mathematical or probabilistic relationship between different variables  

`machine learning` in this book refers to creating and using models learned from data (aka predictive modeling or data mining)  

`supervised` models has access to data with correct labels  
`unsupervised` models do not have data with correct labels  


`overfitting` is when a model performs well on training data but fails to generalize to new data  
could be due to noise in data or learning to predict inputs rather than factors/traits in the data  

`underfitting` is when a model doesn't perform well even on training data

the most simple way to combat overfitting it by splitting the data into a training and testing set randomly   

## Correctness

in a binary judgment scenario, we can separate the decisions into four categories:    
`true positive`, `false positive` (Type 1 Error), `false negative` (Type 2 Error), `true negative`  
these are represented in a confusion matrix   


`accuracy` is defined as the fraction of correct predictions out of the total    
`precision` is how accurate the positive predictions are $\frac{tp}{tp+fp}$  
`recall` is the fraction of positives the model identified $\frac{tp}{tp+fn}$  
sometimes precision and recall are combined into F1 score with $F1 = \frac{2\times p \times r}{p + r}$  
this is the harmonic mean of precision and recall   
models have a tradeoff between precision and recall   
think of as a tradeoff between false positives and false negatives  

`bias-variance trade-off`
a model that doesn't fit the data well has high bias   
we can add more features to the model  
a model could have high variance across datasets   


`features` are simply the inputs to the models

# Chapter 12 k-Nearest Neighbors
suppose you are trying to predict how a person is going to vote and you have some information about their demographic, income, kids, etc. A good predictor is to look at people similar in those dimensions (neighbors) and how they voted  

## The Model
requires
* notion of distance
* assumption that points close to one another are similar

fallbacks
* neglects a lot of information because predictions only depend on points similar to the input point  
* is not a clear indicator of the drivers behinds phenomenons

process
1. decide on a number k
2. find the k-nearest neighbors and find the majority in a dimensions
    * pick one of winners at random
    * weight the votes by distance and pick weighted winner
    * reduce k by eliminating the farthest until there is a unique winner
3. return the winner for a given feature

`curse of dimensionality` causes k-nearest neighbors to perform poorly in higher dimensions because such data tends to be sparse and very 'far'  

# Chapter 13 Naive Bayes  
suppose we are trying to make a spam filter. The key assumption of Naive Bayes is that the presence or absence of each word are independent of one another, conditional on a message being spam or not   

$$P(S|X=x)=\frac{P(X=x|S)}{P(X=x|S) + P(X=x|\neg S)}$$

the assumption allows us to compute each of the probability on the right by multiplying together the probabilities or the partitions  

in practice, underflow is avoided by exponential addition using logs  

to avoid commonly used words becoming one of the identifiers, we use a pseudocount

$$P(X_i |S) = \frac{k + \text{number of spam containing } w_i}{2k + \text{number of spams}}$$

# Chapter 14 Simple Linear Regression
for linear relationships, hypothesize a linear model  
$$y_i = \beta x_i + \alpha + \epsilon _i$$

the error between the model and the actual data is calculated using square errors  

Using calculus and partial derivatives, we can find the minimum of the linear model by setting the gradients to zero and solving the system
```
beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
alpha = mean(y) - beta * mean(x)
```
`coefficient of determination (R-squared)` measures the fraction of the total variation in the dependent variable captured by the model  
it is 1 minus the residual sum of squares divided by the total sum of squares   

we can also solve this problem using gradient descent  

```python
random.seed(0)
theta = [random.random(), random.random()] # random starting point
alpha, beta, = minimize_stochastic(squared_error,
                                   squared_error_gradient, # need to define a gradient function
                                   num_friends_good, # provide actual data
                                   daily_minutes_good, 
                                   theta, 
                                   0.0001)
```

we use the least squares error because of the `maximum likelihood estimation`, which is largets when alpha and beta are chosen to minimize the sum of squared errors, due to the properties of normal distribution.

# Chapter 15 Multiple Regression

a linear model with more independent variables  
the input is a vector of k numbers instead of just one number  

## Further Assumptions of Least Squares Model
 
* the columns of x are linearly independent/ the inputs are unrelated to each other
* the columns of x are all uncorrelated with the errors
> suppose the actual model is a function of number of friends and time spent working, 
> if we hypothesize that the model is only a function of friends, then we underestimate the coefficient for it  
> whenever independent variables are correlated with errors, the least squares solution gives us a biased estimation

## Fitting the Model

since the model has more than one independent variable, solving by hand with calculus gets tedious, so we use stochastic gradient descent  

- define the error function
- define the gradient of the error function
- set a random starting point
- perform gradient descent

## Interpreting the Model

the coefficients of the model represents the all-else-being-equal estimates of the impacts of each factor  
this doesn't directly tell us the interactions between the variables  
we could introduce new variables that are the **products** or **squares** of existing variables

## Goodness of Fit 

adding new variables will increase the R-squared  
we also need to look at the **standard errors** of the coefficients, which measures how certain we are about our estimates of each

the typical approach to measuring standard errors start with assuming that the unaccounted for error are independent normal random variables with mean 0 and some standard deviation  

## Bootstrapping

sometimes we can't be sure about certain statistics of a dataset (e.g. we don't know the distribution)  

we can solve by `bootstrapping`, or randomly sampling our dataset with replacement  
this allows us to have distribution-free inferences about the data  
we can calculate empirically the mean, median, standard deviation, confidence intervals, etc.  

## Standard Errors of Regression Coefficients

we can estimate the standard errors of beta coefficients by taking a sample and estimate beta on that  
if beta varies a lot (standard deviation across samples is high), then we can't be confident in the estimate  

the estimated beta divided by its standard deviation follows a *Student's t-distribution* with n-k degrees of freedom  

as the degrees of freedom gets large (n >> k), t-distribution approaches a normal distribution  

## Regularization

- the more variables, the more likely the model will overfit
- the more nonzero coefficients, the harder it is to interpret  

`regularization` is when we add a penalty to the error as the coefficients gets larger  

in *ridge regression*, the penalty is proportional to the sum of squares of beta_i  
another approach is *lasso regression*, where the penalty is the sum of absolute value of betas multiplied by a constant  
    this tends to force coefficients to be zero  
    good for learning sparse models  

# Chapter 16 Logistic Regression

problems with using multiple regression on a binary classification problem
- the predicted outputs should be 0 or 1, but the outputs of linear model can be huge numbers that are hard to interpret  
- because the outputs are not bounded, the error term always grows very large in one direction, creating bias  

the logistic function is capped between 0 and 1  
$$y_i = f(x_i \beta) + \epsilon_i$$
we can use gradient descent to optimize the function  

first we find the likelihood function and its gradient  
$$p(y_i|x_i, \beta) = f(x_i \beta)^{y_i}(1- f(x_i \beta))^{1-y_i}$$

it is easier to maximize the log likelihood function
$$\log L(\beta | x_i, y_i) = y_i \log f(x_i \beta) + (1-y_i) \log (1-f(x_i \beta))$$

if we assume all the data points are independent, then the total likelihood is the product of individual likelihoods  
the optimized function is hard to interpret, we can only say the general trends of data  

## Support Vector Machines

the boundary between classes is a hyperplane that splits the parameter space into a number of pieces  
an alternative approach to classification is to look for the hyperplane that best separates the data in the training data  
`support vector machine` find the hyperplane that maximizes the distance to the near point in each class  
sometimes a hyperplane might not exist at all  
sometimes we can get around this by transforming data into a higher-dimensional space  

for example, we can map the point x to (x, x**2)  
this is known as the **kernel** trick, which we take a function to compute dot products in the higher-dimensional space  
and find a hyperplane  

# Chapter 17 Decision Trees

uses a tree structure to predict outcomes  
easy to interpret and process is transparent  
very easy to overfit a tree  

`classification` tree produces categorical outputs  
`regression` tree produce numeric outputs   

`entropy` is the uncertainty in the data  
$$H(S) = p_1 \log _2 p_1 - ... - p_n \log _2 p_n$$
if p is the proportion of data labeled with class c  

the entropy of partitions of data is represented as  
$$H=q_1 H(S_1) + ... + q_m H(S_m)$$ 
where q is the proportion of the data contained in S   

`decision nodes` asks a question  
`leaf nodes` makes a prediction  

`ID3` algorithm
* if all data have the same label, create a leaf node that predicts the label and stop
* if list of attributes is empty, create a leaf node that predicts most common label and stop  
* otherwise, try partitioning data by attribute  
* choose partition with lowest partition entropy  
* add a decision node based on chosen attribute
* recur on each partitioned subset using remaining attributes  

`random forests` builds multiple decision trees and lets them vote on how to classify inputs  
one option is to train the decision trees on bootstrapped samples  
the original training set can be used as the test set in this case because of bootstrap aggregating or bagging  
another source of randomness is choosing a random subset of remaining attributes to partition and choosing the best  
    this is an example of `ensemble learning` in which several weak learners (high bias, low-variance models) produce an overall strong model

# Chapter 18 Neural Networks

neural networks are black boxes - inspecting their details doesn't give understanding of how they are solving a problem  

`perceptron` is a single neuron with n binary inputs, computes the weight sum of inputs and "fires" if it is greater than zero  

`feed-forward` neural networks contains an input layer, one or more hidden layers, and an output layer  

neural network require calculus for backpropagation, so smooth activation functions are needed  

`backpropagation` involves computing the gradient of the loss function as a function of each neuron's weights and adjusting the weights in the most negative direction  

# Chapter 19 Clustering 

supervised learning starts with a set of labeled data  
clustering is a type of unsupervised learning work with completely unlabeled data  

the idea is that data is likely to be clustered together  
certain groups of data have correlations with other groups  

`k-means` clustering method chooses the number of clusters k in advance and partitions the inputs into clusters such that the total sum of squared distances from each point to the mean of its assigned cluster is minimized  
there are a lot of ways to cluster, so finding the optimal is hard  
1. start with set of k-means
2. assign each point to the mean to which it is closest 
3. if no point's assignment has been changed, keep the assignment
4. recompute the means and return to step 2

**choosing k** - one approach is plotting the sum of squared error as a function of k and looking at where the graph bends

`bottom-up hierarchical` clustering  
1. make each input its own cluster of one
2. as long as there are multiple clusters, find the two closest clusters and merge them
3. we can get k clusters by unmerging them

we can use the minimum, maximum, or average distance between clusters

# Chapter 20 Natural Language Processing

`n-gram` model is where, given a starting word, randomly select a word from a pair (or group) and repeat until a period.  
to choose the starting word, pick randomly from words that follow a period  
the longer the gram, the less choices the model has, making it more likely to be verbatim to the original text  

`grammar` is another approach in which finite set of rules are applied to a sentence and random words are selected in accordance to those rules  

`gibbs sampling` is a techique for generating samples from multidimensional distributions when we only know some of the conditional distributions  
> start with any valid value for x and y and repeatedly alternate replacing x with random value picked conditional on y and same with y

**topic modeling** is used to understand the underlying connections between data  
`Latent Dirichlet Analysis` assumes that
* some fixed number K of topics
* there is some random variable that assigns each topic a probability distribution over words
* there is another random variable that assigns each document a probability distribution over topics
* each word in a document was generated by first randomly picking a topic and then randomly picking a word

# Chapter 21 Network Analysis

many data problems can be modeled in terms of networks, consisting of nodes and edges that joins nodes

undirected edges goes both ways, directed edges goes one direction  

`betweenness centrality` identifies nodes frequently on the shortest path between pairs of other nodes  

`closeness centrality` is the reciprical of the farness (the sum of the lengths of shortest paths to each other user)  

the two measures of centrality aren't often used on large networks  

`eigenvector centrality` is the entry corresponding to a node in the eigenvector of an adjacency matrix (where (i,j)th entry is either 1 if user i and j are friends)  
users with high eigenvector centrality have lots of connections and connections to people who themselves have high centrality  

`PageRank` algorithm takes into account who endorses and how much endorsements they have
1. there is a total of 1.0 PageRank in the network
2. initially equally distributed among nodes  
3. at each step, a large fraction of each node's PageRank is distributed evenly among outgoing links
4. at each step, the remainder of each node's PageRank is distributed evenly among all nodes

# Chapter 22 Recommender Systems
producing recommendations of some sort  

one approach is simply recommending what is popular  

**user-based collaborative filtering**  
we can measure how similar the users interests are using the angle between vectors  
this approach doesnt work well when number of items get very large  

**item-based collaborative filtering**  
compute similarities between interest directly and generate suggestions for each user by aggrebating interests that are similar to her current interests  

# Chapter 23 Databases and SQL
skipped bruh wtf is there a chapter on sql out of the blue

# Chapter 24 MapReduce
parallel processing on large data sets  
1. use a mapper function to turn each item into zero or more key-value pairs
2. collect together all pairs with identical keys
3. use a reducer function on each collection of group values to produce output values for corresponding key


