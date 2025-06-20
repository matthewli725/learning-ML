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


