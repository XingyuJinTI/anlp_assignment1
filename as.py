# -*- coding: utf-8 -*-
# @Time    : ${10,Oct.2018}
# @Author  : s1836694 & s1802612
# @FileName: ${as}.py
# @Software: ${Spyder3}

import re
import sys
from random import random
from math import log
import math
from collections import defaultdict
import numpy as np
from numpy.random import random_sample

tri_counts=defaultdict(float)       #counts of three characters in input file in trigrams
bi_counts=defaultdict(float)        #counts of two-character history in input file in trigrams
est_probs=defaultdict(float)
cpx_est_probs=defaultdict(dict)


# Q1 - preprocessing
# We remove characters with accents and umlauts and the other punctuation marks
# and also convert all digits to 0, which makes dataset only contain characters
# in English alphabet, space, ‘0’, ‘#’ and ‘.’ character.

def preprocess_line(line):
    
    line = line.lower()
    line = re.sub(r'[0-9]','0',line)

    #the extra thing we do
    line = re.sub(r'^','##',line)       #add '##' at the beginning of every paragraph 
    line = re.sub(r'\n','#\n',line)     #add '#' at the end of every paragraph
    ''' The reason for marking the start and the end of each sentence:
     -Marking the beginning of each sentence makes the 3-gram model 
    always have a starting state ‘##’ as two-character history 
    (when a new sequence starts or when the last sentence is over). 
     -The introduction of the ending symbol ‘#’ makes each sentence finite 
    and allows the model to split each sentence. '''

    line = re.sub(r'[^a-z0\s#.]','',line)       #remove other characters and punctuation marks
    return line


# Q4 - generate N charaters including '#' standing for the end of sentence, 
# but excluding '##' standing for the beginning of sentence
# The is reasonable because the model does output the ending symbel '#', 
# but cannot output the starting symbol '##'
def generate_from_LM(dictionary, N):
    output = ['#', '#']
    lst = range(2,N+2)
    for i in lst:
        key = str(output[i-2])+str(output[i-1])
        #see next comment firstly. There is no '\n' character in model, 
        #so it is neccessary to change the key
        if key == '#\n':        #stands for: the last sentence is over, new sentence starts with '\n'
            key = '##' 
        if key[0] == '\n':      #if output[i-2] is '\n', the real output[i-2] is '#'
            key = '#' + key[1]
        #if the sentence is over at '#', we start a new sentence with '\n' 
        if str(output[i-1])=='#' and key != '##':
            output = output+['\n']
        else:
            distribution = dictionary[key]
            outcomes = np.array(list(distribution.keys()))
            probs = np.array(list(distribution.values()))
            bins = np.cumsum(probs)
            output = output+list(outcomes[np.digitize(random_sample(1), bins)])
    return output

    
# Here we make sure the user provides a training filename when
# calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print('Usage: ', sys.argv[0], '<training_file>')
    sys.exit(1)
infile = sys.argv[1]        #get input argument: the training file


# Q3 - generate model on training dataset
# Q3 - Step 1: smooth model 
# We generate all possible keys for tri_counts and bi_counts.
# Like the 'model-br.en', we removed all keys of type '#_#' and '_#_' except "##_"
tri_keys = list()      # all the possible keys in tri_counts 
bi_keys = list()      # all the possible keys in bi_counts 
ch = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p',
     'q','r','s','t','u','v','w','x','y','z','0','.',' ','#']       #all the possible characters
alpha = 0.01        # the value in add-k smoothing method 

# generate all possible keys for tri_counts
tri_keys = [m + n + k for m in ch for n in ch for k in ch] 
for keys in tri_keys:
    if keys[2] == '#' and keys[0] == '#':
        continue
    if keys[1] == '#' and keys[0] != '#':
        continue
    else :
        tri_counts[keys] = 0

# generate all possible keys for bi_counts
bi_keys = [m + n for m in ch for n in ch ] 
for keys in bi_keys:
    bi_counts[keys] = 0
    
# Q3 - Step 2: traverse training file and calculate values of bi_counts and tri_counts dict
with open(infile) as f:
    for line in f:
        line = preprocess_line(line)
        for j in range(len(line)-(3)):
            trigram = line[j:j+3]
            tri_counts[trigram] += 1
        for i in range(len(line)-(2)):
            bigram = line[i:i+2]
            bi_counts[bigram] += 1

# Q3 - Step 3: use the add-k smoothing method to deal with some unseen contexts
for k2 in bi_keys:
    c = 0
    tplist = [k2 + m for m in ch ]
    for k in tplist:
        tri_counts[k] += alpha
        c += 1
    bi_counts[k2] += c * alpha

# Q3 - Step 4: generate model file
# generate model (cpx_est_probs) using complex data structure (dict in dict) 
# - the value of dictionary is the distribution of the third character given two-character history
complex_model_file = open('myModel', 'w+')
count = 0
temp_trigram=defaultdict(float)
for bigram in sorted(bi_counts.keys()):
    for trigram in sorted(tri_counts.keys()):
        if bigram == trigram[0:2]:
            temp_trigram[trigram[-1]]=tri_counts[trigram]/bi_counts[bigram]
            count+=1
            complex_model_file.write(bigram+trigram[-1]+'\t'+str('%.3e'%temp_trigram[trigram[-1]])+'\n')
    cpx_est_probs[bigram] = temp_trigram.copy()
    temp_trigram.clear()
complex_model_file.close()

# Q4 - use language model to genarate  random output sequence 
# Q4 - Step 1：read parameters from model file, then store in dict rd_model -- key is three character
rd_model=defaultdict(float)
#with open("model-br.en") as f:     # use model-br.en to genarate  random sequence
with open("myModel") as f:      # use myModel to genarate  random sequence
    for line in f:
        rd_trigram = line[0:3]
        rd_model[rd_trigram] = float(line[len(line)-10 : len(line)-1])

# Q4 - Step 2：extract all 2-characters history as the key of rd_cpx_model
rd_cpx_model=defaultdict(dict)
temporal_distribution=defaultdict(float)

for trigram in rd_model.keys():
    rd_history = trigram[0:2]
    rd_cpx_model[rd_history] = temporal_distribution.copy()

for rd_history in rd_cpx_model.keys():
    for trigram in rd_model.keys():
        if rd_history == trigram[0:2]:
            temporal_distribution[trigram[2]] = rd_model[trigram]

    rd_cpx_model[rd_history] = temporal_distribution.copy()
    temporal_distribution.clear()

# Q4 - Step 3： to generate sequence using the dict(dict) data structure
generated_list = generate_from_LM(rd_cpx_model, 300)
generated_file = open('generated.en', 'w+')
for i in range(len(generated_list)): 
    if generated_list[i] == '#':        # remove starting and ending symbol - '##'/'#'
        continue
    generated_file.write(generated_list[i])
generated_file.close()


# Q5 - Calculate Perplexity
# Q5 - Step 1：preprocess the test file and calulate test_counts 
test_counts=defaultdict(int)        #counts of test dataset
with open("test") as f:
    for line in f:
        line = preprocess_line(line)
        for j in range(len(line)-(3)):
            trigram = line[j:j+3]
            test_counts[trigram] += 1

# Q5 - Step 2：read parameters from model file, then store in dict rd_model
rd_model=defaultdict(float)
with open("myModel") as f:      #calculate the perplexity under myModel
#with open("model-br.en") as f:      #calculate the perplexity under model-br.en
    for line in f:
        rd_trigram = line[0:3]
        rd_model[rd_trigram] = float(line[len(line)-10 : len(line)-1])

# Q5 - Step 3：calculate perplexity 
prob = 0
cnt = 0
for trigram in test_counts.keys():
    for rd_trigram in rd_model.keys():
        if rd_trigram == trigram:
            for i in range(test_counts.get(trigram)):
                # in order to keep the code off overflow error, we compute the probabiliy by using log
                prob += log(rd_model.get(rd_trigram))       
                cnt += 1

pp = math.exp(-(1.0/cnt)*prob)
print("perplexity : ",pp)


