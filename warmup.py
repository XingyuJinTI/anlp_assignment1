
#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
from random import random
from math import log
from collections import defaultdict


tri_counts=defaultdict(int) #counts of all trigrams in input


#this function currently does nothing.
def preprocess_line(line):
    return line


#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1] #get input argument: the training file

#This bit of code gives an example of how you might extract trigram counts
#from a file, line by line. If you plan to use or modify this code,
#please ensure you understand what it is actually doing, especially at the
#beginning and end of each line. Depending on how you write the rest of
#your program, you may need to modify this code.
with open(infile) as f:
    for line in f:
        line = preprocess_line(line) #doesn't do anything yet.
        for j in range(len(line)-(3)):
            trigram = line[j:j+3]
            tri_counts[trigram] += 1


#Some example code that prints out the counts. For small input files
#the counts are easy to look at but for larger files you can redirect
#to an output file (see Lab 1).
print("Trigram counts in ", infile, ", sorted alphabetically:")
for trigram in tri_counts.keys():
    print(trigram, ": ", tri_counts[trigram])
'''
print("Trigram counts in ", infile, ", sorted numerically:")
for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
    print(tri_count[0], ": ", str(tri_count[1]))
'''    
    
#open model_br.wu
with open("model-br.wu") as f:
    temp = 1  
    cnt = 0
    for line in f:
        line = preprocess_line(line) #doesn't do anything yet.
        #print(line)
        #print("Trigram counts in ", infile, ", sorted alphabetically:")
    
        for trigram in tri_counts.keys():
            if trigram == line[0:3]:
                for i in range(tri_counts.get(trigram)):
                    temp *= float(line[len(line)-10 : len(line)-1])
                    print(trigram,"-----",float(line[len(line)-10 : len(line)-1]))
                    cnt += 1
                print(float(line[len(line)-10 : len(line)-1]), " ------ ",temp)
    pp = pow(1/temp,1/cnt)
    print("perplexity : ",pp)
    print("temp : ",temp)
    print("cnt : ",cnt)


#print(tri_counts, ":  ", tri_counts.keys(), ":  ", tri_counts.items())       
   
''' 
       for j in range(len(line)-(3)):
            trigram = line[j:j+3]
            tri_counts[trigram] += 1
 '''           

