#!/usr/bin/env python3
import sys
word_count={}
for word in sys.stdin:
    words = word.split()
    if words[0] in word_count.keys():
        word_count[words[0]] = word_count[words[0]]+1
    else:
        word_count[words[0]] = 1
    #print(words)
#print(word_count)
for keys in word_count:
    print(keys," ", word_count[keys])
