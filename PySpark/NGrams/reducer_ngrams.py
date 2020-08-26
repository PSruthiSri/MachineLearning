#!/usr/bin/env python3
import sys
word_count={}
for word in sys.stdin:
    words = word.split()
    if words[0] in word_count.keys():
        word_count[words[0]] = word_count[words[0]]+1
    else:
        word_count[words[0]] = 1
sorted_dict = sorted(word_count.items(), key =lambda kv:(kv[1], kv[0]),reverse = True)
i=0
for keys,values in sorted_dict:
    if(i==10):
        break
    print(keys,values)
    i=i+1
