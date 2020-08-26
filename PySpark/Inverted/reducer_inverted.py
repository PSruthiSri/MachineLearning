#!/usr/bin/env python3

from sys import stdin
import re

word_doc = {}

for word in stdin:
	words = word.split('\t')
	if words[0] in word_doc.keys():
		word_doc[words[0]].append(words[1].replace('\n',''))
	else:
		word_doc[words[0]]=[words[1].replace('\n','')]
for key in word_doc.keys():
	print(key,set(word_doc[key]))




