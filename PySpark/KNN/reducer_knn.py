#!/usr/bin/env python3
from sys import stdin

import re
import statistics
import sys

train=[]
train1=[]
test=[]
word_doc = {}
for line in sys.stdin:
	line = line.split(',')
	line = list(map(float,line))
	if len(line) == 48:
		test.append(line)	
	elif len(line) ==49:
		train.append(line)
		train1.append(line[0:-1])

lis=[]

i=0
for i in range(0,len(test)):
	for j in range(0,len(train1)):
	#for j in range(0,10):
		sum1=0
		k=[]	
		zip_obj = zip(test[i],train1[j])
		for test11,train11 in zip_obj:
			k.append(abs(test11-train11))
			sum1=sum(k)
			#print(testnp[i],"$",sum1,train[j][48])
		lis.append(str(i)+"$"+str(sum1)+"$"+str(train[j][48]))


for word in lis:

	words = word.split('$')
	#label.append(words[2])

	if words[0] in word_doc.keys():
		word_doc[words[0]].append((words[1],words[2].replace('\n','')))
	
	else:
		word_doc[words[0]]=[(words[1],words[2].replace('\n',''))]

for key in word_doc.keys():
	word_doc[key]=sorted(word_doc[key])
	label=[]
	for i in range(0,91):
		label.append(int(float(word_doc[key][i][1])))
	print(statistics.mode(label))


