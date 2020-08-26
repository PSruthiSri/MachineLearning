#!/usr/bin/env python3
import sys
import re

join1={}
join2={}

for line in sys.stdin:
	line = line.split('\t')
	if line[2]=='None':
		join1[line[0]]=line[1]
	else:
		join2[line[0]]=[line[1],line[2],line[3].replace('\n','')]
							

for keys in join2.keys():
	
	#print(keys,join1[keys],join2[keys][0],join2[keys][1],join2[keys][2])
	print(keys, join1[keys],' '.join(join2[keys]))
		


