#!/usr/bin/env python3
import sys
import re
words=[]
for line in sys.stdin:
	spclchars =line.translate ({ord(c): "" for c in "!@#$%^&*()[]{};:,/<>?\|`~=_+—"})
	spclchars = spclchars.replace('\n','')	
	spclchars = spclchars.replace('. ',' ')	
	spclchars=spclchars.replace('science','$')
	spclchars=spclchars.replace('sea','$')
	spclchars=spclchars.replace('fire','$')
	words1=spclchars.split()
	for x in words1:
		words.append(x)
for i in range(0,len(words)):
	if i==0 and words[0]=='$':
		print(words[i]+"_"+words[i+1]+"_"+words[i+2],1)
	elif i==1 and words[1]=='$':
		print(words[i-1]+"_"+words[i]+"_"+words[i+1],1)
		print(words[i]+"_"+words[i+1]+"_"+words[i+2],1)
	elif i== len(words)-1:
		if words[len(words)-1]=='$':
			print(words[i-2]+"_"+words[i-1]+"_"+words[i],1)
	elif i== len(words)-2:
		if words[len(words)-2]=='$':
			print(words[i-2]+"_"+words[i-1]+"_"+words[i],1)
			print(words[i-1]+"_"+words[i]+"_"+words[i+1],1)
	elif words[i]=='$':
		print(words[i-2]+"_"+words[i-1]+"_"+words[i],1)
		print(words[i-1]+"_"+words[i]+"_"+words[i+1],1)
		print(words[i]+"_"+words[i+1]+"_"+words[i+2],1)
	
