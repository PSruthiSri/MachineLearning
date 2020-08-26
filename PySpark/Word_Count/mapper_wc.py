#!/usr/bin/env python3
import sys
import re

for line in sys.stdin:
	line = line.strip()
	spclchars =line.translate ({ord(c): "" for c in "!@#$%^&*()[]{};:,/<>?\|`~=_+—"})
	#print(type(spclchars))
	spclchars = spclchars.replace('. ',' ')
	for word in spclchars.split():
		print(word.lower(), "1")

    
    
