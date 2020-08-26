#!/usr/bin/env python3
import sys
import os

for line in sys.stdin:
    # Get the file path
	path = os.environ["map_input_file"]
    # Get the filename
	doc_id = os.path.basename(path)
        # Map the words and filenames
	line = line.strip()
	spclchars =line.translate ({ord(c): "" for c in "!@#$%^&*()[]{};:,/<>?\|`~=_+—"})
	spclchars = spclchars.replace('. ',' ')
	for word in spclchars.split():
        	print("%s\t%s" % (word.lower(), doc_id))
