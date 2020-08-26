#!/usr/bin/env python3
import sys
import re

for line in sys.stdin:
	line = line.strip()
	line = line.split('\t')
	
	emp_id=salary=country=passcode= None

	if len(line) == 4:
		emp_id,salary,country,passcode= 		line[0],line[1],line[2],line[3].replace('\n','')
	else:
		emp_id,salary = line[0],line[1]

	print('{0}\t{1}\t{2}\t{3}'.format(emp_id,salary,country,passcode))
