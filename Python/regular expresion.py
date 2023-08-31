import re

count = 0

file = open("regex_sum_1782220.txt")

sum = 0


for line in file:
    x = re.findall('[0-9]+', line)
    if len(x) < 1:
        continue
    for i in range(len(x)):
        count = count + 1
        sum = sum + int(x[i])
    
        
    
  
print(sum)
print(count)
    