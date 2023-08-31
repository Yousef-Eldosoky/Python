alphabet = "abcdefghijklmnopqrstuvwxyz"   

test_dups = ["zzz","dog","bookkeeper","subdermatoglyphic","subdermatoglyphics"] 

test_miss = ["zzz","subdermatoglyphic","the quick brown fox jumps over the lazy dog"] 

# From Section 11.2 of: 

# Downey, A. (2015). Think Python: How to think like a computer scientist. Needham, Massachusetts: Green Tree Press. 

def histogram(s):
     d = dict()
     for c in s:
          if c not in d:
               d[c] = 1
          else:
               d[c] += 1
     return d 
 

print('the has_duplicates function:\n')

# the function use the return of histogram then it loops to see if it has any duplicates

def has_duplicates(s):
    d = histogram(s)
    for i in s:
        if d[i] != 1:
            return True
    return False
        
#print(has_duplicates('Joo'))       # for test


for i in test_dups:
    if has_duplicates(i) == True:
        print(i, 'has duplicates\n')
    else:
        print(i, 'has no duplicates\n')
        
print()

print('The missing_letters function:\n')
# the function use the return of histogram then it loops to see the missing letters and return them

def missing_letters(s):
    miss_list = []
    d = histogram(s)
    for i in alphabet:
        if i not in d:
            miss_list.append(i)
    return miss_list

#print(''.join(missing_letters(alphabet)))   # for test


for i in test_miss:
    miss = missing_letters(i)
    if miss == []:
        print(i, 'uses all the letters\n')
    else:
        print(i, 'is missing letters', ''.join(miss), '\n')