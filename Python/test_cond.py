x = 5
y = 5
n = 10
# Conditional execution
if x > 0:
    print('x is positive')
    
# pass condition
if x < 0:
    pass       #TODO: need to handle negative values!


# Alternative execution
if x % 2 == 0:
    print('x is even')
else:
    print('x is odd')


# chained conditionals
if x > y:
    print('x is greater than y')
elif x < y:
    print('x is smaller than y')
else:
    print('x is equal y')


# Nested conditionals
if x == y:
    print('x and y are equal')
else:
    if x > y:
        print('x is greater than y')
    else:
        print('x is smaller than y')
        
        
def countdown(n):
    if n <= 0:
        print('Blastoff!')
    else:
        print(n) 
        countdown(n-1)

#countdown(n)


def print_n(s, n):
    if n <= 0:
        return
    else:
        print(s + ' ')
        print_n(s, n-1)


print_n('Joo', 1)


def do_n(n):
    if n <= 0:
        return
    else:
        print_n('Joo', 10)
        do_n(n-1)
    
    
#do_n(1000)

f = input('enter number: ')

f = int(f)

g = input('enter number: ')

g = int(g)

print(f+g)
