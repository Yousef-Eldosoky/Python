import math
# the countdown function
def countdown(n):
     if n <= 0:
          print('Blastoff!')
     else:
          print(n)
          countdown(n-1)


# the countup function
def countup(n):
     if n >= 0:
          print('Blastoff!')
     else:
          print(n)
          countup(n+1)


#num = int(input('Give number: '))

# ask is the num is pos or neg
#if num >= 0:
    #countdown(num)
#else:
    #countup(num)
    
password = '1234'

# check if the password is correct
def ask_password(x):
    password = input('type your password: ')
    if password == x:
        print('Correct password')
    else:
        print('Incorrect password')
        ask_password(x)

#ask_password(password)


def print_n(s, n):
    while n > 0:
        print(s)
        n = n - 1
        
#print_n('Yousef', 5)

#a = int(input('n: '))

def mysqrt(a):
    x = 5
    while True:
        y = (x + a/x) / 2.0
        if y == x:
            break
        x = y 
    return x

def test_square_root():
    #print('a' + (4*' ') + 'mysqrt(a)' + (4*' ') + 'math.sqrt(a)' + (4*' ') + 'diff')
    a = 1
    while a < 26:
        diff = abs(mysqrt(a) - math.sqrt(a))
        print('a =', a, '|', 'my_sqrt(a) =', mysqrt(a), '|', 'math.sqrt(a) =', math.sqrt(a), '|', 'diff =', diff)
        a = a + 1
        
test_square_root()
    

def eval_loop():
    while True:
        x = input('>>> ')
        if (x == 'done'):
            break
        print(eval(x))

#eval_loop()

fruit = 'banana'

def backward(fruit):
    index = len(fruit) - 1
    while index >= 0:
        letter = fruit[index]
        print(letter)
        index = index - 1
    
#backward('banana')
#print(fruit[:])

prefixes = 'JKLMNOPQ'
suffix = 'ack'

for letter in prefixes:
    if letter == 'O' or letter == 'Q':
        print(letter + 'u' + suffix)
    else:
        print(letter + suffix)


greeting = 'Hello, world!'

greeting = 'J' + greeting[1:]

#print(greeting)

def find(word, letter, index):
    while index < len(word):
        if word[index] == letter:
            return index
        index = index + 1
    return -1

def count(word, a):
    count = 0
    for letter in word:
        if letter == a:
            count = count + 1
    print(count)
