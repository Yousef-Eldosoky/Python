import math

def mysqrt(a):
    x = 5
    while True:
        y = (x + a/x) / 2.0
        if y == x:
            break
        x = y 
    return x

def test_square_root():
    a = 1
    while a < 26:
        diff = abs(mysqrt(a) - math.sqrt(a))
        print('a =', a, '|', 'my_sqrt(a) =', mysqrt(a), '|', 'math.sqrt(a) =', math.sqrt(a), '|', 'diff =', diff)
        a = a + 1
        
test_square_root()

n = 10000
count = 0
while n:
    count = count + 1
    n = n // 10
print (count)