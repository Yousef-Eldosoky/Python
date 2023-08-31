def is_between(x, y, z):
    return x <= y <=z


#print(is_between(3, 9, 8))



def factorial(n):
    space = ' ' * (4 * n)
    print(space, 'factorial', n)
    if not type(n) == int:
        print('Factorial is only defined for integers.')
        return
    elif n < 0:
        print('Factorial is not defined for negative integers.')
        return
    elif n == 0:
        return 1
    else:
        result = n * factorial(n-1)
        print(space, 'returning', n * factorial(n-1))
        return result


#print(factorial(4))


def fibonacci(n):
    space = ' ' * (4 * n)
    print(space, 'fibonacci', n)
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        result = fibonacci(n-1) + fibonacci(n-2)
        print(space, 'returning', result)
        return result

#print(fibonacci(4))

def A(m, n):
    if m == 0:
        return n + 1
    elif m > 0 and n == 0:
        return A(m-1, 1)
    elif m > 0 and n > 0:
        return A(m-1, A(m, n-1))

print(A(3, 4))