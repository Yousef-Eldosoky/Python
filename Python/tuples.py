
def sum_all(t):
    k = 0
    for i in t:
        k += i
    return k

def main():
    
    print("enter numbers or '/' to stop")
    z = 5
    t = tuple()

    while True:
        x = input('num: ')
        if x == '/':
            break
        x = int(x)
        t = t[:] + (x,)
    
    print(sum_all(t))


if __name__ == "__main__":
    main()
