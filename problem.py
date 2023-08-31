import time

def main():

    list = []

    for i in range(1000000):
        list.append(i)

    start = time.time()

    length = len(list) - 1

    if length % 2:
        classification(list, length)
    else:
        if list[0] % 2:
            classification(list, length, -1)
        else:
            classification(list, length, 1)
        
    # print(list)

    end = time.time()
    print(end - start)


def classification(list, length, minus=0):
    i = 0
    while i < (length / 2):
        if list[i] % 2:
            temp = list[(length + minus) - i]
            list[(length + minus) - i] = list[i]
            list[i] = temp
        i = i + 1


main()

