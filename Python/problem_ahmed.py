import time

numbers = []

for i in range(1000000):
    numbers.append(i)


start = time.time()

j = 0

i = 0

while i < len(numbers):
    if not numbers[i] % 2:
        temp = numbers[i]
        numbers[i] = numbers[j]
        numbers[j] = temp
        j = j + 1
    i = i + 1
   

# print(numbers)

end = time.time()
print(end - start)