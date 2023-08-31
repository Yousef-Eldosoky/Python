# from zmq import NULL
avarge_sum = 0
avarge_count = 0
def avarge(x):
  global avarge_sum
  avarge_sum += x
  if (x > 0):
    global avarge_count
    avarge_count += 1
    
    
avarge(2)

print(avarge_sum)