# Return the grade
def grade(x):
    if x < 60:
        return 'F'
    elif x < 70:
        return 'D'
    elif x < 80:
        return 'C'
    elif x < 90:
        return 'B'
    elif x <= 100:
         return 'A'
    else:
        return 'Invalid'



# function return even numbers
def even(lst):
    relst = []
    for i in lst:
        if i % 2 == 0:
            relst.append(i)
    return relst



    
# return max num  
def myMAx(x, y):
    if x > y:
        return x
    else:
        return y
    

# return min num    
def myMax3(x, y, z):
    lst = [x, y, z]
    max = lst[x]
    for i in lst:
        if i > max:
            max = i
    return max
    
        


# test grade
grades = 70

print(grade(grades))       
        
    
    
# test even num
lst = even([45, 6547, 2, 1, 5, 4, 6])

print(lst)


# test max
num = 5
num2 = 8
print(myMAx(num, num2))


# test min 
num3 = 64
num4 = 40
num5 = 4
print(myMax3(num3, num4, num5))

