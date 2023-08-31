# function that print '.'
def new_line():

    print('.')

# function takes the new_line function and repeats it 3 times
def three_lines():

    new_line()

    new_line()

    new_line()


# function takes the three_lines function and repeats it 3 times
def nine_lines():
    three_lines()
    three_lines()
    three_lines()


# the clear_screen function consists of two nine_lines function and two three_lines function and new_line function
def clear_screen():
    nine_lines()
    nine_lines()
    three_lines()
    three_lines()
    new_line()


# print Printing nine lines
print('Printing nine lines')


# call the function
nine_lines()


# print Calling clearScreen()
print('Calling clearScreen()')


# call the function
clear_screen()
    
