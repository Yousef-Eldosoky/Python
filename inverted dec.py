# From Section 11.5 of: 
# Downey, A. (2015). Think Python: How to think like a computer scientist. Needham, Massachusetts: Green Tree Press. 


# open the file wich include the dictionary

fin = open('fruit.txt', 'r')

fin_all = fin.read()

fin.close()

# use the eval() function to put the string we take from the file and convert it to dictionary

fin_dic = dict()

fin_dic = eval(fin_all)    
    

# the function from the last unit that invert the dictionary

def invert_dict(d):
    inverse = dict()
    for key in d:
        val_list = d[key]
        for val in val_list:
            if val not in inverse:
                inverse[val] = [key]
            else:
               inverse[val].append(key)
    return inverse 

# open the file we want to put the inverted dictionary in

fin_output = open('inverted_dict.txt', 'w')

inverted = str(invert_dict(fin_dic))     # convert the dictionary to string

fin_output.write(inverted)     # put the string in the file

fin_output.close()