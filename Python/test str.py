#Part 1:

# Create a string of words separated by spaces

string = "apple eggplant cherry fig banana durian grapefruit honeydew"



# Turn the string into a list of words using split

word_list = string.split()



# Delete three words from the list using different Python operations

del word_list[1] # delete the second word using del

word_list.remove("cherry") # remove the word "cherry" using list.remove

word_list = word_list[:5] # delete the last word using list slicing



# Sort the list of words

word_list.sort()



# Add new words to the list using different Python operations

word_list.append("kiwi") # add a word using list.append

word_list += ["lemon", "mango"] # add multiple words using list concatenation

word_list.extend(["orange", "watermelon"]) # add words using extend



# Turn the list of words back into a single string

new_string = " ".join(word_list)



# Print the resulting string

print(new_string)



# Delete word from the list

animals = ["lion", "elephant", "cat", "dog"]

animals.pop(1)

print(animals)
