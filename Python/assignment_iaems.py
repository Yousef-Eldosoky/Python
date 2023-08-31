import re

# Declare a list of usernames
usernames = ['Ahmed', 'Yousef', 'Adel', 'Mohamed', 'Soad', 'Sayed', 'Mawada', 'Abas', 'Mona', 'Sara', 'Sondos']


print('Register to the site.\n\n')

# Asking for the username.
username = input('Enter username: ')

# loop untill the user type unique username
while True:
    if username not in usernames:
        break
    username = input('User name already exist\nEnter valid username: ')
        


print('\n\n\n')


# Asking for the E-mail.
email = input('Enter email: ')


# loop untill the user type a valied E-mail
while True:
    # using regular expressions to check the mail
    if re.match("^[a-z0-9]+@[a-z]+\.[a-z]+$", email):
        break
    email = input('Enter a valid email: ')
    


print('\n\n\n')



# Asking for the username.
password = input('Type a password: ')



# Looping untill get a strong password.
while True:
    check, check2, check3 = False, False, False
    for letter in password:
        if letter.isdigit():
            check = True
        if letter.isupper():
            check2 = True
    if len(password) >= 8:
        check3 = True
    if check and check2 and check3:
        break
    password = input('''Type a stronger password with upper cases and digits.\n(must be greater than eight): ''')
