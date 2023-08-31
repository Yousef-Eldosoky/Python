# To run this, download the BeautifulSoup zip file
# http://www.py4e.com/code3/bs4.zip
# and unzip it in the same directory as this file

import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input('Enter - ')
html = urllib.request.urlopen(url, context=ctx).read()
soup = BeautifulSoup(html, 'html.parser')

num = int(input("Enter count: "))
pos = int(input("Enter position: "))

# Retrieve all of the anchor tags
count = 0
tags = soup('a')
for i in range(num):
    for tag in tags:
        count = count + 1
        if count == pos:
            print("Retrieving:", tag.get('href', None))
            url = tag.get('href', None)
            break
    # print(i)
    count = 0
    html = urllib.request.urlopen(url, context=ctx).read()
    soup = BeautifulSoup(html, 'html.parser')
    tags = soup('a')


print('Last name in sequence:', tag.contents[0])
