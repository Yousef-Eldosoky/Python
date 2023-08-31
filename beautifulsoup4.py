import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup

url = input('Enter - ')
html = urllib.request.urlopen(url).read()
soap = BeautifulSoup(html, 'html.parser')


tags = soap('a')
for tag in tags:
    print(tag.get('href', None))
    
    
    
