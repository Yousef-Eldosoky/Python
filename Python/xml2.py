import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
import ssl

api_key = False
# If you have a Google Places API key, enter it here
# api_key = 'AIzaSy___IDByT70'
# https://developers.google.com/maps/documentation/geocoding/intro

if api_key is False:
    api_key = 42
    serviceurl = 'http://py4e-data.dr-chuck.net/xml?'
else :
    serviceurl = 'http://py4e-data.dr-chuck.net/comments_42.xml'

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input('Enter - ')
html = urllib.request.urlopen(url, context=ctx).read()


data = html

sum = 0

tree = ET.fromstring(data)
lst = tree.findall('comments/comment')
print('User count:', len(lst))
for item in lst:
    # print('Name:', item.find('name').text)
    # print('Count:', item.find('count').text)
    sum = sum + int(item.find('count').text)
    
print('Sum:', sum)