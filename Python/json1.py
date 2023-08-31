import json
import urllib.request, urllib.parse, urllib.error
import ssl


# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input('Enter - ')
html = urllib.request.urlopen(url, context=ctx).read()


data = html

data = 

sum = 0




info = json.loads(data)
print('User count:', len(info))
# print(info['comments'][0]['count'])
# print(type(info))

for i in range(len(info['comments'])):
    sum = sum + int(info['comments'][i]['count'])
    
print('Sum:', sum)