import requests


'''
response = requests.get('https://api.github.com/this-api-should-not-exist')
response=requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops")

response = requests.get(
    'https://api.github.com/search/repositories',
    params={'q': 'requests+language:python'},
)


response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')
with open(r'img.png','wb') as f:
    f.write(response.content)

'''

pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)

if response.status_code == 200:
    print('Success!')
elif response.status_code == 404:
    print('Not Found.')

print(response.status_code)
print(response.json())
