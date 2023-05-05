# code used to create the combined dataset of spam and ham
# dont forget to shuffel
import pandas as pd
import requests
from bs4 import BeautifulSoup

df = pd.read_csv("verified_online.csv")
spams = pd.DataFrame({"URL":df["url"], "spam":(1 for i in df["url"])})
print (spams)

links = []
front, end = "https://phishtank.org/phish_search.php?page=","&valid=n&Search=Search"
for i in range(5000):
    response = requests.get(front + str(i) + end)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.table

    for i, row in enumerate(table.find_all('tr')):
        if i == 0:
            pass
        else:
            link = row.find_all('td')[1].contents[0]
            if link[-3:]!="...":
                links.append(link)

hams = pd.DataFrame({"URL":links, "spam":(0 for i in links)})
print (hams)

res = pd.concat([hams,spams])
print(res)
res.to_csv('spam and ham.csv')
    

