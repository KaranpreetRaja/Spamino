# returns all the links starting with http or https in the raw email
import re

# with open('email.txt', 'r') as file:
#     raw_content = file.read()
def getAllLinks(raw_content):
    raw_content = raw_content.replace('=\n', '')
    links = []
    sum = 0
    ans = re.findall(r'(https?://[^\s>]+[^/>\s"])', raw_content)
    for i in ans:
        links.append(i)
        sum+=1
    # print (sum, "links found")
    return links

# print (getAllLinks(raw_content))
