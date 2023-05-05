import requests
import re
from logger import logFail, logError

proxy = "http://ba9959e02ffc9bf866212b7c4931924a06404e0f:@proxy.zenrows.com:8001"

def checkSBL(ip):
    try:
        url = "https://www.spamhaus.org/sbl/query/" + ip
        print(url)

        proxies = {"http": proxy, "https": proxy}
        response = requests.get(url, proxies=proxies, verify=False)

        print(response.text)
    except:
        logError(1, "Error checking SBL with Hostname")
        return True

    if response.status_code == 200:
        if ("is not in the SBL" in response.text):
            return True
        else:
            logFail(1, "IP is in the SBL")
            return False
    else:
        logError(1, "Response code was not 200 for SBL check, it was " + str(response.status_code) + " for " + url)
        return True


def checkPBL(ip):
    try:
        url = "https://www.spamhaus.org/pbl/query/" + ip
        print(url)

        proxies = {"http": proxy, "https": proxy}
        response = requests.get(url, proxies=proxies, verify=False)

        print(response.text)
    except:
        logError(1, "Error checking SBL with Hostname")
        return True

    if response.status_code == 200:
        if ("is not in the SBL" in response.text):
            return True
        else:
            logFail(1, "IP is in the SBL")
            return False
    else:
        logError(1, "Response code was not 200 for SBL check, it was " + str(response.status_code) + " for " + url)
        return True



def filterIP(header):
    # regular expression patterns to match IP addresses and domains
    ipPattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    domainPattern = r'@([A-Za-z0-9\-\.]+)'

    ipAddress = re.search(ipPattern, header).group(0)
    domain = re.search(domainPattern, header).group(1)

    passedSBL = checkSBL(domain)
    passedPBL = checkPBL(ipAddress)


    if passedSBL and passedPBL:
        return True
    else:
        return False
    
def test():
    print([filterIP("google.com"), filterIP("bardia.tech"), filterIP("facebook.com"), filterIP("youtube.com"), filterIP("reddit.com"), filterIP("twitter.com"), filterIP("instagram.com")])

test()


