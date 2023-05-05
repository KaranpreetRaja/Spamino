import sys
import os
import email
import email.parser
import csv
import re

def toTuple(mail, bSpam):
    
    body = mail.get_payload()
    body = re.sub('<(.|\n)*?>', '', body)
    subject = mail.get("Subject")

    rp = False 
    if mail.get("Return-Path") != None:
        rp = True

    bulkmail = bulkmain = mail.get("X-Bulkmail")
    if bulkmail == None:
        bulkmail = 0

    sender = mail.get('From')
    tld = ''

    if sender != None:
        tldList = open('tld.txt', 'r').read().splitlines()
        for line in tldList:
            x = re.search(f"@.*\.{line}(?:\n|\Z|\s|\.|>)+.*", sender.upper())
            if x != None:
                tld += "."+line.lower()


    return(subject, body, tld, rp, bulkmail, mail.get('X-Keywords'), bSpam)

def parseEmail(file):
    return email.parser.HeaderParser().parsestr(file)

def call(rootdir, bSpam):

    with open('data.csv', 'w') as stream:
        writer = csv.writer(stream)
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                path = os.path.join(subdir, file)
                f = open(path, 'rb')
                # print(f.read())
                content = f.read().decode("cp850")
                mail = parseEmail(content)
                writer.writerow(toTuple(mail, bSpam))

# call("dataset/ham/", True)
call("dataset/spam/", True)

# f = open('test.txt', 'r').read()
# print(re.sub('<(.|\n)*?>', '', f))