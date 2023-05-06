# This file goes through every check in the SecurityChecks folder and runs them on the email. If any of them fail, the email is marked as spam.

import os
import SecurityChecks.CheckLinkContent
import SecurityChecks.IPDomainChecker
import SecurityChecks.checkMailContent
import SecurityChecks.getLinks
import SecurityChecks.googleBannedChecker
import SecurityChecks
import logger
import shutil

class Checker:

    def __init__(self):
        path = "/emails/"
        self.vectorizer, self.model = SecurityChecks.CheckLinkContent.InitializeVectorizerAndModel()

        while True:
            if os.listdir(path):
                print("checking new file")
                file = open(path + os.listdir(path)[0], "r")
                cur, reason = self.isSpam(file.read())
                if cur:
                    logger.logSpamReason(reason)
                    shutil.move(path + os.listdir(path)[0], "/spams/" + os.listdir(path)[0])
                else:
                    shutil.move(path + os.listdir(path)[0], "/mails/" + os.listdir(path)[0])

    def isSpam(self, email):

        # use checkIPDomain at SecurityChecks/IPDomainChecker.py to check the IP and domain
        # if it fails, return True
        if SecurityChecks.IPDomainChecker.filterIP(email):
            return True, "Email IP/Domain was blocklisted"
        
        logger.logPass(2)

        # gets all the links from the email:
        links = SecurityChecks.getLinks.getAllLinks(email)


        for link in links:
            if SecurityChecks.CheckLinkContent.checkLinkContent(link, self.vectorizer, self.model):
                return True, "Link was banned by Google"
            
        logger.logPass(3)


        # use checkMailContent at SecurityChecks/checkMailContent.py to check the email content
        # if it fails, return True
        if SecurityChecks.checkMailContent.checkContent(email):
            return True, "Mail Content was detected as spam"
        
        logger.logPass(4)


        # use checkLinkContent at SecurityChecks/checkLinkContent.py to check the links in the email
        # if it fails, return True
        for link in links:
            if SecurityChecks.CheckLinkContent.checkLinkContent(link, self.vectorizer, self.model):
                return True, "Content of one of the links was detected as spam"
        
        logger.logPass(5)

        return False, None
        
Checker()

    



