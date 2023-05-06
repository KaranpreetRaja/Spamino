# This file goes through every check in the SecurityChecks folder and runs them on the email. If any of them fail, the email is marked as spam.

import os
import SecurityChecks.checkMailContent
import logger

# make an object
class Checker:

    def __init__(self):
        path = "/emails/"
        self.vectorizer, self.model = SecurityChecks.CheckLinkContent.CheckLinkContent.InitializeVectorizerAndModel()

        # constantly running isSpam function
        while True:
            # if any files in path
            if os.listdir(path):
                # run isSpam function with first file in path and pass in file content as parameter
                # get content of first file
                file = open(path + os.listdir(path)[0], "r")
                cur, reason = self.isSpam(file.read())
                if cur:
                    logger.logSpamReason(reason)
                    # move copy file to "/spams/"
                    os.rename("/spams/" + os.listdir(path)[0])
                else:
                    # move copy file to "/mails/"
                    os.rename("/mails/" + os.listdir(path)[0])
                

    def isSpam(self, email):

        # use checkIPDomain at SecurityChecks/IPDomainChecker.py to check the IP and domain
        # if it fails, return True
        if SecurityChecks.IPDomainChecker.filterIP(email):
            return True, "Email IP/Domain was blocklisted"
        
        logger.logPass(2)

        # gets all the links from the email:
        links = SecurityChecks.getLinks.getAllLinks(email)


        for link in links:
            if SecurityChecks.checkLinkContent.checkLinkContent(link, self.vectorizer, self.model):
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
            if SecurityChecks.checkLinkContent.checkLinkContent(link, self.vectorizer, self.model):
                return True, "Content of one of the links was detected as spam"
        
        logger.logPass(5)

        return False, None
        
Checker()

    



