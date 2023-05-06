# Path: SecurityChecks/linkContentClassifier.py

def logFail(checkNum, message):
    with open("log.txt", "a") as f:
        f.write(f"Check #{checkNum} Failed: {message}\n")

def logPass(checkNum):
    with open("log.txt", "a") as f:
        f.write(f"Check #{checkNum} Passed \n")

def logError(checkNum, message):
    with open("full_log.txt", "a") as f:
        f.write(f"Check #{checkNum} Error: {message}\n")

def logSpamReason(reason):
    with open("/spams/reason", "a") as f:
        f.write(reason)
