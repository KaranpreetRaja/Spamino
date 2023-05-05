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

logPass(1)
logFail(2, "This is a test")
logError(3, "This is a test")