def logFail(checkNum, message):
    with open("log.txt", "a") as f:
        f.write(f"Check #{checkNum} Failed: {message}\n")

def logPass(checkNum):
    with open("log.txt", "a") as f:
        f.write(f"Check #{checkNum} Passed \n")
