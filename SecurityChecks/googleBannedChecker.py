#do u actually want documentation for this?
import pandas as pd

def isItBanned(URL):
    df = pd.read_csv("banned.txt")
    return df['url'].eq(URL).any()
