import pandas as pd

def loadCSV(year):
    # all columns (raw)
    df = pd.read_csv("https://www.dropbox.com/s/jzra8h8ahesqcys/MERGED" + str(year) + "_01_PP.csv?raw=1")
    return df

def loadSkinnyCSV():
    # only 25 columns
    df = pd.read_csv("https://www.dropbox.com/s/jzra8h8ahesqcys/test.csv?raw=1")
    return df