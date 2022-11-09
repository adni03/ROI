import pandas as pd

def loadCSV(year):
    df = pd.read_csv("https://www.dropbox.com/s/jzra8h8ahesqcys/MERGED" + str(year) + "_01_PP.csv?raw=1")
    return df

df = loadCSV(2000)