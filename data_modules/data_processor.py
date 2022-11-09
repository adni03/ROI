import pandas as pd
import dropbox
import io

dbx = dropbox.Dropbox("sl.BSywEOvjDqXl4W6LCdRUfc4OP1oU19MjASj2e1zupB_EtWH5JrYG-W3KhIPW098paoT0FPwfpTGoGYRBEAyqrkJMpnDmCCz7o1Oh5N1diNLw7jyUrcowlutfYzNW4IMCNSv8kWaW")

def loadCSV(year):
    # all columns (raw)
    _, res = dbx.files_download("/MERGED" + str(year) + "_01_PP.csv")

    with io.BytesIO(res.content) as stream:
        df = pd.read_csv(stream, index_col=0)
        return df

def loadSkinnyCSV():
    _, res = dbx.files_download("/test_20features.csv")

    with io.BytesIO(res.content) as stream:
        df = pd.read_csv(stream, index_col=0)
        return df