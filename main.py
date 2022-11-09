import data_modules.data_processor as data_processor

df = data_processor.loadSkinnyCSV() #17s
df = data_processor.loadCSV(2000) #17.2s

dfList = []
for year in range(1996,2020):
    try:
        df = data_processor.loadCSV(year)
        dfList.append(df)
    except:
        print("error processing data for year " + str(year))