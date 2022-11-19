import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# def __init__(self, region, sat_score,
#                  act_score, funding_type,
#                  min_tuition=0, max_tuition=100000):
#     return

df_raw = pd.read_csv("../data/college_data_working.csv")
df_raw.head()

# add ROI column assuming 10% of income used to pay off debt
scorecard_working = df_raw.copy()
scorecard_working["ROI"] = np.ceil(10*12*scorecard_working["MedianDebt"]/scorecard_working["MedianEarnings"])
featureList = ["Geography","AdmissionRate","ACTMedian","SATAverage","AverageCost","Expenditure","AverageFacultySalary","AverageAgeofEntry","ROI"]
scorecard_working = scorecard_working[featureList]

# Feature Cleaning 
## dummify geography
geography = pd.get_dummies(scorecard_working['Geography'], drop_first = True)
scorecard_working = pd.concat([scorecard_working,geography],axis=1)
scorecard_working.drop(["Geography"],axis=1,inplace=True)

# Split train and test
np.random.seed(0)
df_train, df_test = train_test_split(scorecard_working,train_size=0.7, random_state=100)
x_train = df_train.loc[:,df_train.columns != "ROI"]
y_train = df_train["ROI"]
x_test = df_test.loc[:,df_test.columns != "ROI"]
y_test = df_test["ROI"]

# Pipeline for feature scaling (standardization) and fit
pipeline = Pipeline([
    ("StdScaling", StandardScaler()),
    ("LinReg", LinearRegression())
])

pipeline.fit(x_train,y_train)

y_pred = pipeline.predict(x_test)

plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')

p1 = max(max(y_pred), max(y_test))
p2 = min(min(y_pred), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

regression_results = pipeline.named_steps['LinReg']

regression_results.
