import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

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
y_train = df_train.pop("ROI")
x_train = df_train
y_test = df_test.pop("ROI")
x_test = df_test

# Feature Scaling
train_cols = x_train.columns
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=train_cols)
test_cols = x_test.columns
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=test_cols)

# Fit
x_train_lm = sm.add_constant(x_train)
y_train_lm = y_train.values.reshape(-1,1)
lm = sm.OLS(y_train_lm, x_train_lm).fit()
lm.summary()