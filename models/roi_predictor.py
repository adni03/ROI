import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

df_raw = pd.read_csv("../data/college_data_working.csv")
df_raw.head()

# add ROI column assuming 10% of income used to pay off debt
scorecard_working = df_raw.copy()

# Correlation Matrix
correlation_matrix = scorecard_working.corr()
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x = correlation_matrix.columns,
        y = correlation_matrix.index,
        z = np.array(correlation_matrix),
        text=correlation_matrix.values,
    )
)
fig.show()

scorecard_working["ROI"] = np.ceil(10*12*scorecard_working["AverageCost"]/scorecard_working["MedianEarnings"])
featureList = ["Geography","AdmissionRate","ACTMedian","SATAverage","AverageCost","Expenditure","AverageFacultySalary","AverageAgeofEntry","ROI"]
scorecard_working = scorecard_working[featureList]

# Correlation Matrix with Feature Set
correlation_matrix = scorecard_working.corr()
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x = correlation_matrix.columns,
        y = correlation_matrix.index,
        z = np.array(correlation_matrix),
        text=correlation_matrix.values,
    )
)
fig.show()
# Should we consider some of the features correlated to median earnings?

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

# Predict
x_test = sm.add_constant(x_test)
y_pred = lm.predict(x_test)

# Predicted vs Actual
xmax = max(y_test)
ymax = max(y_pred)
plt.xlim((0, xmax))
plt.ylim((0, ymax))
plt.plot([0, xmax], [0, ymax]) # plots line y = x
plt.scatter(y_test,y_pred,color='blue')
plt.xlabel("Actual",fontsize=15)
plt.ylabel("Predicted",fontsize=15)
plt.title("Predicted vs. Actual",fontsize=18)

# Predicted vs. Residuals
plt.figure(figsize=(8,5))
p=plt.scatter(x=lm.fittedvalues,y=lm.resid,edgecolor='k')
xmin=min(lm.fittedvalues)
xmax = max(lm.fittedvalues)
plt.hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
plt.xlabel("ROI Predicted Values",fontsize=15)
plt.ylabel("Residuals",fontsize=15)
plt.title("Predicted vs. Residuals",fontsize=18)
plt.grid(True)
plt.show()

# Heteroscedasticity in larger ROI predictions (error non-normal)

# TODO: Convert ACT to SAT before feature scaling, prune insignificant vars prior to prediction