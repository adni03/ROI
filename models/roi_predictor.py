import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


def goLM(df, featureList):

    df_featured = df[featureList]

    # Correlation Matrix with Feature Set
    showCorrelationMatrix(df_featured)

    # Split train and test
    x_train,y_train,x_test,y_test = trainingSplit(df_featured)

    # Feature Scaling
    x_train,x_test = scaleFeatures(x_train,x_test)

    # Fit
    x_train_lm = sm.add_constant(x_train)
    y_train_lm = y_train.values.reshape(-1,1)
    lm = sm.OLS(y_train_lm, x_train_lm).fit()
    print(lm.summary())

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

def load_data():
    df_raw = pd.read_csv("../data/college_data_working.csv")
    df_raw.head()
    scorecard_working = df_raw.copy()
    return scorecard_working

def showCorrelationMatrix(df):
    # Correlation Matrix
    correlation_matrix = df.corr()
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x = correlation_matrix.columns,
            y = correlation_matrix.index,
            z = np.array(correlation_matrix)
        )
    )
    fig.show()
    return correlation_matrix

def goClean(df):
    ## dummify geography
    geography = pd.get_dummies(df['Geography'], drop_first = True)
    df_Clean = pd.concat([df,geography],axis=1)
    df_Clean.drop(["Geography"],axis=1,inplace=True)
    return df_Clean

def trainingSplit(df):
    # Split train and test
    np.random.seed(0)
    df_train, df_test = train_test_split(df,train_size=0.7, random_state=100)
    y_train = df_train.pop("ROI")
    x_train = df_train
    y_test = df_test.pop("ROI")
    x_test = df_test
    return x_train,y_train,x_test,y_test

def scaleFeatures(x_train,x_test):
    train_cols = x_train.columns
    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=train_cols)
    test_cols = x_test.columns
    x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=test_cols)
    return x_train,x_test

def removeStrings(df):
    return df.drop(df.select_dtypes(['object']).columns,axis=1)

def cleanNaN(df):
    return df.dropna()

def dropCollinear(df,corr):
    # Get upper triangle of matrix
    upper = corr.where(np.triu(np.ones(corr.shape), k = 1).astype(bool))
        
    to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.9)]
    print("dropping {} columns".format(len(to_drop)))
    return df.drop(to_drop,axis=1)

# load data
scorecard_working = load_data()
scorecard_working["ROI"] = np.ceil(10*12*scorecard_working["AverageCost"]/scorecard_working["MedianEarnings"])

# clean 
scorecard_working = cleanNaN(scorecard_working)
scorecard_working = scorecard_working.drop(["AverageCost","ACTMedian","CONTROL"], axis=1) # drop ACT because not as precise as SAT

# show correlation matrix of all columns
corr = showCorrelationMatrix(scorecard_working)
scorecard_working = dropCollinear(scorecard_working, corr)


# reduce to features for LM
featureList = ["MedianFamilyIncome","SATAverage","AverageFacultySalary","AverageAgeofEntry","ROI"]

# LM with important features
goLM(scorecard_working,featureList)
# Heteroscedasticity in larger ROI predictions (error non-normal)

# Random Forest
model = RandomForestRegressor(random_state=1, max_depth=10)
df=pd.get_dummies(scorecard_working)
x_train,y_train,x_test,y_test = trainingSplit(df)
model.fit(x_train,y_train)
features = df.columns
importances = model.feature_importances_
top10indices = np.argsort(importances)[-9:]  # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(top10indices)), importances[top10indices], color='b', align='center')
plt.yticks(range(len(top10indices)), [features[i] for i in top10indices])
plt.xlabel('Relative Importance')
plt.show()

important_features = [features[i] for i in top10indices]
important_features.append("ROI")

# LM with important features
goLM(df,important_features)
