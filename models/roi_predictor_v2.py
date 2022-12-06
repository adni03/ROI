# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import shap


# -----------------------------

class ROIAnalyzer:
    def __init__(self):
        self.rf = None
        self.X = None
        self.explainer = None
        self.shap_values = None
        self.college_names = None
        self.df = pd.read_csv('./data/college_data_working.csv')
        self.df.drop(['LAT', 'LNG', 'ZIP',
                      'UNITID', 'SATAverage', 'ACTMedian',
                      'AverageFacultySalary'], inplace=True, axis=1)
        self.df.dropna(inplace=True)
        self.dictionary = pd.read_csv('./data/dictionary.csv')

    def __scaleFeatures(self, X):
        cols = ['UGDS', 'UGDS_WHITE', 'UGDS_BLACK', 'UGDS_HISP', 'UGDS_ASIAN',
                'UGDS_AIAN', 'UGDS_NHPI', 'UGDS_2MOR', 'UGDS_NRA', 'UGDS_UNKN',
                'PCTFLOAN', 'CDR3', 'UGDS_MEN', 'UGDS_WOMEN', 'LPSTAFFORD_CNT',
                'LPSTAFFORD_AMT', 'PCIP01', 'PCIP03', 'PCIP04', 'PCIP05', 'PCIP09',
                'PCIP10', 'PCIP11', 'PCIP12', 'PCIP13', 'PCIP14', 'PCIP15', 'PCIP16',
                'PCIP19', 'PCIP22', 'PCIP23', 'PCIP24', 'PCIP25', 'PCIP26', 'PCIP27',
                'PCIP29', 'PCIP30', 'PCIP31', 'PCIP38', 'PCIP39', 'PCIP40', 'PCIP41',
                'PCIP42', 'PCIP43', 'PCIP44', 'PCIP45', 'PCIP46', 'PCIP47', 'PCIP48',
                'PCIP49', 'PCIP50', 'PCIP51', 'PCIP52', 'PCIP54', 'AdmissionRate',
                'AverageCost', 'Expenditure', 'MedianDebt', 'AverageAgeofEntry',
                'MedianFamilyIncome']

        ct = ColumnTransformer([
            ('scaler', StandardScaler(), cols)
        ], remainder='passthrough')
        X_cols = X.columns
        X = pd.DataFrame(ct.fit_transform(X), columns=X_cols)

        return X

    def __random_forest(self, X, y):
        rf = RandomForestRegressor(max_depth=10, random_state=1)
        rf.fit(X, y)

        return rf

    def train_analyzer(self):
        y = self.df.pop('MedianEarnings')
        self.X = self.df.copy()

        self.college_names = self.X.pop('INSTNM')
        self.X = pd.get_dummies(self.X)
        self.X = self.__scaleFeatures(self.X)

        self.rf = self.__random_forest(self.X, y)
        self.explainer = shap.TreeExplainer(self.rf)
        self.shap_values = self.explainer.shap_values(self.X)

    def get_feature_importance(self, k=10):
        features = self.X.columns
        importance = self.rf.feature_importances_
        importance = np.round(importance, 4)
        top_indices = np.argsort(importance)[-k:]
        imp_features = [features[i] for i in top_indices]
        feature_desc = [self.dictionary.loc[self.dictionary['Name'] == feature]['Definition'].values[0]
                        for feature in imp_features]

        return pd.DataFrame({'Feature': imp_features, 'Importance': importance[top_indices], 'Description': feature_desc})

    def get_local_importance(self, uni_name):
        idx = self.college_names[self.college_names == uni_name].index[0]

        return self.explainer.expected_value[0], self.shap_values[idx], self.X.iloc[idx]


if __name__ == '__main__':
    roi = ROIAnalyzer()
    roi.train_analyzer()
    feature_imp = roi.get_feature_importance(10)
    print(feature_imp)
