# --------------------
import pandas as pd
import numpy as np


# --------------------


class DataPreprocessor:
    def __init__(self, null_threshold):
        self.interesting_columns = []
        with open('../data/columns.txt', 'r') as r:
            for column in r:
                self.interesting_columns.append(column)

        self.OGFILE = 'data/MERGED2019.csv'
        self.CLEANFILE = 'data/downloaded_clean_2019.csv'
        self.ZIPFILE = 'data/zip_to_lat_lon.csv'
        self.YEAR = 2019
        self.threshold = null_threshold

    def load_csv(self, file):
        return pd.read_csv(file)

    def __replace_privacy_suppressed_nan(self, df):
        return df.replace('PrivacySuppressed', np.nan)

    def __return_cols_enough_data(self, df):
        ok_cols = []
        not_ok_cols = []
        accept_nulls = int(self.threshold * len(df))

        def return_null_counts(series):
            return series.isna().sum()

        df.apply(lambda x: ok_cols.append(x.name)
                if return_null_counts(x) < accept_nulls
                else not_ok_cols.append(x.name))

        return ok_cols, not_ok_cols

    def __build_col_list(self):
        df = self.load_csv(self.OGFILE)
        df = self.__replace_privacy_suppressed_nan(df)
        ok_cols, not_ok_cols = self.__return_cols_enough_data(df)
        df_enough_data = df.drop(not_ok_cols, axis=1)

        return df_enough_data

    def preprocess_data(self, write_flag=False):
        clean_df = self.load_csv(self.CLEANFILE)
        zip_df = self.load_csv(self.ZIPFILE)
        zip_df['ZIP'] = zip_df['ZIP'].astype(str).str.zfill(5)

        df_enough_data = self.__build_col_list()
        joined_df = pd.merge(df_enough_data, clean_df, on='INSTNM', how='inner')

        def shorted_zips(zip):
            return zip.split('-')[0]

        joined_df['ZIP'] = joined_df['ZIP'].apply(lambda x: shorted_zips(x))
        joined_df = pd.merge(joined_df, zip_df, on='ZIP', how='inner')

        if write_flag:
            joined_df.to_csv('data/college_data_working.csv')

        return joined_df


if __name__ == '__main__':
    dp = DataPreprocessor(null_threshold=0.2)
    joined_df = dp.preprocess_data(write_flag=True)
