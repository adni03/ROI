# --------------------
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


# ---------------------

class Recommender:
    def __init__(self, region, sat_score,
                 act_score, funding_type,
                 min_tuition=0, max_tuition=100000):
        self.region = region
        self.sat_score = sat_score
        self.act_score = act_score
        self.funding_type = funding_type
        self.min_tuition = min_tuition
        self.max_tuition = max_tuition

        self.sat_act_df = pd.read_csv('data/SAT_to_ACT.csv')
        self.sat_act_converter = LinearRegression()
        self.sat_act_converter.fit(
            np.array(self.sat_act_df['SAT']).reshape((-1, 1)),
            np.array(self.sat_act_df['ACT'])
        )

    def filter_data(self, df):
        """
        Function that filters the dataframe based on:
          - region
          - min_tuition and max_tuition
          - funding_type
        Return: filtered df
        """

        filtered_df = df.loc[df['Region'].isin(self.region)].copy()
        filtered_df = filtered_df.loc[filtered_df['FundingModel'] \
            .isin(self.funding_type)].copy()
        tuition_filter = (filtered_df['AverageCost'] >= self.min_tuition) \
                         & (filtered_df['AverageCost'] <= self.max_tuition)
        filtered_df = filtered_df.loc[tuition_filter]

        return filtered_df

    def __calculate_distance(self, point1, point2):
        point1[1] = self.sat_act_converter.predict(
            np.array(point1[1]).reshape((-1, 1))
        )
        point2[1] = self.sat_act_converter.predict(
            np.array(point2[1]).reshape((-1, 1))
        )
        return np.linalg.norm(point1 - point2)

    def __train(self, df):
        scores = df[['ACTMedian', 'SATAverage']].to_numpy()
        kmeans = KMeans(n_clusters=3, n_init=1000, random_state=0)
        kmeans.fit(scores)

        return kmeans

    def __getScores(self, dist):
        invDist = 1/dist
        sumAll = np.sum(invDist)
        scores = invDist/sumAll
        scores = scores * 100
        return scores

    def predict(self, df):
        subset = self.filter_data(df)
        kmeans = self.__train(subset)
        cluster_labels = kmeans.labels_

        user_scores = np.array([self.act_score, self.sat_score]) \
            .reshape((1, -1))
        raw_prediction = kmeans.predict(user_scores)
        boolean_mask = cluster_labels == raw_prediction[0]
        university_cluster = subset.loc[boolean_mask]
        university_cluster.reset_index(inplace=True, drop=True)

        distances = []
        university_cluster.apply(lambda x:
                                 distances.append(self.__calculate_distance(
                                     np.array([x['ACTMedian'], x['SATAverage']]),
                                     np.squeeze(user_scores).copy()
                                 )), axis=1
                                 )

        sorted_dis = np.sort(distances)[:10]
        indices = np.argsort(sorted_dis)
        university_recs = university_cluster.loc[indices]
        university_recs['DISTANCES'] = sorted_dis
        university_recs['SCORE'] = self.__getScores(university_recs['DISTANCES'])
        return university_recs