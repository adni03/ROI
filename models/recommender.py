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
        This function takes the whole dataframe and returns a subset based on the
        selections entered by the user via the streamlit app. Their selections
        are stored when a new Recommender object is called and the subsetting is
        done based on those
        :param df: full dataframe (essentially the whole working dataset)
        :return: a subset that satisfies the user's constraints
        """
        filtered_df = df.loc[df['Region'].isin(self.region)].copy()
        filtered_df = filtered_df.loc[filtered_df['FundingModel'] \
            .isin(self.funding_type)].copy()
        tuition_filter = (filtered_df['AverageCost'] >= self.min_tuition) \
                         & (filtered_df['AverageCost'] <= self.max_tuition)
        filtered_df = filtered_df.loc[tuition_filter]

        return filtered_df

    def __calculate_distance(self, point1, point2):
        """
        Given two points, this method calculates the euclidean
        distance between them
        :param point1:
        :param point2:
        :return: euclidean distance between the two points
        """
        point1[1] = self.sat_act_converter.predict(
            np.array(point1[1]).reshape((-1, 1))
        )
        point2[1] = self.sat_act_converter.predict(
            np.array(point2[1]).reshape((-1, 1))
        )
        return np.round(np.linalg.norm(point1 - point2), 4)

    def __train(self, df):
        """
        This method trains the KMeans Clustering model that is the
        basis of the university recommendation. The idea is:
            - Model is trained on data to identify 3 clusters
            - Cluster 1: schools that the user can get into
            - Cluster 2: Best fit cluster
            - Cluster 3: schools that are harder to get into
        :param df: training data
        :return: kmeans model
        """
        scores = df[['ACTMedian', 'SATAverage']].to_numpy()
        kmeans = KMeans(n_clusters=3, n_init=1000, random_state=0)
        kmeans.fit(scores)

        return kmeans

    def __getScores(self, dist):
        """
        Converts euclidean distances to a "Best Fit Score" as follows:
            score = (1/distance)/(sum over all 1/distance)
        :param dist: distances Series
        :return: ndarray of scores
        """
        invDist = 1/dist
        sumAll = np.sum(invDist)
        scores = invDist/sumAll
        scores = scores * 100
        return np.round(scores, 4)

    def predict(self, df):
        # filtering data to get subset
        subset = self.filter_data(df)

        # training on subset data
        kmeans_subset = self.__train(subset)

        # training on full dataset
        kmeans = self.__train(df)

        # obtaining full data labels
        cluster_labels = kmeans.labels_

        # obtaining subset data labels
        cluster_labels_subset = kmeans_subset.labels_

        user_scores = np.array([self.act_score, self.sat_score]) \
            .reshape((1, -1))

        # predictions of kmeans trained on full data
        raw_prediction = kmeans.predict(user_scores)

        # predictions of kmeans trained on subset data
        raw_prediction_subset = kmeans_subset.predict(user_scores)

        # obtaining masks
        boolean_mask = cluster_labels == raw_prediction[0]
        boolean_mask_subset = cluster_labels_subset == raw_prediction_subset[0]

        # subset the two dataframes based on the cluster the user belongs to
        university_cluster = df.loc[boolean_mask].copy()
        university_cluster_subset = subset.loc[boolean_mask_subset].copy()

        # calculate distances for the main df cluster
        university_cluster['Distances'] = university_cluster.apply(lambda x:
                                 self.__calculate_distance(
                                     np.array([x['ACTMedian'], x['SATAverage']]),
                                     np.squeeze(user_scores).copy()
                                 ), axis=1
                                 )

        # calculate distances for the subset df cluster
        university_cluster_subset['Distances'] = university_cluster_subset.apply(lambda x:
                               self.__calculate_distance(
                                   np.array([x['ACTMedian'], x['SATAverage']]),
                                   np.squeeze(user_scores).copy()
                               ), axis=1
                               )

        # sort the distances for both the result dfs
        university_cluster.sort_values(by=['Distances'], inplace=True)
        university_cluster_subset.sort_values(by=['Distances'], inplace=True)
        university_cluster.reset_index(inplace=True, drop=True)
        university_cluster_subset.reset_index(inplace=True, drop=True)

        # compute scores for both the dfs
        university_cluster['Score'] = self.__getScores(university_cluster['Distances'])
        university_cluster_subset['Score'] = self.__getScores(university_cluster_subset['Distances'])

        return university_cluster, university_cluster_subset
