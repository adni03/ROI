import streamlit as st
import pandas as pd
import altair as alt
import models.recommender as CollegeRecommender
from vega_datasets import data

@st.cache
def load_data():
    df = pd.read_csv('data/college_data_working.csv')
    return df

st.title("University MatchMaker")
with st.spinner(text="Loading data..."):
    df = load_data()
    st.text('Find the right college for you!')

    col1, col2, col3 = st.columns(3)

    with col1:
        sat_score_val = int(st.text_input('SAT Score', '1400'))

    with col2:
        act_score_val = int(st.text_input('ACT Score', '32'))

    with col3:
        funding_options = ['Private', 'Public']
        funding_sel = st.multiselect('Funding Model', funding_options)

    region_options = df['Region'].unique()
    region_sel = st.multiselect('Select region', region_options)

    values = st.slider(
        'Select a range for tuition',
        2000, 100000, (5000, 55000))

    if st.button('Find a Match!'):
        rec = CollegeRecommender.Recommender(region=region_sel, sat_score=sat_score_val,
                                             act_score=act_score_val, funding_type=funding_sel,
                                             min_tuition=values[0], max_tuition=values[1])
        uni_recs = rec.predict(df)

        print(uni_recs.head())

        admrate_df = uni_recs[['INSTNM', 'SCORE', 'LAT', 'LNG']]

        admrate_barchart = alt.Chart(admrate_df,
                                  title='Comparitive Best Fit Scores for Universities',
                                  width=420).mark_bar().encode(
            y=alt.X('INSTNM', title="University Name", sort='-x'),
            x=alt.Y('SCORE', title="Best Fit Score"),
            tooltip=[alt.Tooltip('SCORE:Q', title="Best Fit Score")]
        )
        admrate_barchart

        states = alt.topo_feature(data.us_10m.url, feature='states')
        backgroundMap = alt.Chart(states).mark_geoshape(
            fill='lightblue',
            stroke='white'
        ).project('albersUsa').properties(
            width=700,
            height=500
        )

        admrate_df['icon'] = '📍'
        points = alt.Chart(admrate_df).mark_text(
            size=25
        ).encode(
            latitude = 'LAT:Q',
            longitude = 'LNG:Q',
            text=alt.Text('icon'),
            tooltip = 'INSTNM'
        )

        backgroundMap + points