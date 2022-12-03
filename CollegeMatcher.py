import streamlit as st
import pandas as pd
import altair as alt
import models.recommender as CollegeRecommender
from vega_datasets import data


@st.cache
def load_data():
    df = pd.read_csv('data/college_data_working.csv')
    return df

st.set_page_config(layout='wide', page_title='University MatchMaker')

add_selectbox = st.sidebar.selectbox('Select a view to explore:',
                                     ('University Recommender',
                                      'Explore Universities',
                                      'University ROI'))

# Method to transform data frame for degree bar chart
def get_degree_df(df, *filter):
    # include columns we're interested in
    columns = ['PCIP02', 'PCIP06', 'PCIP07', 'PCIP08']
    for i in range(10, 55):
        if i not in [17, 18, 20, 21, 28, 32, 33, 34, 35, 36, 37, 53]:
            columns.append('PCIP' + str(i))
    columns.append('INSTNM')
    # degrees to rename columns
    degrees = ['Agriculture', 'Natural Resources and Conservation', 'Architecture',
               'Area, Ethnic, Cultural, Gender, and Group Studies',
               'Communication and Journalism', 'Communications Technologies', 'Technicians and Support Services',
               'Computer And Information Sciences', 'Personal And Culinary Services', 'Education', 'Engineering',
               'Engineering Technologies And Engineering-Related Fields',
               'Foreign Languages, Literatures, And Linguistics',
               'Family and Consumer Sciences and Human Sciences', 'Legal Professions And Studies',
               'English Language And Literature',
               'Liberal Arts And Sciences, General Studies And Humanities', 'Library Science',
               'Biological And Biomedical Sciences',
               'Mathematics And Statistics', 'Military Technologies And Applied Sciences',
               'Multi and Interdisciplinary Studies',
               'Parks, Recreation, Leisure, And Fitness Studies', 'Philosophy And Religious Studies',
               'Theology And Religious Vocations',
               'Physical Sciences', 'Science Technologies and Technicians', 'Psychology',
               'Homeland Security, Law Enforcement, Firefighting And Related Protective Services',
               'Public Administration/Social Services', 'Social Sciences', 'Construction Trades',
               'Mechanic And Repair Technologies',
               'Precision Production', 'Transportation and Materials Moving', 'Visual and Performing Arts',
               'Health Professions',
               'Business, Management, Marketing, and Related Support Services', 'History']
    # Transform dataframe and include only required columns
    fos = df.loc[:, df.columns.isin(columns)].set_index('INSTNM')
    fos = fos.rename(columns=dict(zip(fos.columns.tolist(), degrees))).T
    fos = fos.T.reset_index()
    fos = pd.melt(fos, id_vars=['INSTNM'])
    print(fos)
    return fos

def get_race_df(df, *filter):
    # include columns we're interested in
    columns = ['UGDS_WHITE', 'UGDS_BLACK', 'UGDS_HISP', 'UGDS_ASIAN', 'UGDS_AIAN', 'UGDS_NHPI', 'UGDS_2MOR', 'UGDS_NRA',
               'UGDS_UNKN', 'UGDS_WHITENH', 'UGDS_BLACKNH', 'UGDS_API', 'UGDS_AIANOLD', 'UGDS_HISPOLD', 'INSTNM']

    # degrees to rename columns
    races = ['White','Black','Hispanic','Asian','American Indian/Alaskan Native','Native Hawaiian/Pacific Islander','Two or more','Non-resident alien',
             'Unknown','White Non-Hispanic','Black Non-Hispanic','Asian Pacific Islander','American Indian/Alaskan Native (prior to 2009)','Hispanic (prior to 2009)']

    # Transform dataframe and include only required columns
    race_df = df.loc[:, df.columns.isin(columns)].set_index('INSTNM')
    race_df = race_df.rename(columns=dict(zip(race_df.columns.tolist(), races))).T
    race_df = race_df.T.reset_index()
    race_df = pd.melt(race_df, id_vars=['INSTNM'])
    return race_df

def get_gender_df(df, *filter):
    # include columns we're interested in
    columns = ['UGDS_MEN', 'UGDS_WOMEN', 'INSTNM']
    # degrees to rename columns
    # Transform dataframe and include only required columns
    gender_df = df.loc[:, df.columns.isin(columns)].set_index('INSTNM')
    gender_df = gender_df.rename(columns=dict(zip(gender_df.columns.tolist(), ['Men','Women']))).T
    gender_df = gender_df.T.reset_index()
    gender_df = pd.melt(gender_df, id_vars=['INSTNM'])
    return gender_df


## Part 1: Code for recommender output display. This includes Sliders which take user input
## After providing input and clicking the button, a map of the recommended schools will be
## displayed, along with interactive bar charts for top 10 schools (recommended by similarity score)
## and for the top degrees by proportion of students in those schools.

if add_selectbox == 'University Recommender':
    with st.spinner(text="Loading data..."):
        df = load_data()
        st.text('Find the right college for you!')

        # widget layout/setup
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

        # When clicked...
        if st.button('Find a Match!'):
            # Get recommendations (right now trained on act_score adn sat_score on filtered dataset,
            # but working to train on all user input w/ dim reduction so that we can consistently
            # output the top 10 - some of the filters reduce training set drastically s.t. there are
            # only 4 schools total being trained on.)

            rec = CollegeRecommender.Recommender(region=region_sel, sat_score=sat_score_val,
                                                 act_score=act_score_val, funding_type=funding_sel,
                                                 min_tuition=values[0], max_tuition=values[1])

            whole_rec, uni_recs = rec.predict(df)
            # Note, I changed recommender.py s.t. it outputs the entire df and then index here for display
            admrate_df = uni_recs[:10]
            # distance_scores = uni_recs[['INSTNM', 'SCORE']]
            admrate_df[['Score']] = admrate_df[['Score']].apply(lambda x: round(x, 2))

            # Visualization column setup and layout
            # col0 for map, col1 for bar charts
            ### SOMETHING I'D LIKE TO CHANGE IS TOOLTIP SHOWING TRUE IF NOT OVER PINPOINT

            # map background
            states = alt.topo_feature(data.us_10m.url, feature='states')
            backgroundMap = alt.Chart(states).mark_geoshape(
                fill='lightblue',
                stroke='white').project(
                'albersUsa').properties(
                width=700,
                height=500
            )
            # pinpoints setup
            admrate_df['icon'] = '📍'
            points = alt.Chart(admrate_df).mark_text(
                size=25
            ).encode(
                latitude='LAT:Q',
                longitude='LNG:Q',
                text=alt.Text('icon'),
                tooltip='INSTNM'
            )
            st.write(backgroundMap + points)

            college_filter = alt.selection_multi(fields=['INSTNM'])

            admrate_barchart = alt.Chart(admrate_df,
                                         title='Comparitive Best Fit Scores for Universities',
                                         width=400).mark_bar(
                tooltip=True).encode(
                y=alt.X('INSTNM', title="University Name", sort='-x'),
                x=alt.Y('Score', title="Best Fit Score"),
                color=alt.condition(college_filter, alt.ColorValue("steelblue"), alt.ColorValue("grey")),
                tooltip=[alt.Tooltip('Score:Q', title="Best Fit Score"),
                         alt.Tooltip('SATAverage', title="Average SAT"),
                         alt.Tooltip('ACTMedian', title="Median ACT"),
                         alt.Tooltip('AverageCost', title='Average Cost'),
                         alt.Tooltip('AdmissionRate', title='Admission Rate')]).add_selection(
                college_filter).interactive()

            melted_df = pd.melt(admrate_df, id_vars=['INSTNM'],
                                value_vars=['UGDS_WOMEN', 'UGDS_MEN'],
                                value_name='ScoreValue',
                                var_name='Scores')

            melted_df['INSTNM'] = melted_df['INSTNM'].apply(lambda x: x.strip())

            act_sat_chart = alt.Chart(melted_df,
                                      title='Comparision of Gender at Universities'
                                      ).mark_bar().encode(
                x=alt.X('Scores:O'),
                y=alt.Y('ScoreValue:Q', title="Score Value"),
                color=alt.Color('Scores', scale=alt.Scale(scheme='paired'), legend=None),
                column=alt.Column('INSTNM:N', header=alt.Header(labelAngle=-90, labelAlign='right'), title="Universities")
            ).transform_filter(college_filter)

            melted_cost_df = pd.melt(admrate_df, id_vars=['INSTNM'],
                                value_vars=['AverageCost', 'Expenditure'],
                                value_name='Costs',
                                var_name='Scores')

            cost_barchart = alt.Chart(melted_cost_df).mark_bar().encode(
                x=alt.X('INSTNM', title="University Name"),
                y=alt.Y('sum(Costs)', title="Total Expenditure"),
                color=alt.Color('Scores', scale=alt.Scale(scheme='paired')),
                tooltip = [alt.Tooltip('sum(Costs)', title="Total Expenditure")]
            ).transform_filter(college_filter)

            hcharts = alt.hconcat(act_sat_chart, cost_barchart)
            charts = alt.vconcat(
                admrate_barchart, hcharts
            )
            charts


# PART 2: Interactive tool which allows user to investigate statistics and visualizations for a specific school
# Nice to have: If we finish updates to recommender.py, it'd be nice to have similarity score for the searched school.

if add_selectbox == 'Explore Universities':

    df = load_data()
    st.subheader("Explore a Specific University")

    # drop-down to select school
    school_options = df['INSTNM']
    school_sel = st.selectbox('Type or select a school', school_options)

    # masked_df
    masked_df = df[df['INSTNM'] == school_sel]
    st.markdown(
        f"""<h5 style='text-align: center; color: black; padding:50px'> {school_sel} is a {masked_df.FundingModel.values[0].lower()}\
     school located in the {masked_df.Region.values[0]}. It resides in a {masked_df.Geography.values[0]} \
    with the zip code {masked_df.ZIP.values[0]}. It accepted {round(masked_df.AdmissionRate.values[0] * 100,2)} percent of \
    students in 2019, and has {int(masked_df.UGDS.values[0])} students enrolled. The highest degree which can be attained here is a {masked_df.HighestDegree.values[0]}, \
    and the predominant degree type is {masked_df.PredominantDegree.values[0]}.""", unsafe_allow_html=True)

    # Admission Statistics
    st.markdown(f"""<h5 style='text-align: left; color: black'>ADMISSIONS STATISTICS</h5>""", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Average Admission Rate (%)", round(masked_df['AdmissionRate'] * 100, 2))
    col2.metric("Average SAT Score", masked_df.SATAverage)
    col3.metric("Average ACT Score", masked_df.ACTMedian)
    col4.metric("Median Family Income", '$' + str(int(masked_df.MedianFamilyIncome)))
    col5.metric("Average Age of Entry", int(round(masked_df.AverageAgeofEntry,0)))

    #Financial Summary
    st.markdown(f"""<h5 style='text-align: left; color: black'>FINANCIAL SUMMARY</h5>""", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Average Cost Per Year", '$' + str(masked_df.AverageCost.values[0]))
    col2.metric("Expenditure", '$' + str(masked_df.Expenditure.values[0]))
    col3.metric("Average Faculty Salary Per Month", '$' + str(masked_df.AverageFacultySalary.values[0]))
    col4.metric("Median Debt (Post-Graduation)", '$' + str(int(masked_df.MedianDebt)))
    col5.metric("Median Earnings (Post-Graduation)", '$' + str(int(masked_df.MedianEarnings)))

    # Create Demographic Visualizations (can't make this interactive with the given design because the data is already aggregated
    st.markdown(f"""<h5 style='text-align: left; color: black'>DYNAMIC VISUALIZATIONS</h5>""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    #create gender pie chart
    gender_pie = alt.Chart(get_gender_df(masked_df),
                           title='Percentage by Gender',
                           width=350
                           ).mark_arc().encode(
                           theta=alt.Theta(field="value", type="quantitative"),
                           color=alt.Color(field="variable", type="nominal"),
                           tooltip=[alt.Tooltip('value', title="Proportion")]
                           )
    # creat degree bar chart
    degree_select_barchart = alt.Chart(get_degree_df(masked_df),
                                            title='Top 10 Degrees (by Proportion)',
                                            width=600).mark_bar(
                    tooltip=True).encode(
                    y=alt.Y('variable', title="Degree Type", sort='-x'),
                    x=alt.X('value', title="Proportion"),
                    # if want to make a separate
                    # color='variable',
                    tooltip=[alt.Tooltip('value', title="Percentage")]).transform_window(
                    rank='rank(value)',
                    sort=[alt.SortField('value', order='descending')]).transform_filter(
                    alt.datum.rank <= 10).configure_legend(columns=2, orient='bottom')

    # crete gender bar chart
    race_barchart = alt.Chart(get_race_df(masked_df),
                                            title='Proportion by degree ',
                                            width=600).mark_bar(
                    tooltip=True).encode(
                    y=alt.Y('variable', title="Degree Type", sort='-x'),
                    x=alt.X('value', title="Proportion"),
                    # if want to make a separate
                    # color='variable',
                    tooltip=[alt.Tooltip('value', title="Percentage")]).transform_window(
                    rank='rank(value)',
                    sort=[alt.SortField('value', order='descending')]).transform_filter(
                    alt.datum.rank <= 10).configure_legend(columns=2, orient='bottom')

    # write visualizations to UI (feel free to update format)
    col1.write('\n')
    col1.write(degree_select_barchart)
    col1.write('\n')
    col1.write(race_barchart)

    col2.write('\n')
    col2.write(gender_pie)

