import streamlit as st
import pandas as pd
import altair as alt
import models.recommender as CollegeRecommender
from vega_datasets import data
import models.roi_predictor_v2 as ROI
import shap
from streamlit_shap import st_shap


@st.cache
def load_data():
    df = pd.read_csv('data/college_data_working.csv')
    return df

st.set_page_config(layout='wide', page_title='University MatchMaker')

add_selectbox = st.sidebar.selectbox('Select a view to explore:',
                                     ('University Recommender',
                                      'Explore Universities',
                                      'University ROI Analysis'))

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
    fos['value'] = round(fos['value'] * 100, 2)
    return fos


def get_race_df(df, *filter):
    # include columns we're interested in
    columns = ['UGDS_WHITE', 'UGDS_BLACK', 'UGDS_HISP', 'UGDS_ASIAN', 'UGDS_AIAN', 'UGDS_NHPI', 'UGDS_2MOR', 'UGDS_NRA',
               'UGDS_UNKN', 'UGDS_WHITENH', 'UGDS_BLACKNH', 'UGDS_API', 'UGDS_AIANOLD', 'UGDS_HISPOLD', 'INSTNM']

    # degrees to rename columns
    races = ['White', 'Black', 'Hispanic', 'Asian', 'American Indian/Alaskan Native',
             'Native Hawaiian/Pacific Islander', 'Two or more', 'Non-resident alien',
             'Unknown', 'White Non-Hispanic', 'Black Non-Hispanic', 'Asian Pacific Islander',
             'American Indian/Alaskan Native (prior to 2009)', 'Hispanic (prior to 2009)']

    # Transform dataframe and include only required columns
    race_df = df.loc[:, df.columns.isin(columns)].set_index('INSTNM')
    race_df = race_df.rename(columns=dict(zip(race_df.columns.tolist(), races))).T
    race_df = race_df.T.reset_index()
    race_df = pd.melt(race_df, id_vars=['INSTNM'])
    race_df['value'] = round(race_df['value'] * 100, 2)
    return race_df


def get_gender_df(df, *filter):
    # include columns we're interested in
    columns = ['UGDS_MEN', 'UGDS_WOMEN', 'INSTNM']
    # degrees to rename columns
    # Transform dataframe and include only required columns
    gender_df = df.loc[:, df.columns.isin(columns)].set_index('INSTNM')
    gender_df = gender_df.rename(columns=dict(zip(gender_df.columns.tolist(), ['Men', 'Women']))).T
    gender_df = gender_df.T.reset_index()
    gender_df = pd.melt(gender_df, id_vars=['INSTNM'])
    gender_df['value'] = round(gender_df['value'] * 100, 2)
    return gender_df


## Part 1: Code for recommender output display. This includes Sliders which take user input
## After providing input and clicking the button, a map of the recommended schools will be
## displayed, along with interactive bar charts for top 10 schools (recommended by similarity score)
## and for the top degrees by proportion of students in those schools.

if add_selectbox == 'University Recommender':

    df = load_data()

    st.image('./pics/carnegie-hero-banner.jpg', caption='Source: https://www.oracle.com/customers/carnegie-mellon/')
    st.markdown(
        """<h1 style='text-align: left!important; color: black;'> University Matcher """, unsafe_allow_html=True)
    st.markdown(
        """<p style='text-align: left; color: #474c54'> University Matcher helps you get started on your
        college applications by <b>recommending universities</b>, <b>providing comparisons</b> and an
        expected <b>return-on-investment</b> for your education. Here you can find universities that
        suit you, dig deeper into demographics and financial information, and understand factors that
        contribute to a good return-on-investment.<br>""", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: left; color: #474c54'> Find the right college for you!", unsafe_allow_html=True)


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
        2000, 60000, (5000, 35000))

    # When clicked...
    if st.button('Find a Match!'):
        with st.spinner(text="Generating recommendations..."):
            # Get recommendations (right now trained on act_score adn sat_score on filtered dataset,
            # but working to train on all user input w/ dim reduction so that we can consistently
            # output the top 10 - some of the filters reduce training set drastically s.t. there are
            # only 4 schools total being trained on.)

            if (len(funding_sel) == 0):
                print('here ' ,funding_sel)
                funding_sel = ['Private', 'Public']

            if (len(region_sel) == 0):
                region_sel = region_options
            
            print('here out ' ,funding_sel)
            print('here out ' ,region_sel)

            rec = CollegeRecommender.Recommender(region=region_sel, sat_score=sat_score_val,
                                                 act_score=act_score_val, funding_type=funding_sel,
                                                 min_tuition=values[0], max_tuition=values[1])

            whole_rec, uni_recs = rec.predict(df)
            # Note, I changed recommender.py s.t. it outputs the entire df and then index here for display
            admrate_df = uni_recs[:10]
            print(admrate_df)
            # distance_scores = uni_recs[['INSTNM', 'SCORE']]
            admrate_df[['Score']] = admrate_df[['Score']].apply(lambda x: round(x, 2))

            # Visualization column setup and layout
            # col0 for map, col1 for bar charts
            ### SOMETHING I'D LIKE TO CHANGE IS TOOLTIP SHOWING TRUE IF NOT OVER PINPOINT

            st.markdown(
                """<p style='text-align: left; color: #474c54'> Based on your scores and 
                    preferences, here is a list of universities that best suits your profile.
                    We calculate a <b>Best Fit Score</b> and present universities in order of this score. If you'd like to know
                    more about how this score was computed, please expand the section below to find out. """, unsafe_allow_html=True)

            # map background
            states = alt.topo_feature(data.us_10m.url, feature='states')
            backgroundMap = alt.Chart(states).mark_geoshape(
                fill='lightblue',
                stroke='white').project(
                'albersUsa').properties(
                width=600,
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
            uni_map_col1, uni_map_col2 = st.columns([2, 3], gap="medium")
            with uni_map_col2:
                st.write(backgroundMap + points)
            with uni_map_col1:
                display_df = admrate_df[['INSTNM', 'AverageCost', 'MedianEarnings']].copy()
                display_df.rename(columns={'INSTNM': 'University', 'AverageCost': 'Tuition', 'MedianEarnings': 'Earnings'}
                                  , inplace=True)
                st.dataframe(display_df)

            with st.expander('How we calculate the Best Fit Score', expanded=True):
                st.markdown(
                    """<p style='text-align: left; color: #474c54'> We take into consideration your funding 
                and location preferences first and find universities that satisfy those criteria. Next, we look at the
                average SAT and ACT scores for each of these universities and calculate a squared difference. The smaller
                the distance is, better the fit. To make the score more intuitive, we invert the distances and divide by
                the sum of the inverted distances to arrive at the Best Fit Score.""",
                    unsafe_allow_html=True)

            st.info("""Shift+Click on the bars below to compare the Gender composition and Total Expense between universities.""")
            st.write()

            college_filter = alt.selection_multi(fields=['INSTNM'])

            admrate_barchart = alt.Chart(admrate_df,
                                         title='Comparitive Best Fit Scores for Universities',
                                         height=300,
                                         width=800).mark_bar(
                tooltip=True).encode(
                y=alt.X('INSTNM', title="University Name", sort='-x'),
                x=alt.Y('Score', title="Best Fit Score"),
                color=alt.condition(college_filter, alt.ColorValue("steelblue"), alt.ColorValue("lightblue")),
                tooltip=[alt.Tooltip('Score:Q', title="Best Fit Score"),
                         alt.Tooltip('SATAverage', title="Average SAT"),
                         alt.Tooltip('ACTMedian', title="Median ACT"),
                         alt.Tooltip('AverageCost', title='Average Cost'),
                         alt.Tooltip('AdmissionRate', title='Admission Rate')]).add_selection(
                college_filter)

            melted_df = pd.melt(admrate_df, id_vars=['INSTNM'],
                                value_vars=['UGDS_WOMEN', 'UGDS_MEN'],
                                value_name='ScoreValue',
                                var_name='Scores')

            melted_df['Scores'] = melted_df['Scores'].apply(lambda x: 'WOMEN' if x == 'UGDS_WOMEN' else 'MEN')
            melted_df['INSTNM'] = melted_df['INSTNM'].apply(lambda x: x.strip())

            gender_chart = alt.Chart(melted_df,
                                     title='Comparision of Gender at Universities'
                                     ).mark_bar().encode(
                x=alt.X('Scores:O', title="Gender"),
                y=alt.Y('ScoreValue:Q', title="Proportion of gender"),
                color=alt.Color('Scores', scale=alt.Scale(scheme='paired'), legend=None),
                column=alt.Column('INSTNM:N', header=alt.Header(labelAngle=-90, labelAlign='right'),
                                  title="Universities")
            ).transform_filter(college_filter)

            melted_cost_df = pd.melt(admrate_df, id_vars=['INSTNM'],
                                     value_vars=['AverageCost', 'Expenditure'],
                                     value_name='Costs',
                                     var_name='Scores')

            cost_barchart = alt.Chart(melted_cost_df,
                                      title='Comparison of expenditure at Universities').mark_bar().encode(
                x=alt.X('INSTNM', title="University Name"),
                y=alt.Y('sum(Costs)', title="Total Expenditure"),
                color=alt.Color('Scores', scale=alt.Scale(scheme='paired')),
                tooltip=[alt.Tooltip('Costs:Q', title='Expenditure')]
            ).transform_filter(college_filter)

            hcharts = alt.hconcat(gender_chart, cost_barchart)
            charts = alt.vconcat(
                admrate_barchart, hcharts
            )
            charts

            st.subheader(f"About your Best Fit University: {admrate_df['INSTNM'][0]}")
            school_sel = admrate_df['INSTNM'][0]
            masked_df = df[df['INSTNM'] == school_sel]
            st.markdown(
                f"""<p style='text-align: left; color: #474c54;'> {school_sel} is a {masked_df.FundingModel.values[0].lower()}\
                 school located in the {masked_df.Region.values[0]}. It resides in a {masked_df.Geography.values[0]} \
                with the zip code {masked_df.ZIP.values[0]}. It accepted {round(masked_df.AdmissionRate.values[0] * 100, 2)} percent of \
                students in 2019, and has {int(masked_df.UGDS.values[0])} students enrolled. The highest degree which can be attained here is a {masked_df.HighestDegree.values[0]}, \
                and the predominant degree type is {masked_df.PredominantDegree.values[0]}.""", unsafe_allow_html=True)

            # Admission Statistics
            st.markdown(f"""<h5 style='text-align: left; color: black'>ADMISSIONS STATISTICS</h5>""",
                        unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Average Admission Rate (%)", round(masked_df['AdmissionRate'] * 100, 2))
            col2.metric("Average SAT Score", masked_df.SATAverage)
            col3.metric("Average ACT Score", masked_df.ACTMedian)
            col4.metric("Median Family Income", '$' + str(int(masked_df.MedianFamilyIncome)))
            col5.metric("Average Age of Entry", int(round(masked_df.AverageAgeofEntry, 0)))

            # Financial Summary
            st.markdown(f"""<h5 style='text-align: left; color: black'>FINANCIAL SUMMARY</h5>""",
                        unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Average Cost Per Year", '$' + str(masked_df.AverageCost.values[0]))
            col2.metric("Expenditure", '$' + str(masked_df.Expenditure.values[0]))
            col3.metric("Average Faculty Salary Per Month", '$' + str(masked_df.AverageFacultySalary.values[0]))
            col4.metric("Median Debt (Post-Graduation)", '$' + str(int(masked_df.MedianDebt)))
            col5.metric("Median Earnings (Post-Graduation)", '$' + str(int(masked_df.MedianEarnings)))

# PART 2: Interactive tool which allows user to investigate statistics and visualizations for a specific school
# Nice to have: If we finish updates to recommender.py, it'd be nice to have similarity score for the searched school.

if add_selectbox == 'Explore Universities':
    df = load_data()
    st.image('./pics/carnegie-hero-banner.jpg', caption='Source: https://www.oracle.com/customers/carnegie-mellon/')
    st.markdown(
        """<h1 style='text-align: left!important; color: black;'> Explore a Specific University """, unsafe_allow_html=True)

    # drop-down to select school
    school_options = df['INSTNM']
    school_sel = st.selectbox('Type or select a school', school_options)

    # masked_df
    masked_df = df[df['INSTNM'] == school_sel]
    st.markdown(
        f"""<h5 style='text-align: center; color: black; padding:50px'> {school_sel} is a {masked_df.FundingModel.values[0].lower()}\
     school located in the {masked_df.Region.values[0]}. It resides in a {masked_df.Geography.values[0]} \
    with the zip code {masked_df.ZIP.values[0]}. It accepted {round(masked_df.AdmissionRate.values[0] * 100, 2)} percent of \
    students in 2019, and has {int(masked_df.UGDS.values[0])} students enrolled. The highest degree which can be attained here is a {masked_df.HighestDegree.values[0]}, \
    and the predominant degree type is {masked_df.PredominantDegree.values[0]}.""", unsafe_allow_html=True)

    # Admission Statistics
    st.markdown(f"""<h5 style='text-align: left; color: black'>ADMISSIONS STATISTICS</h5>""", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Average Admission Rate (%)", round(masked_df['AdmissionRate'] * 100, 2))
    col2.metric("Average SAT Score", masked_df.SATAverage)
    col3.metric("Average ACT Score", masked_df.ACTMedian)
    col4.metric("Median Family Income", '$' + str(int(masked_df.MedianFamilyIncome)))
    col5.metric("Average Age of Entry", int(round(masked_df.AverageAgeofEntry, 0)))

    # Financial Summary
    st.markdown(f"""<h5 style='text-align: left; color: black'>FINANCIAL SUMMARY</h5>""", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Average Cost Per Year", '$' + str(masked_df.AverageCost.values[0]))
    col2.metric("Expenditure", '$' + str(masked_df.Expenditure.values[0]))
    col3.metric("Average Faculty Salary Per Month", '$' + str(masked_df.AverageFacultySalary.values[0]))
    col4.metric("Median Debt (Post-Graduation)", '$' + str(int(masked_df.MedianDebt)))
    col5.metric("Median Earnings (Post-Graduation)", '$' + str(int(masked_df.MedianEarnings)))

    # Create Demographic Visualizations (can't make this interactive with the given design because the data is
    # already aggregated
    st.markdown(f"""<h5 style='text-align: left; color: black'>DYNAMIC VISUALIZATIONS</h5>""", unsafe_allow_html=True)

    # create gender pie chart
    gender_pie = alt.Chart(get_gender_df(masked_df),
                           title='Percentage by Gender',
                           width=250
                           ).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="variable", scale=alt.Scale(scheme='paired'), type="nominal"),
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
                              title='Proportion by Race ',
                              width=600).mark_bar(
        tooltip=True).encode(
        y=alt.Y('variable', title="Race", sort='-x'),
        x=alt.X('value', title="Proportion"),
        # if want to make a separate
        # color='variable',
        tooltip=[alt.Tooltip('value', title="Percentage")]).transform_window(
        rank='rank(value)',
        sort=[alt.SortField('value', order='descending')]).transform_filter(
        alt.datum.rank <= 10).configure_legend(columns=2, orient='bottom')

    # write visualizations to UI (feel free to update format)
    
    col1, col2 = st.columns([2, 1])

    with col1:
        df = get_degree_df(masked_df).sort_values(by = ['value'], ascending=False)
        max_degree = df.values.tolist()[0]
        st.markdown(f"""<p style='text-align: left; color: black;'> The top 10 degrees at {school_sel} are as shown below.\
            The degree with the highest proportion is <strong>{max_degree[1]}</strong> with a proportion of \
                <strong>{max_degree[2]}%</strong> of the total degrees awarded. Hover over each bar to see the percentage \
                of the degree.""", unsafe_allow_html=True)
        col1.write('\n')
        col1.write(degree_select_barchart)
        col1.write('\n')
        df_race = get_race_df(masked_df).sort_values(by = ['value'], ascending=False)
        max_race = df_race.values.tolist()[0]
        st.markdown(f"""<p style='text-align: left; color: black;'> The proportion of different races at {school_sel} are \
            as shown below. The race with the highest percentage is <strong>{max_race[1]}</strong> with a proportion of \
                <strong>{max_race[2]}%</strong> of the total student population. Hover over each bar to see the percentage \
                of each race.""", unsafe_allow_html=True)
        col1.write(race_barchart)

    with col2:
        df = get_gender_df(masked_df)
        st.markdown(f"""<p style='text-align: left; color: black;'> Men make up <strong>{df['value'][0]}% </strong> and women make up \
            <strong>{df['value'][1]}%</strong> of the student population at {school_sel}""",
                unsafe_allow_html=True)
        col2.write(gender_pie)

# PART 3: ROI Analysis

if add_selectbox == 'University ROI Analysis':
    st.image('./pics/carnegie-hero-banner.jpg', caption='Source: https://www.oracle.com/customers/carnegie-mellon/')
    st.markdown(
        """<h1 style='text-align: left!important; color: black;'> What Impacts your ROI? """, unsafe_allow_html=True)
    st.markdown(
        """<p style='text-align: left; color: #474c54'> Most students attend college in order to get a better job 
        with a higher salary. But the <b>financial returns</b> to college vary widely depending on the <b>institution</b> a student 
        attends and the <b>subject</b> they study. While prospective students often ask themselves whether college is 
        worth it, the more important questions are how they can make college worth it and <b> what factors affect the return
        on their investment.</b> Here, we give you a more comprehensive analysis based on <b>SHAP or
        SHapley Additive Explanations. </b>""", unsafe_allow_html=True)

    st.subheader('Relative Variable Importance')
    st.markdown(
        """<p style='text-align: left; color: #474c54'> The Relative variable importance graph plots the independent variables 
        <b>in order of their effect</b> on the return on investment on education <b>across all universities</b>. 
        We consider the <b>earnings after graduation</b> as the measure of return on investment. The variable with the 
        highest score is set as the most important variable, and the other variables follow in order of importance.""",
        unsafe_allow_html=True)

    st.markdown(
        """<p style='text-align: left; color: #474c54'> For this analysis, the following plot communicates the factors
        you should consider to <b>maximize</b> the return on your investment.""", unsafe_allow_html=True)

    roi = ROI.ROIAnalyzer()
    with st.spinner(text="Calculating importance values..."):
        roi.train_analyzer()

        rel_feat_col1, _, _ = st.columns(3)
        with rel_feat_col1:
            num_feats = st.slider('Select the number of factors to view', 0, 30, 10)

        feature_imp = roi.get_feature_importance(num_feats)
        feature_imp.sort_values(by=['Importance'], ascending=False, inplace=True)

        rel_imp_barchart = alt.Chart(feature_imp,
                                     height=30*num_feats,
                                     width=800,
                                  title='Relative Variable Importance for Earnings post graduation').mark_bar().encode(
            x=alt.X('Importance', title="Relative Importance"),
            y=alt.Y('Feature', title="Variable", sort='-x'),
            tooltip=[alt.Tooltip('Importance:Q', title='Value'),
                     alt.Tooltip('Description:N', title='Description')]
        )
        _, rel_imp_col, _ = st.columns((1, 8, 1))
        with rel_imp_col:
            st.altair_chart(rel_imp_barchart)

    st.markdown(
        f"""<p style='text-align: left; color: #474c54'> The plot displays the <i>top {num_feats} factors</i> and their 
        corresponding importance. The most important factor to consider is <b>{feature_imp.iloc[0, 0]}</b>, which is 
        <b>{feature_imp.iloc[0, 2]}</b>. Hover over each bar to view the value and the variable's description."""
        , unsafe_allow_html=True)

    st.subheader('Variable Importance for individual universities')

    st.markdown(
        """<p style='text-align: left; color: #474c54'> The above importance values tell us what factors contribute to 
        a high income post graduation across all universities. To dig a little deeper, we will use <b>SHapley Additive
        Explanations.</b> Force plots show us exactly which variables had the most influence on the return on investment
        for a <b>particular university.</b>""",
        unsafe_allow_html=True)

    st.markdown(
        """<p style='text-align: left; color: #474c54'> For this analysis, please select a university you would like to
        look into and the variables for which you would like a description.""", unsafe_allow_html=True)

    with st.form('shap'):
        shap_col1, shap_col2 = st.columns(2)
        with shap_col1:
            # drop-down to select school
            shap_school_sel = st.selectbox('Type or select a school', roi.college_names)
            submitted = st.form_submit_button('View Force Plot')
        with shap_col2:
            # drop-down to view variable defs
            var_sel = st.multiselect('Type or select a variable to view its description', roi.dictionary['Name'],
                                     default='CDR3')
            st.table(roi.dictionary[roi.dictionary['Name'].isin(var_sel)].reset_index(drop=True))
        if submitted:
            expected_value, shap_values, X = roi.get_local_importance(shap_school_sel)
            shap_chart = shap.force_plot(expected_value, shap_values, X)
            st_shap(shap_chart)

    st.markdown("""<p style='text-align: left; color: black; font-size: 20px'> Interpreting the plot:
    <ul>
        <li>The axis scale represents the earnings value scale and the predicted value is in <b>bold</b> font</li>
        <li>The variables in <b>red</b> contribute positively to the earnings</li>
        <li>The variables in <b>blue</b> contribute negatively to the earnings</li>
        <li>The size of the block represents how large of a contribution the variable makes to the final predicted earnings</li>
        <li>The value under the block is in the same scale as the axis</li>
        <li>Mouse over smaller blocks to view the contribution that variable made to the predicted earnings</li>
    </ul>
    
    From this plot, for a particular university, we observe factors that influence the return on investment for students
    studying/looking to study there. 
    """, unsafe_allow_html=True)
