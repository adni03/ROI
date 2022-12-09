# Final Project Report

**Project URL**: https://cmu-ids-fall-2022-final-project-roi-collegematcher-ngipmw.streamlit.app/

**Video URL**: https://www.youtube.com/watch?v=57Zp_kgilxs

 ----

## Abstract
University shortlisting is a difficult process often limited by individual bias and time constraints. Students have access to several websites which shortlist universities but these are many times at scale and not as personalized to the individual. Receiving a generic list of recommendations is not quite helpful to students who are interested in understanding why a specific university is the best fit for them. Our application aims at answering this question. We have three sections in the application- university recommendations, exploring universities and a return on investment analyzer. We make recommendations to students by taking certain filters such as SAT score, ACT score, region, etc and use them to suggest the top 10 universities that are a best fit for the individual. In addition, our dashboard also provides a comparative analysis of selected universities through interactive bar charts that helps them gain a better understanding of the gender ratios and expenditures at the selected universities. The explore universities section allows them to gain a deeper understanding of selected universities by providing different statistics and visualizations of proportion of degrees, races and gender at a selected university. The Return on Investment section is the most interesting part of the project which gives a list of factors that influence the return on investment at a global level and then breaks it down even at the university level. In addition to giving an insight into what factors influence the ROI, the application also provides an intuitive force plot that show how much each of the factors is responsible for increasing or decreasing the return on investment. 

## Introduction
Each year about 3.2 million students graduate from high school, of which 2 million continue their education going to college [1]. Shortlisting universities is often a cumbersome process which is limited by individual bias and timing constraints. Students need to consider various factors, including but not limited to university ranking, location, tuition cost, financial aid, campus life and safety. Evaluating the importance of these different factors to select a university could be tricky.

According to NCES, there are 3,982 institutions in the United States, out of which 1,625 are public four-year and two-year colleges; 1,660 private non-profit four-year and two-year schools; and 657 for-profit schools [2]. The sheer volume of the colleges makes it impossible to review the academic and financial requirements, and the culture to find a university that best suits a student. A college shortlist is a list of approximately six to ten colleges that helps a student to streamline their application process. Creating a college shortlist helps them organize a list of potential schools, encourages them to narrow down their choices, and prompts them to identify schools to further communicate with.

Most students attend college in order to get a better job with a higher salary. But the financial returns to college vary widely depending on the institution a student attends and the subject they study. While prospective students often ask themselves whether college is worth it, the more important questions are how they can make college worth it and what factors affect the return on their investment. A student’s return on investment is based on the time it takes to recoup their original financial investment toward tuition by earning a salary post-graduation. Factors that influence this return on investment are not set in stone and require analysis to identify. These factors influence the university shortlists, making the whole process much more arduous. 

Our application provides students with a dashboard that helps personalize their college shortlists based on their profile, by recommending a list of schools “best fit” for them. This recommendation is based on their preferences for region, funding type, tuition range and academic scores. We also produce a dashboard that details the academic and financial summaries for a university of their choice and display various visualizations describing the demographics of that school. Lastly, we enable the student to dig deeper into the factors they should consider to maximize the return on their investment by illustrating the influence features like tuition cost have on their post-graduation income. Through the analysis in our application, we provide answers to the following questions:
* What universities are best fit for me?
* What university-level characteristics influence my ROI?

To produce these insights, we use the [College Scorecard](https://collegescorecard.ed.gov/data/) dataset which contains historical data for universities from the years 1996-2021 [3]. The data for each year consists of several university level factors and demographic variables of the student population at the university. 

## Related Work
Several college recommender websites like College Board's BigFuture, Cappex, College Search from The Princeton Review, etc. have gained popularity among students. These engines provide tools like Best Fit estimation, college comparisons, and scholarship finders, but are hidden behind pay-walls. Shortlisting universities is a taxing task on its own, and the variety of recommender engines adds further complexity to the process. Furthermore, most websites do not talk about return on investment and instead, focus only on providing a comprehensive list of colleges.

While an extensive list of universities is not a demerit, the amount of information displayed can easily overwhelm the student. More often than not, after the first ten universities, the recommendations are poorly matched and become more about quantity over quality. We serve as a springboard for refining university shortlists by being concise and producing only the top ten universities that best fit the student’s profile. 

Another important component of our application is the ROI analysis. The Center on Education and the Workforce at Georgetown University produced a list of best universities in terms of return on investment [4] but the data is unintuitive to view and the report does not list all the universities in the College Scorecard dataset. We provide visualizations that explain factors that affect post-graduation earnings across all universities and enable the student to understand what impacts the return on investment on an individual university level as well.

The explanations are generated by training a model that predicts the post-graduation earnings based on university level factors as well as student information. Through the interactive visualizations, we have identified a list of ten features that play a major role in dictating the potential earnings of a student at a given university.

## Methods

### Data Preprocessing
#### Background:
The US Department of Education compiled information about various universities in the United States by combining information from the student financial aid system and federal tax returns in an effort to make educationcal investments less speculative. The result was the College Scorecard dataset. There are approximately 3000 features in this dataset, and are broadly categorized into:
* `SCHOOL`: Location, programs offered, faculty, etc.
* `STUDENT`: Average age, family income, etc.
* `COST`: Cost of attendance, room and board, etc.
* `AID`: Debt, average loan balances, etc.
* `EARNINGS`: Earnings after 1, 3, 6, 8; 10 years, income by gender, race, etc.
* `REPAYMENT`: 1, 3, 5 year repayment rates by family income; percent defaulters, etc.
* `ADMISSIONS`: SAT scores, ACT scores, etc.
* `ACADEMICS`: Degrees awarded by program in percentage, average time for completion, etc.
* `COMPLETION`: Completion rate according to family income bracket, gender, etc.

#### Categories/Features Selected:
We filtered out a subset of the features available for our use case. This selection was done in two ways. First, we looked at the comprehensive data dictionary provided by the dataset owners and compiled a list of features that at first glance that seemed relevant and interesting to explore. These features belonged to the first eight categories in the above list. 

On performing exploratory analysis, we discovered that the data was quite messy and rather incomplete, with over half of the rows missing important data in some cases. To expand the set of working features, we looped through the features in the categories and identified ones with at most 20% missing data and added them to our working dataset. To compensate for the missing data in a few crucial features, we incorporated an additional source of data obtained from [GitHub](https://github.com/lux-org/lux-datasets/blob/master/data/college.csv). We also added latitude and longitude information for each of the universities obtained from a dataset that mapped ZIP codes to their corresponding values. With this, our tally of features came up to 74.

### Recommender:
Our application uses machine learning techniques such as clustering and regression to produce ten universities that best suit the needs of the user. Broadly, the steps listed below are followed to provide a personalized list, based on which visualizations are created and displayed.
- Data Filtering
- Modeling
- Score calculation

#### Data Filtering:
University recommendations are very sensitive to the preferences of the student and their academic profile. To take into consideration their choices, we ask the user to provide us their SAT and ACT scores, the types of university they are looking for, their regions of interest and the tuition range they are comfortable with. With these parameters, we filter out the data to obtain a list of universities that best match their criteria. We realize that having the flexibility with these input parameters is quite important to the stakeholder which in our case are students.  

#### Modeling:
The overarching approach we took was to cluster the universities into three groups based on scores and recommended universities from a cluster that best matched the student’s profile. The reasoning behind choosing three clusters was to segregate the universities into how difficult it would be for the student to get accepted into: Easy, Medium and Hard. Given the student’s academic test scores, they will be assigned to a cluster and their recommendations come from universities that belong to this cluster.

The algorithm of choice for this clustering is K-Means Clustering [5]. Given the filtered data, we train a k-means model and obtain the cluster labels for each of the universities. We then run the model on the scores submitted by the user to obtain a cluster assignment. This assignment is then used to identify the universities where students’ profile is most similar to the user. 

Once the list of most similar universities is obtained, we then calculate the euclidean distance between the user’s score and the average student’s scores in a university from this list. The recommendation is made up of ten universities with the smallest such distance.

A nuance here is that from our analysis we observed that the SAT and ACT scores are correlataed and follow a linear relationship. Hence, we trained a simple linear regression model that converts the SAT score which ranges between 400 and 1600 to the range followed by ACT scores, which is 1 to 36. The distance is calculated after this conversion.

#### Best Fit Score:
Intuitively, the higher the score, the better chance a student has to get accepted to a university. We calculate the Best Fit Score from the distances obtained from the model as follows:

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/equation.png?raw=true"/>
</p>

Where, 
- `n`: top-n universities
- `d`: distance calculated in the above section
- `i`: i-th university

We order the universities on this score and present it to the user in the dashboard in the form of a table and a barplot showing the score.

### Return on Investment Analyzer:
Our feature set consists of 74 columns that talk about demographics like gender and race, the percentage of degrees a university awarded program wise, admission rate, etc. To understand how these factors influence the post-graduation earnings, we trained a machine learning model - RandomForest [6] to model the relationship between these features and the earnings column.

We then used the calculated feature importance from within the Sci-kit learn implementation of the algorithm and SHAP (SHapley Additive exPlanations) [7] to produce insights about the features that have a significant effect on the predicted income. The following steps were followed:
- Modeling
- Interpretation

#### Interpretability:
An important component of our application is the analysis we provide on ROI. This links well to interpretability in machine learning. SHAP is a method that explains how individual predictions are made by a machine learning algorithm. This technique deconstructs a prediction into a sum of contributions from each of the model’s input variables [8]. 

The output of SHAP is a matrix of numbers that has the same dimensions as the input data, where these numbers are SHAP values. The output of the model can be reproduced as the sum of these SHAP values and a base value. This base value is the mean of the model output across the dataset. Crucially, for our application, SHAP gives us how much contribution each feature made to prediction the post-graduation income on a university level bases. For a more general overview, the `feature_importance_` attributed provided by Sci-kit learn enables us to identify factors across all universities.

## Results
The Application has three sections- best fit university recommendations based on user input for certain features, exploring different universities and analyzing the return on investment for universities. The application has been separated into these sections to allow for the users to have an immersive experience for each topic of interest. 

### University Matchmaker 
The University Matchmaker has a section to take input from the user - SAT Score, ACT Score, Funding Model, Region and Tuition Cost to make recommendations based on the user’s choices. The Funding Model and Region allow for multiple selections to take into consideration the varied choice of every student. 

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/rec_dashboard.gif?raw=true"/>
</p>

#### Visualizations:
Once the user enters the parameters of interest and clicks on the button, the recommendations are displayed based on the best fit score calculations. We display a list of the universities alongside a US Map which has location pins with the recommended universities. The primary motivation for the design of the visualizations is to provide the user with an intuitive understanding of the results. By displaying the list of universities with the tuition vs earnings, it gives the student a snapshot of the return on investment. 

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/map_table.png?raw=true"/>
</p>

This visualization is followed by a bar chart showing the best fit scores for the recommended universities in a descending order focussing on the ones which are of best interest to the student based on their selections. Hovering over each of the bars gives a glimpse of the important details such as the average SAT score, median ACT score, average cost and admission rate for the specific university. In addition, we have a multi-bar chart which shows the comparison of the number of male and female students at the university. Analyzing the gender ratios at educational institutions is an important factor that determines the college selection for students. We have a stacked bar chart that shows the total expenditure for the university, the stack values show the tuition cost and the expenditure at the university. As the user selects the bars in the best fit university chart, the other bar charts update accordingly to reflect the metrics for the selected universities. Reviewing such metrics in a comparative manner could be quite useful to the curious high schooler trying to compare different universities of interest. 

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/compare_unis.gif?raw=true"/>
</p>

We have a section that gives the university metrics for the best fit university to give an overall idea about the university. 

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/about_best_uni.png?raw=true"/>
</p>

### Explore Universities 

The Explore Universities is a section which allows the users to get a deeper understanding of the universities. A specific university can be selected from a dropdown of the complete list of available universities. On selecting a specific university, the first section shows a detailed view of several parameters such as - admission rate, average SAT score, cost of attendance, expenditure, median earnings and others to give the student a better idea about the selected university.

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/explore_uni.gif?raw=true"/>
</p>

#### Visualizations:
The visualizations in this section include a bar chart that shows the top 10 degrees by proportion awarded at the institution showing what kind of degrees are awarded the most at a particular institute. This is useful to users who want to explore universities that have specializations in certain domains. We also show a pie chart displaying the percentages of men and women at the university. This is followed by a bar chart showing the distribution of races at the selected university. The intuition behind using bar charts is that they are the simplest and most easily interpretable forms of visualization. When looking at the proportions of degrees or the race distribution, the user gets a clear picture of the bird’s eye view without much effort or unraveling.

<img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/degree.png?raw=true" width="680"/> <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/gender.png?raw=true" width="320"/> 

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/race.png?raw=true" width="675"/>
</p>

### Return on Investment Section 
The Return on Investment section is the most interesting part in our application. In addition to providing best fit recommendations to the students, we analyze the return on investment for universities. There are several factors which determine the return on investment for a specific university and in this section, we delve deeper into understanding how these factors impact it. Using machine learning, we generate a list of factors that contribute to higher median earnings. 

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/global_imp.gif?raw=true"/>
</p>

#### Visualizations:
The factors that influence ROI are displayed in a bar chart that shows the relative importance of each factor. The user can use a slider to change the number of factors which then updates the chart to show the selected number of factors. Understanding the relative importance of each of these factors provides important information to the students shortlisting colleges. The bar chart enables the student to understand how important a specific factor is and the weight they should give when they look for universities. We believe that by bringing to light the importance of factors seldom considered in university shortlisting, it opens up a new realm of possible university contenders. 

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/global_imp.png?raw=true"/>
</p>

The factors influencing ROI at a specific university change based on the university. To help students understand these factors better, we provide the user with a force plot. Here, they can select a specific university from the dropdown which contains a list of all the universities. In addition, there is a multi-select drop down to select different feature names which provide a description for selected features. Understanding the meaning of the features is an essential part in trying to uncover the impact of these features on the return on investment. 

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/shap_1.gif?raw=true"/>
</p>

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/shap_2.gif?raw=true"/>
</p>

The visualization consists of a bar which shows how the features affect the median earnings (ROI), the features in red push the value to right making it higher, hence contributing to a better ROI where as the features in blue push the value to the left making it lower, hence driving a lesser ROI. It is interesting to note that for different universities, the factors that increase or decrease the ROI keep changing. On hovering over each of these features, it displays the proportion of relative impact that feature has in increasing or decreasing the ROI value. 

<p align="center">
  <img src="https://github.com/CMU-IDS-Fall-2022/final-project-roi/blob/main/pics/force_plot.png?raw=true"/>
</p>

## Discussion
The process of applying to universities is a laborious process that involves multiple time consuming steps of which the first is shortlisting colleges. Students and their parents have to carefully consider a variety of factors such as university ranking, location, tuition cost, financial aid, campus life, safety, etc. while looking for universities. This task is a large undertaking considering the number of colleges in the United States. With our project, we aimed to decrease the effort required to shortlist institutions by providing personalized recommendations, university level and demographic information, and a comprehensive analysis of the factors that a student should consider to maximize their return on investment.

Our recommendations dashboard solves one half of the problem by listing universities that are best fit for a student. We provide information on the best fit college and enable the user to compare the top recommended universities. Next, by analyzing the SHAP values and feature importance values from the RandomForest model, we generated a few insights. Firstly, in a more general sense, the percentage of degrees awarded in Engineering, the tuition cost, the percentage of degrees awarded in Healthcase for a university greatly influence the post-graduation income. It is interesting to note that the student’s family income also plays a part in the final return on investment. Looking at each university, we produced an illustration that displays how different factors (which are unique to each university to a certain degree) increase and decrease the post-graduation income.

## Future Work
The College Scorecard dataset is an exhaustive dataset that contains a whole lot of information. The sheer size of the dataset poses a problem when it comes to preprocessing. The first step in extending this application is to include student retention, demographics, financial and academic data from multiple years. Due to time and resource constraints, we were only able to preprocess data from 2019. In the future, we will be able to provide richer insights into universities.

Next, a by product of deeper data cleaning is that we can utilize the full range of data available (from 1996 till present) to develop a more robust recommendation algorithm that is trained on historical data. In its current form, the algorithm makes predictions based on one year’s worth of data, limiting the rank accuracy. 

Lastly, we will be able to provide an over-the-years analysis for both university level features and demographics, describing how factors like gender make up, tuition fees, admission rates, etc. have changed for a particular university over time. 

Since most college recommender websites base their lists off of the College Scorecard dataset, we believe that our application has the potential to help students curate a list of universities that truly suit them best without the pay-walls.

## References:
[1] The NCES Fast Facts Tool provides quick answers to many education questions (National Center for Education Statistics). National Center for Education Statistics (NCES) Home Page, a part of the U.S. Department of Education. (n.d.). Retrieved November 1, 2022, from https://nces.ed.gov/fastfacts/display.asp?id=51

[2] Digest of Education Statistics, 2020. National Center for Education Statistics (NCES) Home Page, a part of the U.S. Department of Education. (n.d.). Retrieved December 7, 2022, from https://nces.ed.gov/programs/digest/d20/tables/dt20_317.10.asp?current=yes 

[3] Data Home: College Scorecard. Data Home | College Scorecard. (n.d.). Retrieved December 7, 2022, from https://collegescorecard.ed.gov/data/

[4] Ranking 4,500 Colleges by ROI (2022) - CEW Georgetown. (n.d.). CEW Georgetown. Retrieved December 7, 2022, from https://cew.georgetown.edu/cew-reports/roi2022/

[5] MacQueen, J. (1967) Some Methods for Classification and Analysis of Multivariate Observations. Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability, 1, 281-297.
