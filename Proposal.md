# Final Project Proposal

**GitHub Repo URL**: https://github.com/CMU-IDS-Fall-2022/final-project-roi

Team: 
- Aditya Nittala (snittala)
- Anjana Bytha (abytha)
- Brittany Redmond (bredmond)
- Devin Ulam (dulam)

Each year about 3.2 million students graduate from high school, of which 2 million continue their education going to college [1]. Shortlisting universities is often a cumbersome process which is limited by individual bias and timing constraints. Students need to consider various factors, including but not limited to university ranking, location, tuition cost, financial aid, campus life and safety. Evaluating the importance of these different factors to select a university could be tricky. 

For our analysis we will use the College Scorecard dataset which contains historical data for universities from the years 1996 - 2021 [2]. The data for each year consists of several university level factors and demographic variables of the student population at each university. In this project, we aim to address the issue of shortlisting universities by leveraging this dataset to answer the two following questions. 

**Question 1: What universities are best fit for me?**

Given a student profile based on factors such as location, annual income, age, preferred degree etc., we recommend the top 10 universities that are a best fit for the student. We define “best fit” as the universities the student has the best chance of getting into, and where they will thrive. Additionally, for each of the suggested universities, we present an analysis of various factors over the years. For example- how did the admission rate change over the years, how has the ratio of female vs male students changed, is there a difference in the age of students starting college. 

We believe that our application could provide students a better picture of where they stand in terms of academic requirements and student life expectations. The university recommendations could be a starting point, from which they can make further enquiries and choose schools that work best for them.

**Question 2: What university-level characteristics influence a student’s ROI?**

A student’s ROI is based on the time it takes to recoup their original financial investment toward tuition by earning a salary post-graduation. Using the College Scorecard dataset, we seek to answer the question: What university-level characteristics influence a student’s ROI?

The College Scorecard dataset includes historical metrics dating back to 1996, representing a multitude of factors that could influence the success of a university’s students. We will start by determining ROI for student records spanning between 1996-2020 using tuition, room & board, and expenses as a cost basis. Contrasting the cost basis with post-graduate salaries will provide us an estimated ROI. With this, we can explore the dataset to determine features that influence the variance of ROI across universities and fields of study.  This would include filtering out features such as household income, federal scholarships, student demographics, retention rates, etc. We hope to arrive at a set of features that can be used to predict a given university's average student ROI such that a university could tweak characteristics to optimize student outcomes. This analysis lends itself to a secondary outcome that provides the student a more formal evaluation to utilize in their university selection process. 

We acknowledge that the preliminary questions above are broad in scope. That said, we are likely to encounter more granular questions that further enhance the ultimate goal we are trying to achieve with this dataset: the evaluation of a student-university “best fit”. 

**References:**\
[1] 	The NCES Fast Facts Tool provides quick answers to many education questions (National Center for Education Statistics). National Center for Education Statistics (NCES) Home Page, a part of the U.S. Department of Education. (n.d.). Retrieved November 1, 2022, from https://nces.ed.gov/fastfacts/display.asp?id=51 \

[2] 	Data Home: College Scorecard. Data Home | College Scorecard. (n.d.). Retrieved November 1, 2022, from https://collegescorecard.ed.gov/data/ 




