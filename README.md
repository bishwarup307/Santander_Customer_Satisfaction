# Santander_Customer_Satisfaction
Find out who are the happy customers for the bank by analyzing their wide transaction data in different product lines.

Project: Kaggle: Santander Customer Satisfaction (* the biggest competition in Kaggle to-date in terms of number of participants)

Objective: Given a wide range of transaction data across different product lines of the bank, we were to find out the customers
who are likely to be unsatisfied with the bank's service. The inherent problem was that, the training instances contained a 
number of cases where the customer were not happy with the bank's service but also they never bothered to lodge a complaint and
that made training the model difficult due to shortage of sufficient signal. Also, the pretty small size of the data came with 
a huge amount of noise which was a challange to deal with. We explored a number of different methods for feature selection and 
also trained different models keeping a close eye on the cross-validation score for each of them. Finally, we ranked 4th out of
a total of 5123 participating teams.

Time Frame: March, 2016 - May, 2016

Evaluation Metric: Area Under the ROC curve

Team - Bishwarup Bhattacharjee - Branden Murray - Mohamed Bibimoune

Total Participating Teams: 5123

Final Standing: 4th

Maximum AUC achieved: 0.82847

________________________________________________________________________________________________

Models used:

Elastic Net (R)

Xgboost (R/ python)

RandomForest(R/ python/ h2o)

ExtraTrees (python)

Suppor Vector Machine (Linear Kernel) (R)

K-nearest neighbours (R/ python)
