
# Probability of Credit card Default

## Problem Statement

This business problem is a supervised learning example for a credit card company. The objective is to predict the probability of default (whether the customer will pay the
credit card bill or not) based on the variables provided. There are multiple variables on the credit card account, purchase and delinquency information which can be used in the modelling.
        
Credit card companies operate in a landscape where minimising risk and maximising
profitability is crucial. Then analysis of credit card default probability play is a pivotal
role in achieving this balance. By accurately assessing the likelihood of default,
companies can proactively manage potential losses, optimise credit limits and tailor
interest rates for individual customers. This analysis empowers companies to make
informed decisions, enhance customer relationships and maintain a healthier bottom
line. As a data analyst, delving into this realm provides an opportunity to contribute to
the industry’s stability and success.

## Steps followed :

##  1) EDA: Univariate Analysis
#### Distribution of data for categorical variables:
- 'Default' is the target variable. As we can see it is highly imbalanced with only 1288 defaulters (class 1) out of total 99979 credit card users, composing only 1% of the dataset.
- Such high data imbalance is not suitable for model building even though the domain is such that data imbalance is a given.

                       To balance the target class i.e. “Defaulters” we will use SMOTE, a resampling technique. 

- Moving forward, the credit card company should try to add more defaulters in coming future to this existing dataset so as to make it more balanced, as balanced datasets lead to more robust model building.

![target_var Medium](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/7d0ed0fe-78d3-4ac2-ab88-60dbdb47ab9e)


- Overall “Entertainment” is the most transacted merchant group comprising almost 50% of all transactions while “clothing & shoes” come at the second place with 17% contribution.
- Hence the credit card company should target users who have more number of transactions or high value transactions in the entertainment group.

![merc_grp](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/1b85ef5b-5b94-471a-b5c1-6e23db76a9da)

### Inferences from Univariate analysis:
- The dataset contained a lot of uncleaned data, there were 7 variables present which had more than 50% of data as missing while one variable had 29% data as missing. Remaining variables had around 11% of the data as missing, barring few exceptions.
- Due to this high degree of noise and unclean data, lot of variables were lost while cleaning the data and we were finally left with just 14 variables out of the original 36.
- It's worth noting that the second last row has all the values missing, hence we will look for rows with a lot of missing values and drop such rows since imputing large amounts of missing values row-wise will distort our dataset.
- Variable "acct_status" should only have 0 and 1 as values as per the data dictionary, but in the actual dataset it has 1,2,3 and 4 as values. We need to find the reasoning behind this nomenclature or else drop this variable too if the logic cannot be found.
- we will also drop such variables which are not required for our model building namely 'userid', 'name_in_email'.
- Almost all the continuous numeric variables have highly skewed boxplots, with only a few exceptions.

                  There are a lot of Outliers in almost all the variables. These need to be treated.   


## 2) EDA: Bivariate & Multivariate Analysis

- For checking the correlation between independent variables we will be using heatmap instead of scatter-plots, reason being heat-map are better at quantifying the correlation.
- Due to the dataset having large number of numerical variables, the pair-plot becomes illegible as it produces all the possible combination of correlated variables in one single visualisation.
- Due to this, we use heat-maps as our main tool to check correlation between the independent continuous variables as here we can identify the degree of correlation just by looking at colour of the individual boxes.

#### The plot combinations and inferences are shown below:
![bivar](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/4147fc8b-b127-4028-a51f-5a89ff80722a)

#### Inferences from Bivariate analysis:

**• Age vs default :** As we can see, mean age of defaulters is around 30 yrs whereas that of non defaulters is close to 40 yrs, meaning younger users in the age group of 20 to 30 yrs are more likely to default.

**• Average payment span in last 3 months vs default :** Here, on an average, defaulters have taken around 20 days to make the payment of their credit card bill while non defaulters on an average have taken around 15 days.

• Also there are a lot of outliers in this variable specially in case of non-defaulters, some taking as long as 80 days to pay the bill.

• This means that in general users who take more than 15 days to make payment for their credit card bills are more likely to default and should be prioritised over, with respect to enquiry calls and payment intimation mails.

• But still this cannot be taken too seriously due to the presence of outliers in non-defaulters.

**• Maximum bill amount payment in last 12 months vs default :** Here, we see that, there is not much difference between defaulters and non-defaulters. Except for the fact that there are very few outliers in case of defaulters and a lot of outliers in case of non-defaulters.

• One reason for this can be the fact that non defaulters may be more well to do and therefore are not afraid of spending more in contrast to defaulters.

**• Current payment status vs default :** Here, with respect to default, there is not much difference, as there is similar proportion of defaulters among both the users who have payed the current credit card bill and the users who haven’t.

• Hence, 'Maximum bill amount payed in last 12 months' as well as ‘Current payment status’ i.e whether the user has paid the current credit card bill, has almost no effect on whether the user will default or not.


### Heat map:
![heatmap](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/3e4e8267-2300-46bc-91a5-50b61c1e58d5)

#### Inferences from Multivariate analysis :

• As we can see in the heat map, there are a few variables that are highly correlated.

• For example, ’avg_payment_span_0_12m’ with ‘avg_payment_span_0_3m’ I.e. average payment span of last 12 months to average payment span of last 3 months.

• Another highly correlated pair is max paid invoice in last 12 months to max paid invoice in last 24 months.

• This shows that there is multicollinearity present in the data and thus we will have to reduce it during model building.

                            For this, we will use VIF(Variance Inflation Factor) technique.
• **VIF** is a statistical measure used in regression analysis to assess multicollinearity among independent variables. It measures the degree to which the accuracy of a regression model is affected when independent variables are correlated with each other.

## 3) Data Cleaning and Pre-processing :

#### A) Missing Value treatment

Visually inspecting the missing values in the dataset using heatmap (white bars show missing values)
![missing_val](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/f5acc995-7641-40e3-af06-ca1648fae49b)

• There are a lot of missing values in the dataset. Total missing values is around 20% of the whole dataset. Therefore we should be quite careful as to how we are going to treat these missing values.

• Among them we should drop the variables which have at least 30% of the column as missing values, as imputing such large number of missing values will lead to what we call as an Analyst bias.

• It refers to an approach related bias, since missing values imputation differ from analyst to analyst and there is no sure shot way to do it.

#### B) Outlier treatment
Checking the outliers per column using boxplot :

![outliers](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/e898a289-dcac-4642-9705-d0e021de8428)

All the variables have outliers except 'time_hours'.

• Some variables have one or two outliers while some have a large number of outliers.

                 Thus, we are going to use Inter-quartile-range (IQR) method for treating the outliers.
• All the outliers got treated using the IQR method, but some of the variables were found to have only zeroes left while some had only ones left after the outlier treatment. Therefore after dropping such variables, finally we were left with 14 variables.

#### Final boxplots after outlier treatment :

![final_boxplots](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/6e1ed83e-fbdd-4fe8-a290-c657ddb0b0b1)


## 4) Model Building :
#### A) Model Selection 
Our target variable “default” is a binary variable with 0 denoting non-defaulting credit card users and 1 denoting defaulting credit card users.

So this is essentially a classification problem where our main goal is to predict the users that will default. To solve this problem we will be using the following techniques or models:

**• Logistic Regression • Linear Discriminant Analysis (LDA) • Bagging (Bootstrap Aggregating) with Random Forest as base estimator • AdaBoost (Adaptive Boosting) • Gradient Boosting.**

#### B) Model Training
Here we utilise historical data to train and fine-tune the selected models.

First we need to split the data into training and testing sets, then train the models on the training set & finally use the test set to evaluate model performance. 

                                Train to Test data ratio was kept at 70:30.
• The original data I.e. the predictors and response are split into train and test data. Then we applied SMOTE to balance the training dataset.

                        New ratio of Non-defaulters to defaulters was set to 60:40

**SMOTE (Synthetic Minority Over-sampling Technique)** is an algorithm used in machine learning to address class imbalance by generating synthetic examples for the minority class, thereby increasing its representation in the dataset.

SMOTE is applied only on the training data and not on the testing data.
This is because we want to preserve the original data for testing our model and as mentioned above SMOTE introduces synthetic examples in our data which is okay to be trained on but not to test the data.

Checking the class proportion of target variable and applying SMOTE on train data:

![smote Small](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/795ca2c0-def3-451e-8635-971f2952b3f2)


#### C) Model Tuning
• For tuning the models thus built, we have used two techniques:

                ‣ Predicting classes based on their optimum threshold value than the default value of 0.5.
                ‣ Using GridSearchCV to optimise the hyper-parameters of the built models.
#### NOTE : All the models were built and their corresponding performance metricswere identified which are explained below in the model validation part.

#### Their corresponding classification reports, confusion matrices and ROC curves along with all the relevant charts can be found in the attached python Jupyter notebook (.ipynb file).

## 5) Model Validation
• To evaluate the performance of the above mentioned modelling techniques we generally use “Classification report” based performance metrics namely ‘Accuracy’, ‘Precision’ and ‘Recall’ along with ROC(Region of convergence) and AUC(Area under the curve) scores.

**• Since our goal is to predict the credit card users that will default, the performance metric ‘Recall’ becomes the most important metric for us.**

**• Reason being,** we are more concerned with correctly predicting the actual defaulters rather than the checking likelihood of these positive predictions being correct, which is measured by ‘Precision’.

• In business terms this means that we are okay even if some non-defaulters are predicted as defaulters as long as we are able to predict most of the actual defaulters.

• Another way to evaluate the performance of the above mentioned modelling techniques is by using Confusion Matrix and the two kinds of errors associated with it namely, Type I and Type II Errors.

**‣ Type I Error (aka False Positive) -** Occurs when the model incorrectly predicts a user as a defaulter, leading to the rejection of a potentially creditworthy customer.

**‣ Type II Error (aka False Negative) -** Occurs when the model fails to predict that a user will default, resulting in the approval of a high-risk customer who may default on payments.

**‣ Based on our business problem, out of the two, ‘Type II Error' is the one we must mitigate in order for our model to perform better.**

• Based on the steps mentioned earlier, all the respective models were built and tuned. The final selected version of each model along with their performance metrics are shown below:

![perf](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/64b7a503-45ad-4d55-b1cd-224c22479bc9)

![perform](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/204beffe-61fa-4640-9f7d-8d66e52fd97f)


#### Performance metrics detailed analysis :
**LDA and Logistic Regression** are the **best performing** models **with respect to Recall as well as type II error**, our target performance metrics. 

Also, **LDA model has a slight edge over Logistic regression in terms of recall value. Hence, we would choose LDA to be our final model**, just for the sake of moving ahead with one best model for detailed analysis and deployment.


## 6) Performance Metrics with respect to test data for the Best-Fit LDA Model

#### Classification report:

![clasifreport](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/276f0dbe-53ed-4ee2-b2f9-f518f65a86c1)

• For predicting Credit card Defaulters (Label 1 ):
‣ Recall (82%) – Out of all the credit card users who actually defaulted, 82% of
them have been predicted correctly .
‣ Precision (3%) – Out of every 100 predictions of defaulters, only 3 were
predicted correctly I.e. a lot of false positives got generated due to the highly
imbalanced nature of original data.
‣ Accuracy (67%) – 67 % of total predictions are correct.
• The selected model shows no overfitting or under-fitting with respect to recall
values.
26
#### ROC Curve and AUC Scores:

![auc](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/6698234f-de5f-4f1e-96c9-8e2c8c4026ea)

A good model will have an ROC curve that approaches the upper-left corner of the
plot. The closer the curve is to the upper-left corner, the better the model's
discriminatory power. The AUC value ranges from 0 to 1, where a higher AUC indicates
better model performance.
‣ AUC = 1: Perfect model, the ROC curve covers the entire area, and the model
makes perfect predictions.
‣ AUC = 0.5: Random model, the ROC curve is a diagonal line from the bottom-left
to the top-right, indicating no discriminatory power.

‣ In our case, for LDA model, AUC Scores = 0.8, which indicates that the model is
performing relatively better, it’s not perfect but not too bad also as the value is
nearer to 1 than to 0.5. The model is average when it comes to ROC curve.




#### Confusion Matrix:

![conf](https://github.com/VivekSagarSingh/Probability-of-Credit-card-Default/assets/153344691/1f257974-2be1-4416-8043-18a61ebbf953)

• Type I and Type II Errors: With respect to type I (False Positives) and type II (False
Negatives) errors, as mentioned earlier, our main focus is to minimise type II Errors.
‣ Looking at the type II Errors, the model with the least type II error is LDA (70).
‣ This means that out of a sample of 386 actual defaulters in the test data , our
best-fit model failed to identify 70 defaulters, which comes around 18%. This is
the best we could do across all the models built.


## 7) Insights from the final (most optimum) model i.e. LDA Model
• Looking at our most optimum model i.e. tuned LDA model, the top 3 predictors based on the Linear discriminant function that play a major role in predicting classes are as follows (Linear discriminant function is a linear combination of the input features that maximises the separation between classes):

**‣ ’num_active_inv’(0.65) :** This represents Total number of active invoices per user, i.e. Unpaid bills. It is positively correlated to predicting classes meaning, based on analysis we can say that **users with large number of active invoices are 60% more likely to belong to class 1(defaulters). Hence, this becomes our most important predictor.**

**‣ ‘status_max_archived_0_6_months’(0.27) :** This represents maximum number of times the account was in archived status in the last six months. It is also positively correlated to predicting classes meaning, based on analysis we can say that **users with higher number of archived status counts in last 6 months are around 25% more likely to belong to class 1(defaulters).**

**‣ ‘num_arch_ok_12_24m’(-0.25) :** This represents number of archived purchases
that were paid between 24 months in the past to the present date and 12 months
in the past to the present date. But this variable is negatively correlated to
predicting classes meaning, based on analysis we can say that **users with higher
number of archived purchases in the given period are 25% less likely to belong
to class 1 (defaulters). And it makes sense also, since these users are clearing
their pending bills thus expressing their willingness to continue using their
credit cards.**

• Also looking at the performance evaluation of the LDA model, we can see that there
is a high variation between the train and test ‘Precision’ values. This doesnot
necessarily mean that there is overfitting.

• This variation is due to the fact that we have used SMOTE to resample the training
data in order to balance it. This was done in order to properly train the machine
learning models, as original dataset had too few positive classes to properly train
the models.

**• There is also a high imbalance between the precision and recall values in the test
data** with precision values being quite low. Ideally we want a good balance between
recall and precision, but that is not the case here.

**• The main reason for such a low precision value is the highly imbalanced nature of
the dataset** (only 1% of the total users i.e. 1000 users amongst almost 1 lakh users
were defaulters while the rest were non-defaulters.

• The extremely low precision value suggests that the model is producing a large
number of false positives for each true positive it identifies. While this may seem
rather unusual, it is not uncommon in certain situations such as highly imbalanced
datasets or when the priority is maximising recall.

• Moreover another reason for this high imbalanced nature of dataset is the domain to
which it belongs i.e. Credit card default prediction or more broadly financial risk
assessment. Since finance is the base for almost all economic activities in the world,
financial risk assessments are taken very seriously due to which the ratio of
defaulters to non-defaulters is usually quite low leading to such high imbalances.



## 8) Model Implication on the business and recommendations
**• Since ‘Total number of active invoices per user’ is our most important predictor**, the
credit card company should device specific plans around this variable.

**★ One recommendation here would be to clusters of users based on number of
active invoices. It can offer lucrative discounts and offers to users who have a
lot of active invoices encouraging them to settle their bills.**

**★ Company can also cross match such customers to the merchant groups they
transacted the most and promise lucrative offers for related products on the
condition that they settle their active invoices in a limited time period.**

• Another important predictor is ‘maximum number of times the account was in
archived status in the last six months’.

**★ Here the recommendation would be to allocate dedicated human resource
and time to setup direct communication line with customers who are still not
written off yet whose accounts went in archived status most number of times
in last 6 months.**

• Our third most important predictor is ‘number of archived purchases that were paid
between 24 months in the past to the present date and 12 months in the past to the
present date’.

**★ Here the company can provide discount coupons related to user’s most
transacted merchant groups for users with higher number of archived
purchase payments aesthete users settled their archived purchases on their
own accord, meaning they are willing to stay with the credit card company.
Providing lucrative offers to such users will encourage them to engage more
in transactions through the company’s credit card.
30**

• Another main highlight of our model is the high variation between Precision and
Recall value and what it means for the business.

• In the context of a credit card company, it can be particularly significant due to
the unique nature of the business and its focus on risk management and customer
satisfaction. 

**Some implications are mentioned below :**

**Operational Costs:**

• High False Positives: Low precision means a high rate of false positives, where the
model incorrectly predicts credit card defaults. This is also clearly visible in our
final model.

• Investigating these false positives can be costly in terms of resources, time, and
manpower.

**★ One major recommendation for the business here would be, conducting
manual reviews of flagged accounts.** This is absolutely essential in order to
keep track and make sense out of the business.

**Customer Experience :**

• Customer Inconvenience: Customers who are subject to false positives may
experience inconvenience, such as temporary card suspensions, transaction
declines, or account freezes. This can lead to customer dissatisfaction and
potentially damage the company's reputation.

**★ Hence, here the recommendation would be to allocate dedicated human
resource and time to mitigate such issues as soon as they occur.**

**Risk Management :**

• High recall indicates that the model captures most of the actual
defaults, this can help the company mitigate credit risk. Identifying a large portion
of true defaults is essential for minimising financial losses.

**Regulatory Compliance :**

• Credit card companies are subject to various regulatory requirements and
standards. High recall can be essential to ensure compliance with regulatory
standards related to risk assessment and fraud prevention.

**Financial Impact :**

• Loss Mitigation: While high recall helps mitigate financial losses from defaults, low
precision can lead to financial losses due to the approval of high-risk customers
who eventually default.

• Here the recommendation would be to try to strike a balance between precision
and recall based on its risk tolerance and business objectives. Adjusting the
prediction threshold can help achieve a practical trade-off between the two
metrics.

• This can not be achieved to a greater extent using this dataset, as one way to
achieve this balance is by optimising threshold values which has already been
done.

**★ So here the recommendation would be that the company use this model to
operate for now and try to add to this dataset more defaulters, so that
overtime the iterative models can strike a better balance between precision
and recall values.**

**Continuous Monitoring and Improvement :**

• Feedback and Adaptation: Continuous monitoring of model performance,
feedback collection, and adaptation are crucial for optimising the model's
performance over time as mentioned above.

**★ Hence, here the recommendation would be that the company should be
prepared to make adjustments based on evolving conditions and business
priorities.**

















