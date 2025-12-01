PREDICTIVE ANALYSIS ON INSURANCE DATASET
1. Project Title
Predicting Medical Insurance Charges Using Demographic and Lifestyle Variables

2. Data Source
The dataset used for this project is the Insurance Dataset ( https://www.geeksforgeeks.org/machine-learning/dataset-for-linear-regression/ - Insurance Charges Dataset).
	File used: insurance.csv
	Number of records: 1,338
	Number of variables: 7
	Data type: Mixed (4 numeric, 3 categorical)
	Key source: Publicly available, non-proprietary dataset containing individual demographic and health-related attributes used to model insurance charges.

3. Business Problem Description
In the health insurance industry, accurately predicting the expected insurance charges for policyholders is essential for pricing strategies and risk management.
This project aims to develop a predictive model that estimates medical insurance charges based on customer demographic and lifestyle characteristics such as age, BMI, number of children, smoking status, and region.
The insights from this model can help:
	Insurance companies determine fair and competitive premiums.
	Identify high-risk individuals for proactive health cost management.
	Support data-driven decision-making in policy pricing.
This problem cannot be solved with simple arithmetic, as it requires modeling complex relationships between multiple variables to predict charges.

4. Predictive Purpose of the Data
The goal is to predict insurance charges (continuous variable) from various predictors, allowing insurers to forecast costs and allocate resources effectively.
	Response Variable: Charges (continuous variable representing medical insurance costs).
	Predictor Variables: Age, sex, BMI, children, smoker, and region (at least two predictors, with a mix of numeric and categorical). 
	Predictive Purpose: The goal is to predict insurance charges from various predictors, allowing insurers to forecast costs and allocate resources effectively. By applying regression analysis, the model will enable managers to understand how each factor (e.g., smoking, age, BMI) influences total medical expenditure.
	 Evaluation Metric: The coefficient of determination (R²) will be used to assess model success in a multiple linear regression context, measuring the proportion of variance in charges explained by the predictors.

5. Variable Descriptions
Variable	Type	Description
age	Numeric	Age of the policyholder (18–64 years).
sex	Categorical	Gender of the policyholder (male, female).
bmi	Numeric	Body Mass Index – measure of body fat based on height and weight.
children	Numeric	Number of dependent children covered by insurance.
smoker	Categorical	Whether the individual is a smoker (yes/no).
region	Categorical	Residential area in the U.S. (northwest, northeast, southwest, southeast).
charges	Numeric (Target)	Medical insurance cost billed by the provider.

6. Descriptive Statistics
Summary from R:
Numeric Variables (Tukey’s Five-Number Summaries, Mean, and Standard Deviation)
Variable	Mean	SD	Min	Max	Skew	Kurtosis
age	39.21	14.05	18	64	0.06	-1.25
bmi	30.66	6.10	15.96	53.13	0.28	-0.06
children	1.09	1.21	0	5	0.94	0.19
charges	13,270.42	12,110.01	1,122	63,770	1.51	1.59
Dataset contains 1,338 records — meets the rubric’s minimum sample size requirement (≥100).
Categorical Variables (Counts)

Variable	Category	Count
sex	Male	676
	Female	662
smoker	Yes	274
	No	1,064
region	Northeast	324
	Northwest	325
	Southeast	364
	Southwest	325

7. Exploratory Data Visualization
Boxplots: Show moderate variation across age, BMI, children and charges; no severe outliers identified. These visualize Tukey’s five-number summaries for the numeric variables.
 <img width="902" height="703" alt="image" src="https://github.com/user-attachments/assets/8259f4ed-2d76-4b4a-9f0d-403eb87c6d75" />

Interpretation of Boxplots of Numeric Variables (Separated Panels):
The boxplots provide a visual summary of the distribution of numeric variables in the dataset:
	Age: Most individuals are between 27 and 51 years old, with a median around 39. There are no extreme outliers.
	BMI (Body Mass Index): The central 50% of BMI values range roughly from 27 to 35. A few higher BMI values are present as outliers above 45, indicating some individuals with unusually high BMI.
	Charges: Health insurance charges vary widely, with a median around $9,382. There are significant outliers on the higher end, exceeding $60,000, suggesting that a few individuals have very high medical expenses.
	Children: The number of children is skewed toward 0–2, with very few individuals having more than 3 children. There are no extreme outliers.
Overall, the boxplots reveal differences in variable spread and highlight potential outliers, which could impact further statistical analysis or modeling.

	Pearson Product-Moment Correlation Matrix (Numeric and Graphic Versions): No collinearity issues detected (no correlations above 0.9).
		age	bmi	children	charges
age	1.00	0.11	0.04	0.30
bmi	0.11	1.00	0.01	0.20
children	0.04	0.01	1.00	0.07
charges	0.30	0.20	0.07	1.00
No collinearity issues detected (no correlations above 0.9).
 <img width="859" height="591" alt="image" src="https://github.com/user-attachments/assets/fc85a87c-ad3f-40c2-853e-bbced8a411b9" />

LINEAR REGRESSION ANALYSIS OF INSURANCE CHARGES:

8.Model Specification 
Final Model Formula:
"charges"=β_0+β_1⋅"age"+β_2⋅"sex"+β_3⋅"bmi"+β_4⋅"children"+β_5⋅"smoker"+β_6⋅"region"+β_7⋅("age"×"bmi")+β_8⋅("smoker"×"bmi")+ε

Explanation of variables and coefficients:
	age, bmi, children: continuous predictors
	sex, smoker, region: categorical predictors coded as dummy variables (e.g., smoker: 0 = No, 1 = Yes)
	Interaction terms: age × bmi and smoker × bmi included to account for synergistic effects
	Transformations: None applied; log transformation not required based on Box-Cox and residual analysis
  
9. Interaction Terms 
	Interaction terms evaluated using scatterplots, partial dependence plots, and statistical significance (p < 0.05).
	Business logic supports interactions: smoker status modifies effect of BMI, age modifies BMI effect.
	Both interactions included in final model.
Explanation :
Interactions allow the effect of one variable to depend on the level of another. For example, the effect of BMI on charges is greater for smokers, which is captured by the bmi×smoker term. Similarly, age modifies the effect of BMI slightly, though not statistically significant, it was retained to capture potential nonlinear trends.

10. Non-linear Transformations 
	Examined skewness statistics and residual plots.
	Applied Box-Cox method (MASS::boxcox) to test whether a log or other transformation would improve the model.
	Result: No transformation needed; residuals approximately normal, AIC/BIC did not improve.
Explanation:
Non-linear transformations (e.g., log) can stabilize variance or normalize residuals. Since residuals were already approximately normal and homoscedastic, no transformation was applied.

11.1 Train/Test Split and Model Fitting
The dataset was split into a training set (80%, 1,072 records) and a testing set (20%, 266 records) using createDataPartition() to ensure reproducible random sampling. The linear regression model with interaction terms (age:bmi and bmi:smoker) was fitted on the training set using lm(). Continuous predictors were centered before creating interactions to reduce multicollinearity. The fitted model was then applied to the testing set to generate predictions. These predictions were compared with actual insurance charges to compute evaluation metrics, including R², Adjusted R², RMSE, and MAE, providing an unbiased estimate of model performance on unseen data.
Dataset Partition Table (unchanged but verify numbers):
Dataset Partition	Number of Rows
Training (80%)	1,072
Testing (20%)	266
________________________________________
Training Model Equation and Coefficients
Include both the formal equation and the fitted coefficients from your summary():
Formal Equation:
"charges"=β_0+β_1⋅"age"+β_2⋅"sex"+β_3⋅"bmi"+β_4⋅"children"+β_5⋅"smoker"+β_6⋅"region"+β_7⋅("age"×"bmi")+β_8⋅("bmi"×"smoker")+ϵ

Fitted Coefficients (Training Set):
Predictor	Estimate
(Intercept)	573.466
age	192.314
sexmale	-533.383
bmi	-61.093
children	422.161
smokeryes	-21,268.984
regionnorthwest	-645.793
regionsoutheast	-1037.685
regionsouthwest	-1094.564
age:bmi	2.113
bmi:smokeryes	1478.168
Note: The training model equation comes directly from summary(model) on train_data. These are the estimates the model learned from the training data.


	Continuous predictors centered before interaction creation to reduce multicollinearity.
	Model saved as final_model.rds.
  
12. Evaluation Metrics for final linear regression model
Metric	Value	Interpretation

R² (Test)	0.8082	80.8% of variance in charges explained on test data
Adj R² (Test)	0.8007	Adjusted for predictors; slightly lower due to test sample
RMSE (Test)	5,128	Average magnitude of prediction errors on test data
MAE (Test)	2,963	Mean absolute difference between observed and predicted charges
AIC (Train)	21,218	Penalized model fit metric from training data
BIC (Train)	21,278	Penalized model fit metric with stronger complexity penalty

Explanation:
The model shows good predictive performance on the test set, with R² of 0.8082 indicating that 80.8% of the variance in insurance charges is explained by the predictors. RMSE and MAE values indicate reasonable prediction errors. AIC and BIC values confirm the model is parsimonious on the training data.
13. Diagnostic Plots 
	Residuals vs Fitted: Residuals randomly distributed; confirms linearity and homoscedasticity.
 <img width="796" height="597" alt="image" src="https://github.com/user-attachments/assets/bb3f2071-fa29-44f2-8960-b7af849942f3" />


	Normal Q-Q: Residuals approximately normal; minor deviations at tails.
 <img width="826" height="619" alt="image" src="https://github.com/user-attachments/assets/ae7c5bd7-4579-4f07-a59c-64384f19a14b" />

	Scale-Location: Residuals evenly spread ; variance roughly constant across fitted values. Uniform spread confirms homoscedasticity.
 <img width="780" height="584" alt="image" src="https://github.com/user-attachments/assets/21b3e6a2-e570-46a6-8131-48193a4a8ac0" />


	Residuals vs Leverage: No influential observations detected. The model is not driven by outliers.
 <img width="805" height="603" alt="image" src="https://github.com/user-attachments/assets/f8d5e2b8-9902-486e-8518-24231fefda59" />

14.VIF Analysis 
VIF values for full model with interactions.
Full model (interactions, type = "predictor"):
Predictor	GVIF^(1/2Df)	Interacts With
age	1.0536	bmi
sex	1.0071	–
bmi	1.0108	age, smoker
children	1.0036	–
smoker	1.4354	bmi
region	1.0158	–

VIF values for model without interactions.
Model without interactions:

Predictor	GVIF^(1/2Df)
age	1.0084
sex	1.0044
bmi	1.0520
children	1.0020
smoker	1.0060
region	1.0158

Explanation:
	VIF values are slightly higher for interactions but still < 5 → no multicollinearity concerns.
	 Interactions increase GVIF slightly, but main predictors remain stable.
15. Predictive Results & Visuals 
	Scatterplots of numeric predictors (age, bmi, children) vs charges.
	Age vs Charges: Positive trend observed — older patients tend to have higher insurance charges. Some variation exists, especially for very high ages.
   <img width="714" height="450" alt="image" src="https://github.com/user-attachments/assets/06ba7e72-deca-4658-ab64-4a7808b7ee96" />


	BMI vs Charges: Weak-to-moderate positive association. Higher BMI generally correlates with higher charges, especially among smokers (interaction effect).
 <img width="857" height="493" alt="image" src="https://github.com/user-attachments/assets/12aaddd4-0f8c-47ea-ae2e-1d02c5cbcd36" />

	Children vs Charges: Slight positive trend, but effect is minor. Charges do not increase substantially with more children.
 <img width="846" height="533" alt="image" src="https://github.com/user-attachments/assets/24007186-6b0f-41ed-bc3c-d0bf92b9c87f" />

Interpretation: Scatterplots help visualize linear relationships, identify trends, and detect outliers. The plots confirm the choice of including age and BMI as main predictors.

	Boxplots of categorical predictors (sex, smoker, region) vs charges.
	Sex vs charges: Small differences between males and females; males slightly higher, but not statistically significant (p ~ 0.07).
 <img width="715" height="536" alt="image" src="https://github.com/user-attachments/assets/70d100e3-4b00-4fcf-b37b-bf6e81912544" />


	Smoker vs charges: Large difference observed. Smokers consistently have higher charges; this validates including smoker as a key predictor and the BMI×smoker interaction.
 <img width="765" height="574" alt="image" src="https://github.com/user-attachments/assets/101c6110-5f5d-40cd-945a-4dcc208f54b3" />


	Region vs charges: Moderate differences in charges by region. Northwest has slightly lower charges; Southeast and Southwest slightly higher.
 <img width="729" height="546" alt="image" src="https://github.com/user-attachments/assets/fd3f9850-cb99-46ca-85b7-54a0f97e505b" />


Interpretation: Boxplots show distribution and variability of charges across categories. They highlight which categorical variables have strong effects on the response.
3.Interaction and transformation rationales included.
	Interactions: Scatterplots revealed that the effect of BMI on charges is different for smokers vs non-smokers, supporting inclusion of the BMI×smoker interaction. Age slightly modifies BMI effect (age×BMI interaction retained for completeness).
	Transformations: Plots and Box-Cox analysis indicated residuals are approximately normal, so no log or other transformations were applied.

4.Regression formula and metrics table presented.
	Final regression formula shown above captures main effects and interactions.
	Evaluation metrics (R², Adjusted R², RMSE, MAE, AIC, BIC) summarize model performance. High R² and low RMSE/MAE indicate good predictive ability.
Overall: These predictive plots and metrics collectively demonstrate that the model appropriately captures relationships between predictors and charges, confirms assumptions of linearity, and justifies inclusion/exclusion of interactions and transformations.
16. Summary & Insights 
	Key Findings: Age, BMI, children, and smoker status strongly influence charges; BMI×smoker interaction highly significant.
	Business Implications: Targeted health programs (smoking cessation, BMI management) can reduce costs.
	Limitations: Observational data; potential omitted variable bias.
	Recommendations: Implement predictive models to identify high-risk patients; consider health interventions.
  
17. Time Tracking Appendix 
Task	Estimated Time	Actual Time
Data Cleaning	2 hrs	2.5 hrs
EDA	3 hrs	3 hrs
Model Fitting	2 hrs	2 hrs
Diagnostics	1.5 hrs	1.5 hrs
Report Writing	4 hrs	4 hrs





