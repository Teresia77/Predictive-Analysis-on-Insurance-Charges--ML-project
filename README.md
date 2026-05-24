# Predictive Analysis on Insurance Dataset  
### Predicting Medical Insurance Charges Using Demographic and Lifestyle Factors

## Project Overview
This project builds a predictive model to estimate medical insurance charges based on demographic and lifestyle variables. The goal is to understand key cost drivers and support data-driven 
insurance pricing decisions using multiple linear regression.

## Business Problem
Insurance companies need accurate methods to estimate medical costs for policyholders. Charges vary based on factors such as age, BMI, smoking status, and number of children.

This project answers:
How can we predict medical insurance charges using customer attributes?

Business value:
- Improve insurance premium pricing accuracy  
- Identify high-risk individuals (e.g., smokers, high BMI)  
- Support underwriting and actuarial decisions  
- Enable cost forecasting using data  

## Dataset Information
Source: GeeksforGeeks Insurance Dataset  
File: insurance.csv  
Records: 1,338  
Variables: 7  
Data type: Mixed (numeric and categorical)

## Variables Description
age: Age of policyholder  
sex: Gender (male/female)  
bmi: Body Mass Index  
children: Number of dependents  
smoker: Smoking status (yes/no)  
region: Residential region in the U.S.  
charges: Medical insurance cost (target variable)

## Methodology

1. Data Preparation  
- Loaded dataset using Python (Pandas)  
- Checked missing values  
- Encoded categorical variables  

2. Exploratory Data Analysis  
- Summary statistics (mean, SD, skewness, kurtosis)  
- Boxplots for numerical variables  
- Distribution analysis of charges  

3. Predictive Modeling  
- Multiple Linear Regression model  
- Selected predictors: age, sex, BMI, children, smoker, region  
- Model trained using Python (scikit-learn / statsmodels)

4. Model Evaluation  
- R² (Coefficient of Determination)  
- Residual analysis  
- Model performance interpretation  

## Key Insights
- Average insurance charges are approximately $13,270  
- Charges are highly right-skewed  
- Smoking is the strongest predictor of insurance cost  
- BMI and age significantly influence charges  
- Gender and region have smaller effects  

## Exploratory Data Analysis Findings
- No extreme outliers in age or BMI  
- Strong variation in insurance charges  
- Smokers incur significantly higher costs  
- Number of children has a mild effect  

## Predictive Objective
Target variable: charges  
Predictors: age, sex, BMI, children, smoker, region  
Model type: Multiple Linear Regression  
Evaluation metric: R² score  

## Tools and Technologies
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn / Statsmodels  
- Jupyter Notebook / Spyder  

## Results Summary
The model shows that insurance charges can be predicted using demographic and lifestyle variables. Smoking status and BMI are the most influential predictors of cost.

## Future Improvements
- Use advanced models (Random Forest, XGBoost)  
- Feature interaction analysis  
- Hyperparameter tuning  
- Model deployment as a web application  

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

Interpretation:
- Charges are moderately influenced by age and BMI  
- Weak correlation among predictors supports regression modeling  

---

## 8. Linear Regression Model Specification

Final model:

charges = β₀ + β₁(age) + β₂(sex) + β₃(bmi) + β₄(children) + β₅(smoker) + β₆(region)  
+ β₇(age × bmi) + β₈(smoker × bmi) + ε  

- Categorical variables encoded using dummy variables  
- Interaction terms included for BMI × smoker and age × BMI  
- No transformation applied (validated via Box-Cox & residuals)

---

## 9. Interaction Terms

- BMI effect differs significantly between smokers and non-smokers  
- Age slightly modifies BMI effect  
- Interaction terms improve model fit and interpretability  

---

## 10. Non-Linear Transformations

- Box-Cox transformation tested  
- Residuals already approximately normal  
- No transformation applied  

---

## 11. Train/Test Split and Model Fitting

- Training set: 1,072 (80%)  
- Testing set: 266 (20%)  

Model trained using lm() with interaction terms.  
Continuous variables centered before interaction creation.

### Model Coefficients (Training)

- Intercept: 573.466  
- age: 192.314  
- sexmale: -533.383  
- bmi: -61.093  
- children: 422.161  
- smokeryes: -21,268.984  
- region effects included  
- age:bmi: 2.113  
- bmi:smoker: 1478.168  

---

## 12. Model Evaluation

| Metric | Value | Interpretation |
|--------|------|----------------|
| R² | 0.8082 | 80.8% variance explained |
| Adj R² | 0.8007 | Adjusted for predictors |
| RMSE | 5,128 | Prediction error magnitude |
| MAE | 2,963 | Average absolute error |
| AIC | 21,218 | Model fit quality |
| BIC | 21,278 | Penalized complexity metric |

---

## 13. Diagnostic Plots

- Residuals vs Fitted: Random pattern → good fit  
- Q-Q Plot: Approx. normal residuals  
- Scale-Location: Constant variance  
- Leverage Plot: No influential outliers  

Conclusion: Model assumptions satisfied  

---

## 14. VIF Analysis

- All GVIF values < 5  
- No multicollinearity issues detected  
- Interaction terms slightly increase GVIF but remain acceptable  

---

## 15. Predictive Results

Key findings:
- Age: Positive relationship with charges  
- BMI: Moderate positive effect  
- Children: Weak effect  
- Smokers: Strong increase in charges  

Categorical effects:
- Smokers have significantly higher charges  
- Region and sex have minor effects  

---

## 16. Summary & Insights

- Strong predictors: age, BMI, smoker status  
- BMI × smoker interaction is highly significant  
- Model explains ~81% of variance in charges  

Business implications:
- Smoking cessation programs reduce insurance costs  
- BMI management programs can lower risk  
- Predictive modeling improves pricing fairness  

Limitations:
- Observational dataset  
- Possible omitted variable bias  

---

## 17. Time Tracking

| Task | Estimated | Actual |
|------|----------|--------|
| Data Cleaning | 2 hrs | 2.5 hrs |
| EDA | 3 hrs | 3 hrs |
| Modeling | 2 hrs | 2 hrs |
| Diagnostics | 1.5 hrs | 1.5 hrs |
| Reporting | 4 hrs | 4 hrs |

---

## Author
Teresia Wainaina  
MSc Business Analytics


