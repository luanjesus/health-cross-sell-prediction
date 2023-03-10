<h2 align="center"> 
	🚧  Em construção...  🚧
</h2>

<span style="color:yellow"><strong>Disclaimer:</strong> The following context is completely fictitious. The company, context and business issues were created exclusively for the development of the project and are based on a proposal from <a href="https://comunidadeds.com/formacao?utm_source=linkedin&utm_medium=company-page">Comunidade DS</a> and inspired by a <a href="https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction">Kaggle challenge</a>.<span> 

For a more comprehensive and detailed view of the exploratory data analysis, please refer to the following Jupyter notebook.

[![EDA - Jupyter](https://img.shields.io/badge/Exploratory%20Data%20Analysis-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://github.com/luanjesus/health-cross-sell-prediction/blob/main/eda.ipynb)

# HealthGuard Insurance Cross-Sell Prediction: Building a Model to Predict Customer Interest in Vehicle Insurance

<img src="https://github.com/luanjesus/health-cross-sell-prediction/blob/main/repos/img/logo.png" alt="" style="max-width: 100%;">

Our client, an insurance company, is seeking our assistance in creating a predictive model to determine the likelihood of policyholders from the previous year being interested in the company's vehicle insurance. 

Insurance policies involve an agreement in which an insurance company guarantees compensation for specific losses, damages, illnesses, or death in exchange for regular premium payments. For instance, customers may pay an annual premium of $5000 to receive health insurance coverage worth $200.000. If a customer becomes ill and requires hospitalization, the insurance provider will cover the cost of hospitalization up to Rs. $200.000. This concept relies on probabilities, as only a few customers will require hospitalization each year out of a larger pool of premium-paying customers.

Similarly, vehicle insurance requires customers to pay an annual premium to an insurance provider. In case of an unfortunate accident, the insurance provider will compensate the customer with a compensation called 'sum assured'. 

Predicting a customer's interest in vehicle insurance is crucial for the company to optimize its business model and revenue.

# 1. Business Problem

The product team is seeking to create a predictive tool capable of identifying, in a database of 127,000 customers, those with the highest likelihood of joining the new premium health/automobile insurance plan. This is because the plan enrollment will be done through direct contact by the sales team, via phone calls, with only 20000 calls available to contact customers.

1. Main insights about the most relevant attributes of customers interested in acquiring car insurance.
2. What percentage of customers interested in acquiring car insurance will the sales team be able to contact by making 20,000 calls?
3. If the sales team's capacity increases to 40,000 calls, what percentage of customers interested in acquiring car insurance will the sales team be able to contact?
4. How many calls does the sales team need to make to contact 80% of customers interested in acquiring car insurance?

# 2. Data

## 1.1. Data Source

The data was extracted from a Kaggle platform challenge. It contains information about demographics (gender, age, region code type), Vehicles (Vehicle Age, Damage), Policy (Premium, sourcing channel) etc.

**Data Extraction:** [Kaggle Datasets](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)

## 1.2. Data Description

| Feature	| Description |
|-----------|-------------|
|id |	Unique ID for each customer
|Gender|	Gender of the customer
|Age	|Age of the customer
|Driving_License	|0: Customer does not have DL, 1: Customer already has DL
|Region_Code	|Unique code for the region of the customer
|Previously_Insured|	1: Customer already has Vehicle Insurance, 0: Customer doesn't |have Vehicle Insurance
|Vehicle_Age|	Age of the Vehicle
|Vehicle_Damage |	1: Customer got his/her vehicle damaged in the past, 0: Customer didn't get his/her vehicle damaged in the past
|Annual_Premium	|The amount customer needs to pay as premium in the year
|Policy_Sales_Channel |	Anonymized Code for the channel of outreaching to the customer, i.e., Different Agents, Over Mail, Over Phone, In Person, etc.
|Vintage	|Number of Days, Customer has been associated with the company
|Response|	1: Customer is interested, 0: Customer is not interested
