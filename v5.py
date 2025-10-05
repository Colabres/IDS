# %% [markdown]
# # Introduction to Data Science 2025
# 
# # Week 5

# %% [markdown]
# ## Exercise 1 | Privacy and data protection

# %% [markdown]
# First, look up the European Data Protection Regulation (http://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679&from=en) (“GDPR”). Note that Articles 1-99 start on p. 32 of the document. We will refer to the articles and their parts by, e.g., “Art 6 (1) a) GDPR” which means Article 6 (“Lawfulness of processing”), first paragraph, item a in the GDPR.
# 
# 1. Valid Consent?
# 
#     Find a service you use to which you have given consent for the processing of your personal data (Art 6 (1) a) GDPR). Have a look at the privacy notices, policies, or settings of this service.
# 
#     - Are the basic legal conditions for this consent in your opinion in line with the new requirements and conditions set by the GDPR?
# 
#     - You should provide an answer with justification based on the GDPR, where you refer to specific articles and paragraphs.
# 
# 2. Your Right to Access your Personal Data
# 
#     You have the right to know if personal data about you is processed by a controller. You also have the right to get access to, for example, the processing purposes, the data categories, data transfers, and duration of storage.
# 
#     - Find the relevant parts in GDPR and study your rights as a “data subject”.
# 
#     - File a right-to-access request with a data processing service of your choosing. Describe the mechanism that is put in place by the service to enable you to exercise this right (if any).
# 
#     - Whether you get a response or not, think about how well your rights as a data subject are respected in practice. Your answer should again refer to specific articles and paragraphs of the GDPR.
# 
# 3. Anonymisation & Pseudonymisation
# 
#     - What is the difference between anonymisation and pseudonymisation of personal data?
# 
# **Submit your findings in a PDF file, just a short report is enough.**

# %% [markdown]
# ## Exercise 2 | Fairness-aware AI

# %% [markdown]
# This template generates data about the working hours and salaries of n=5000 people. The salary equals 100 x working hours plus/minus normal distributed noise. If you run the template, it produces an hours vs monthly salary scatter plot with gender=0 (men) in orange and gender=1 (women) in orange. The plot includes a trend line for each group, and an overall trend line for all data combined (in red). 
# 
# A linear regression model (see the next code cell) that only includes the working hours as a covariate without the protected characteristic (gender) should have slope close to 100.0.

# %%
#%matplotlib inline
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# %% [markdown]
# ### Simulating the data

# %%
#sample size n
n = 5000

# # gender
gen = np.random.binomial(1, 0.5, size=n)

# work hours
#hrs = np.random.binomial(60, 0.5, size=n)

# salary = 100 * hours + noise (std.deviation 10)
#sal = hrs * np.random.normal(100, 10, size=n) 


hrs = np.empty(n, dtype=int)
hrs[gen == 0] = np.random.binomial(60, 0.55, size=(gen == 0).sum())  
hrs[gen == 1] = np.random.binomial(60, 0.45, size=(gen == 1).sum())
sal = hrs * np.random.normal(100, 10, size=n)
sal[gen == 1] -= 400
# create a nice data frame
data = pd.DataFrame({"Gender": gen, "Hours": hrs, "Salary": sal})

# %% [markdown]
# ### Scatterplot of the simulated data
# Women samples (gender = 1) are shown with blue, men samples (gender = 0) are shown in orange.
# Blue and orange lines are the trend lines of each group accordingly.
# The overall trend line is shown in red.

# %%
# sns.regplot(x="Hours", y="Salary", data=data[data["Gender"]==1], color="darkBlue", scatter_kws={'alpha':0.5})

# sns.regplot(x="Hours", y="Salary", data=data[data["Gender"]==0], color="darkOrange", scatter_kws={'alpha':0.5})

# sns.regplot(x="Hours", y="Salary", data=data, marker="None", color="red")

# plt.show()

# %% [markdown]
# ### Linear regression
# Learn the overall regression model, which is what an algorithm with no access to the gender ("protected characteristic") would learn from the data.

# %%
reg_all = LinearRegression().fit(hrs.reshape(-1,1), sal.reshape(-1,1))
reg_men = LinearRegression().fit(hrs[(gen == 0)].reshape(-1,1), sal[(gen == 0)].reshape(-1,1))
reg_women = LinearRegression().fit(hrs[(gen == 1)].reshape(-1,1), sal[(gen == 1)].reshape(-1,1))
# print out the slope: it should be close to 100.0 without learning the 'protected characteristic' (gender)
print("slope: %.1f" % reg_all.coef_[0][0])
print("slope: %.1f" % reg_men.coef_[0][0])
print("slope: %.1f" % reg_women.coef_[0][0])
# %% [markdown]
# ### Task
# 
# Now edit the code to simulate each of the following scenarios:
# 
# a) the salary of women is reduced by 200 euros ("direct discrimination")
# 
# b) the working hours of men are binomially distributed with parameters (60, 0.55) while the working hours of women are binomially distributed with parameters (60, 0.45), no changes in per-hour salary ("no discrimination")
# 
# c) both of the above changes at the same time ("indirect discrimination")
# 
# You should be able to demonstrate that the slope of the linear regression model is only changed in one of these scenarios.
# 
# Based on this experiment, answer the following questions:
# 1. In which of the scenarios the slope (coefficient) of the regression model changes?
# Slope angle change only in scenario C in A and B angle stay the same.
# 2. How could you model the data in a way that enables you to detect indirect discrimination? Hint: Should you include the protected characteristic in the model or not?
# We should include the protected characteristic in the model this way we can ditect te hiden bias.
# To answer the second question, demonstrate your solution by building a regression model and interpreting the estimated coefficients.
# After the modification we can clearly see that for both groups calculated slop is the same but for combined group its different that indecates bias.
# **Submit this exercise by submitting your code and your answers to the above questions on the MOOC platform. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences.**


