#examples from noutes
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
# read from CSV
df = pd.read_csv('titanic.csv')

print(df.shape)

# import numpy as np
# #shows uniq variables
# print(np.unique(df['state']))

# df['state'].describe()
# df.describe(include = 'all')

#%% [markdown]
# # Introduction to Data Science 2025
# 
# # Week 2

# %% [markdown]
# Exercise 1 | Titanic: data preprocessing and imputation
# Note: You can find tutorials for NumPy and Pandas under 'Useful tutorials' in the course material.

# %% [markdown]
# Download the Titanic dataset (https://www.kaggle.com/c/titanic) train.csv from Kaggle or directly from the course material, and complete the following exercises. If you choose to download the dataset from Kaggle, you will need to create a Kaggle account unless you already have one, but it is quite straightforward.
# 
# The dataset consists of personal information of all the passengers on board the RMS Titanic, along with information about whether they survived the iceberg collision or not.
# 
# 1. Your first task is to read the data file and print the shape of the data.


#     Hint 1: You can read them into a Pandas dataframe if you wish.
#     
#     Hint 2: The shape of the data should be (891, 12).

# %%
# Use this cell for your code

df = pd.read_csv('titanic.csv')
print(df.shape)

# %% [markdown]
# 2. Let's look at the data and get started with some preprocessing. Some of the columns, e.g Name, simply identify a person and are not useful for prediction tasks. Try to identify these columns, and remove them.
# 
#     Hint: The shape of the data should now be (891, 9).

# %%
# Use this cell for your code
df = df.drop(['PassengerId','Name','Ticket'],axis=1)

#df.describe()
# %% [markdown]
# 3. The column Cabin contains a letter and a number. A smart catch at this point would be to notice that the letter stands for the deck level on the ship. Keeping just the deck information would be more informative when developing, e.g. a classifier that predicts whether a passenger survived. The next step in our preprocessing will be to add a new column to the dataset, which consists simply of the deck letter. You can then remove the original Cabin-column.
# 
#     Hint: The deck letters should be ['A' 'B' 'C' 'D' 'E' 'F' 'G' 'T'].

# %%
# Use this cell for your code
df['Deck'] = df['Cabin'].str[0]
df = df.drop('Cabin',axis=1)
 
# %% [markdown]
# 4. Youâ€™ll notice that some of the columns, such as the previously added deck number, are categorical (https://en.wikipedia.org/wiki/Categorical_variable). To preprocess the categorical variables so that they're ready for further computation, we need to avoid the current string format of the values. This means the next step for each categorical variable is to transform the string values to numeric ones, that correspond to a unique integer ID representative of each distinct category. This process is called label encoding and you can read more about it here (https://pandas.pydata.org/docs/user_guide/categorical.html).
# 
#     Hint: Pandas can do this for you.

# %%
# Use this cell for your code
df['Deck'] = df['Deck'].astype('category').cat.codes

# %% [markdown]
# 5. Next, let's look into missing value imputation. Some of the rows in the data have missing values, e.g when the cabin number of a person is unknown. Most machine learning algorithms have trouble with missing values, and they need to be handled during preprocessing:
# 
#     a) For continuous variables, replace the missing values with the mean of the non-missing values of that column.
# 
#     b) For categorical variables, replace the missing values with the mode of the column.
# 
#         Remember: Even though in the previous step we transformed categorical variables into their numeric representation, they are still categorical.

# %%
# Use this cell for your code
df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Embarked'] = df['Embarked'].astype('category').cat.codes
df['Embarked'] = df['Embarked'].replace(-1,np.nan)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Deck'] = df['Deck'].replace(-1,np.nan)
df['Deck'] = df['Deck'].fillna(df['Deck'].mode()[0])

for col in df.columns:
    print(f"--- {col} ---")
    print(df[col].unique())
    print()
    
# %% [markdown]
# 6. At this point, all data is numeric. Write the data, with the modifications we made, to a .csv file. Then, write another file, this time in JSON format, with the following structure:

# %%
#[
#    {
#        "Deck": 0,
#        "Age": 20,
#        "Survived", 0
#        ...
#    },
#    {
#        ...
#    }
#]

# %%
# Use this cell for your code
df.to_csv('titanic_clean.csv', index=False)
df.to_json('titanic_clean.json', orient='records', lines=False)
# %% [markdown]
# Study the records and try to see if there is any evident pattern in terms of chances of survival.
# From what i see in the data survivers are mostly female
# %% [markdown]
# Remember to submit your code on the MOOC platform. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences.

# %% [markdown]
# ## Exercise 2 | Titanic 2.0: exploratory data analysis
# 
# In this exercise, weâ€™ll continue to study the Titanic dataset from the last exercise. Now that we have done some preprocessing, itâ€™s time to look at the data with some exploratory data analysis.

# %% [markdown]
# 1. First investigate each feature variable in turn. For each categorical variable, find out the mode, i.e., the most frequent value. For numerical variables, calculate the median value.

# %%
# Use this cell for your code
print(f"In average Survived: {df['Survived'].mode()[0]} Sex: {df['Sex'].mode()[0]} Pclass: {df['Pclass'].mode()[0]} Age: {df['Age'].mean()} SibSp: {df['SibSp'].mode()[0]} Parch: {df['Parch'].mode()[0]} Fare: {df['Fare'].mean()} Embarked: {df['Embarked'].mode()[0]} Deck: {df['Deck'].mode()[0]}")
# %% [markdown]
# 2. Next, combine the modes of the categorical variables, and the medians of the numerical variables, to construct an imaginary â€œaverage survivorâ€. This "average survivor" should represent the typical passenger of the class of passengers who survived. Also following the same principle, construct the â€œaverage non-survivorâ€.
# 
#     Hint 1: What are the average/most frequent variable values for a non-survivor?
#     
#     Hint 2: You can split the dataframe in two: one subset containing all the survivors and one consisting of all the non-survivor instances. Then, you can use the summary statistics of each of these dataframe to create a prototype "average survivor" and "average non-survivor", respectively.

# %%
# Use this cell for your code
alive = df[df['Survived'] == 1]
dead = df[df['Survived'] == 0]
print(f"Average survivor Sex: {alive['Sex'].mode()[0]} Pclass: {alive['Pclass'].mode()[0]} Age: {alive['Age'].mean()} SibSp: {alive['SibSp'].mode()[0]} Parch: {alive['Parch'].mode()[0]} Fare: {alive['Fare'].mean()} Embarked: {alive['Embarked'].mode()[0]} Deck: {alive['Deck'].mode()[0]}")
print(f"Average non survivor Sex: {dead['Sex'].mode()[0]} Pclass: {dead['Pclass'].mode()[0]} Age: {dead['Age'].mean()} SibSp: {dead['SibSp'].mode()[0]} Parch: {dead['Parch'].mode()[0]} Fare: {dead['Fare'].mean()} Embarked: {dead['Embarked'].mode()[0]} Deck: {dead['Deck'].mode()[0]}")
# %% [markdown]
# 3. Next, let's study the distributions of the variables in the two groups (survivor/non-survivor). How well do the average cases represent the respective groups? Can you find actual passengers that are very similar to the (average) representative of their own group? Can you find passengers that are very similar to the (average) representative of the other group?
# 
#     Note: Feel free to choose EDA methods according to your preference: non-graphical/graphical, static/interactive - anything goes.

# %%
# Use this cell for your code
alive['Sex'].value_counts().plot(kind="bar",color="blue")
plt.title("Sex of Survivors")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

alive['Pclass'].value_counts().plot(kind="bar",color="blue")
plt.title("Class of Survivors")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

sns.kdeplot(alive["Age"].dropna(), shade=True, color="green")
plt.title("Age of Survivors")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

dead['Sex'].value_counts().plot(kind="bar",color="blue")
plt.title("Sex of Survivors")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

dead['Pclass'].value_counts().plot(kind="bar",color="blue")
plt.title("Class of Survivors")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

sns.kdeplot(dead["Age"].dropna(), shade=True, color="green")
plt.title("Age of Survivors")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
# %% [markdown]
# 4. Next, let's continue the analysis by looking into pairwise and multivariate relationships between the variables in the two groups. Try to visualize two variables at a time using, e.g., scatter plots and use a different color to encode the survival status.
# 
#     Hint 1: You can also check out Seaborn's pairplot function, if you wish.
# 
#     Hint 2: To better show many data points with the same value for a given variable, you can use either transparency or â€˜jitterâ€™.

# %%
# Use this cell for your code
sns.scatterplot(
    data=df,
    x="Age",
    y="Fare", 
    hue="Survived",
    alpha=0.6
)
plt.title("Age vs Fare colored by Survival")
plt.show()

sns.scatterplot(
    data=df,
    x="Age",
    y="Pclass", 
    hue="Survived",
    alpha=0.6
)
plt.title("Age vs Fare colored by Survival")
plt.show()
# %% [markdown]
# 5. Finally, recall the preprocessing we did in the first exercise. What can you say about the effect of the choices that were made to use the mode and mean to impute missing values, instead of, for example, ignoring passengers with missing data?

# %% [markdown]
# Use this (markdown) cell for your written answer

# %% [markdown]
#if i wouldnt have replaced the data with a mean and mode i would likly have a cleaner but smaller data set probably not big enoght to make any real conclysions but now since i dont realy know how much data was corrupted by this choise i dont know how relible is the end result.
# Remember to submit your code on the MOOC platform. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences.

# %% [markdown]
# ## Exercise 3 | Working with text data 2.0
# 
# This exercise is related to the second exercise from last week. Find the saved pos.txt and neg.txt files, or, alternatively, you can find the week 1 example solutions on the MOOC platform after Tuesday.

# %% [markdown]
# 1. Find the most common words in each file (positive and negative). Examine the results. Do they tend to be general terms relating to the nature of the data? How well do they indicate positive/negative sentiment?

# %%
# Use this cell for your code
with open('pos.txt', 'r', encoding='utf-8') as f:
    pos_reviews = [line.strip() for line in f]

pos = pd.DataFrame({'review': pos_reviews})
pos["words"] = pos["review"].str.split()
words = pos["words"].explode()
word_counts_pos = words.value_counts()

with open('neg.txt', 'r', encoding='utf-8') as f:
    neg_reviews = [line.strip() for line in f]

neg = pd.DataFrame({'review': neg_reviews})
neg["words"] = neg["review"].str.split()
words = neg["words"].explode()
word_counts_neg = words.value_counts()
print(word_counts_neg)
# %% [markdown]
# 2. Compute a TF/IDF (https://en.wikipedia.org/wiki/Tfâ€“idf) vector for each of the two text files, and make them into a 2 x m matrix, where m is the number of unique words in the data. The problem with using the most common words in a review to analyze its contents is that words that are common overall will be common in all reviews (both positive and negative). This means that they probably are not good indicators about the sentiment of a specific review. TF/IDF stands for Term Frequency / Inverse Document Frequency (here the reviews are the documents), and is designed to help by taking into consideration not just the number of times a term occurs (term frequency), but also how many times a word exists in other reviews as well (inverse document frequency). You can use any variant of the formula, as well as off-the-shelf implementations. Hint: You can use sklearn (http://scikit-learn.org/).

# %%
# Use this cell for your code
with open('pos.txt', 'r', encoding='utf-8') as f:
    pos_reviews = f.read() 

with open('neg.txt', 'r', encoding='utf-8') as f:
    neg_reviews = f.read() 
docs = [pos_reviews,neg_reviews]
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(docs)
df = pd.DataFrame(matrix.toarray(),index=['pos','neg'],columns=vectorizer.get_feature_names_out())
threshold = 0.01
filtered_cols = df.columns[(df >= threshold).any(axis=0)]
df_filtered = df[filtered_cols]

df_filtered = df_filtered[df_filtered.sum().sort_values(ascending=False).index]
print(df_filtered.head())
# %% [markdown]
# 3. List the words with the highest TF/IDF score in each class (positive | negative), and compare them to the most common words. What do you notice? Did TF/IDF work as expected?

# %%
# Use this cell for your code
top_pos = df.loc['pos'].sort_values(ascending=False).head(10)
print("Top 10 words in positive reviews:")
print(top_pos)

top_neg = df.loc['neg'].sort_values(ascending=False).head(10)
print("Top 10 words in negative reviews:")
print(top_neg)
#most words are common for both so the TF/IDF didnt work as expected.
# %% [markdown]
# 4. Plot the words in each class with their corresponding TF/IDF scores. Note that there will be a lot of words, so youâ€™ll have to think carefully to make your chart clear! If you canâ€™t plot them all, plot a subset â€“ think about how you should choose this subset.
# 
#     Hint: you can use word clouds. But feel free to challenge yourselves to think of any other meaningful way to visualize this information!

# %%

pos_words = df.loc['pos'][df.loc['pos'] >= 0.01].to_dict()
pos_wc = WordCloud(
    width=800, 
    height=400, 
    background_color='pink'
).generate_from_frequencies(pos_words)

plt.figure(figsize=(12, 6))
plt.imshow(pos_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Word Cloud", fontsize=16)
plt.show()


neg_words = df.loc['neg'][df.loc['neg'] >= 0.01].to_dict()
neg_wc = WordCloud(
    width=800, 
    height=400, 
    background_color='blue'
).generate_from_frequencies(neg_words)

plt.figure(figsize=(12, 6))
plt.imshow(neg_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Word Cloud", fontsize=16)
plt.show()
# %% [markdown]
# Remember to submit your code on the MOOC platform. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences.

# %% [markdown]
# ## Exercise 4 | Junk charts
# 
# Thereâ€™s a thriving community of chart enthusiasts who keep looking for statistical graphics that they find inappropriate, and which they call â€œjunk chartsâ€, and who often also propose ways to improve them.

# %% [markdown]
# 1. Find at least three statistical visualizations you think are not very good and identify their problems. Copying examples from various junk chart websites is not accepted â€“ you should find your own junk charts, out in the wild. You should be able to find good (or rather, bad) examples quite easily since a significant fraction of charts can have at least *some* issues. The examples you choose should also have different problems, e.g., try to avoid collecting three bar charts, all with problematic axes. Instead, try to find as interesting and diverse examples as you can.

# %% [markdown]
# 2. Try to produce improved versions of the charts you selected. The data is of course often not available, but perhaps you can try to extract it, at least approximately, from the chart. Or perhaps you can simulate data that looks similar enough to make the point.
# 
# 

# %% [markdown]
# Submit a PDF with all the charts (the ones you found and the ones you produced).
