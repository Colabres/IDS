# %% [markdown]
# # Introduction to Data Science 2025
# 
# # Week 3

# %% [markdown]
# ## Exercise 1 | Working with geospatial data (GIS)

# %% [markdown]
# To get at least a bit familiar with GIS (https://en.wikipedia.org/wiki/Geographic_information_system) data and the concept of map projections, we’ll do a simple task of plotting two data sets that are given in different coordinate systems.
# 
# 1. Download the world_m.zip (https://github.com/HY-TKTL/intro-to-data-science-2017/blob/master/world_m.zip) and cities.zip (https://github.com/HY-TKTL/intro-to-data-science-2017/blob/master/cities.zip) files that each include a 
# set of GIS files. Most notably, the shp files are Shapefile-files (https://en.wikipedia.org/wiki/Shapefile) with coordinates (don’t look, it’s binary!). The prj files contain information (in plain text, so okay to look) about 
# the coordinate systems. Open the files using your favorite programming environment and packages.  
# 
#     Hint: We warmly recommend Geopandas (http://geopandas.org/) for pythonistas.

# %%
# Use this cell for your code
import geopandas as gpd

world = gpd.read_file("world_m.shp")
cities = gpd.read_file("cities.shp")
print(world.crs)
print(cities.crs)
# %% [markdown]
# 2. The world_m file contains borders of almost all countries in the world. Plot the world.

# %%
# Use this cell for your code
import matplotlib.pyplot as plt
m = world.plot(figsize=(12, 8), color='lightgrey', edgecolor='black')
plt.title("World Map")
#plt.show()
# %% [markdown]
cities = cities.to_crs(world.crs)
cities.plot(ax=m, color='red', markersize=20)
plt.show()
# %%
# Use this cell for your code

# %% [markdown]
# 4. Perform a map projection to bring the two data-sets into a shared coordinate system. (You can choose which one.) Now plot the two layers together to make sure the capital cities are where they are supposed to be.

# %%
# Use this cell for your code
m = world.plot(figsize=(12, 8), color='lightgrey', edgecolor='black')
plt.title("World Map")
cities = cities.to_crs(world.crs)
cities.plot(ax=m, color='red', markersize=20)
plt.show()
# %% [markdown]
# **Remember to submit your code on the MOOC platform. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences.**

# %% [markdown]
# ## Exercise 2 | Symbol classification
# 
# We’ll be looking into machine learning by checking out the HASYv2 (https://zenodo.org/record/259444#.Wb7efZ8xDhZ) dataset that contains hand written mathematical symbols as images. The whole dataset is quite big, so we’ll restrict ourselves to doing 10-class classification on some of the symbols. Download the data and complete the following tasks.

# %% [markdown]
# 1. Extract the data and find inside a file called hasy-data-labels.csv. This file contains the labels for each of the images in the hasy_data folder. Read the labels in and only keep the rows where the symbol_id is within the inclusive range [70, 79]. Read the corresponding images as black-and-white images and flatten them so that each image is a single vector of shape 32x32 = 1024. Your dataset should now consist of your input data of shape (1020, 1024) and your labels of shape (1020, ). That is, a matrix of shape 1020 x 1024 and a vector of size 1020.

# %%
# Use this cell for your code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
labels = pd.read_csv("hasy-data-labels.csv")
labels = labels[((labels["symbol_id"] >= 70) & (labels["symbol_id"] <=79) )]
matrix = []
print(labels.columns)
print(labels["path"].head())
for path in labels["path"]:
    img = plt.imread(f"{path}")
    img = img[:, :, 0]
    img = img.flatten()
    matrix.append(img)

matrix = np.array(matrix)
print(matrix.shape)
ids = labels["symbol_id"].values
print(ids)

# %% [markdown]
# 2. Randomly shuffle the data, and then split it into training and test sets, using the first 80% of the data for training and the rest for evaluation.

# %%
# Use this cell for your code
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(matrix, ids,test_size=0.2,shuffle=True)

# %% [markdown]
# 3. Fit a logistic regression classifier on the data. Note that we have a multi-class classification problem, but logistic regression is a binary classifier. For this reason, you will find useful Sklearn's "multi_class" attribute (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). Use a multinomial loss and the softmax function to predict the probability of each class as the outcome.  The classifier should select the class with the highest probability. Most library implementations will do this for you - feel free to use one.

# %%
# Use this cell for your code
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, max_iter=1000,multi_class="multinomial").fit(X_train, y_train)
print("Training accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))
# %% [markdown]
# 4. In order to evaluate the model, let’s create our own dummy classifier that simply guesses the most common class in the training set (looking at the test set here would be cheating!). Then, evaluate your logistic regression model on the test data, and compare it to the majority class classifier. The logistic regression model should have significantly better accuracy as the dummy model is merely making a guess.
# 
#     Hint: Sklearn's DummyClassifier( ) might save you a bit of time.

# %%
# Use this cell for your code
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
print("Dummy accuracy:", dummy_clf.score(X_test, y_test))
print("Logistic regression accuracy:", clf.score(X_test, y_test))
# %% [markdown]
# 5. Plot some of the images that the logistic classifier misclassified. Can you think of an explanation why they were misclassified? Would you have gotten them right?
#     
#     Hint: Matplotlib has a function (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html) that can help you with plotting.

# %% [markdown]
# Here are some examples of the syntax used to fit a logistic regression classifier (using Sklearn or statsmodel with Python, or GLM with R):

# %%
#Sklearn (python)

#from sklearn.linear_model import LogisticRegression

#Fit on the training data set, with:
#X : {array-like, sparse matrix}, shape (n_samples, n_features) , n_samples rows x n_features columns
#with attributes that describe each sample.
#y : array-like, shape (n_samples,) , n_samples target values for each sample.

#model = LogisticRegression()
#model.fit(X, y)

# %%
#Statsmodels (python)

#import statsmodels.api as sm
#model = sm.Logit(y, X)

# %%
#GLM (R)

#model <- glm(y ~.,family=binomial(link='logit'), data=X)

# %%
# Use this cell for your code
pred = clf.predict(X_test)
misclassified = np.where(pred != y_test)[0]
plt.figure(figsize=(12, 6))

for i, idx in enumerate(misclassified[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(32, 32), cmap="gray")    

plt.tight_layout()
plt.show()
# %% [markdown]
# **Remember to submit your code on the MOOC platform. You can return this Jupyter notebook (.ipynb) or .py, .R, etc depending on your programming preferences.**


