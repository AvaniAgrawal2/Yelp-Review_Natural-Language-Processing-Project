#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Natural Language Processing Project
# 
# In this NLP project you will be attempting to classify Yelp Reviews into 1 star or 5 star categories based off the text content in the reviews.
# We will use the [Yelp Review Data Set from Kaggle](https://www.kaggle.com/c/yelp-recsys-2013).
# 
# Each observation in this dataset is a review of a particular business by a particular user.
# 
# The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.
# 
# The "cool" column is the number of "cool" votes this review received from other Yelp users. 
# 
# All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.
# 
# The "useful" and "funny" columns are similar to the "cool" column.
# 

# ## Imports
#  **Import the usual suspects. :) **

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data
# 
# **Read the yelp.csv file and set it as a dataframe called yelp.**

# In[3]:


yelp = pd.read_csv('yelp.csv')


# ** Check the head, info , and describe methods on yelp.**

# In[4]:


yelp.head(2)


# In[5]:


yelp.info()


# In[6]:


yelp.describe()


# **Create a new column called "text length" which is the number of words in the text column.**

# In[7]:


yelp['text_length'] = yelp['text'].apply(len)


# # EDA
# 
# Let's explore the data
# 
# ## Imports
# 
# **Import the data visualization libraries if you haven't done so already.**

# In[8]:


sns.set_style('white')


# **Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings. Reference the seaborn documentation for hints on this**

# In[9]:


g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text_length',bins=50)


# **Create a boxplot of text length for each star category.**

# In[10]:


sns.boxplot(x='stars', y='text_length', data=yelp ,palette='rainbow' )


# **Create a countplot of the number of occurrences for each type of star rating.**

# In[11]:


sns.countplot(x='stars', data = yelp, palette = 'rainbow' )


# ** Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation:**

# In[12]:


stars = yelp.groupby('stars').mean()
stars


# **Use the corr() method on that groupby dataframe to produce this dataframe:**

# In[13]:


stars.corr()


# **Then use seaborn to create a heatmap based off that .corr() dataframe:**

# In[14]:


sns.heatmap(stars.corr(),cmap='coolwarm' , annot = True)


# ## NLP Classification Task
# 
# Let's move on to the actual task. To make things a little easier, go ahead and only grab reviews that were either 1 star or 5 stars.
# 
# **Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.**

# In[15]:


yelp_class = yelp[(yelp['stars']==1) |(yelp['stars']==5)]


# In[16]:


yelp_class.info()


# ** Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)**

# In[25]:


X =yelp_class['text']
y = yelp_class['stars']


# **Import CountVectorizer and create a CountVectorizer object.**

# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# ** Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.**

# In[27]:


X = cv.fit_transform(X)


# ## Train Test Split
# 
# Let's split our data into training and testing data.
# 
# ** Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101 **

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33 ,random_state=101)


# ## Training a Model
# 
# Time to train a model!
# 
# ** Import MultinomialNB and create an instance of the estimator and call is nb **

# In[30]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# **Now fit nb using the training data.**

# In[33]:


nb.fit(X_train,y_train)


# ## Predictions and Evaluations
# 
# Time to see how our model did!
# 
# **Use the predict method off of nb to predict labels from X_test.**

# In[34]:


predctions = nb.predict(X_test)


# ** Create a confusion matrix and classification report using these predictions and y_test **

# In[35]:


from sklearn.metrics import confusion_matrix, classification_report


# In[36]:


print(confusion_matrix(y_test,predctions))
print('\n')
print(classification_report(y_test,predctions))


# **Great! Let's see what happens if we try to include TF-IDF to this process using a pipeline.**

# # Using Text Processing
# 
# ** Import TfidfTransformer from sklearn. **

# In[38]:


from sklearn.feature_extraction.text import TfidfTransformer


# ** Import Pipeline from sklearn. **

# In[39]:


from sklearn.pipeline import Pipeline


# ** Now create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()**

# In[40]:


pipe = Pipeline([('bow' , CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', MultinomialNB())])


# ## Using the Pipeline
# 
# **Time to use the pipeline! Remember this pipeline has all your pre-process steps in it already, meaning we'll need to re-split the original data (Remember that we overwrote X as the CountVectorized version. What we need is just the text**

# ### Train Test Split
# 
# **Redo the train test split on the yelp_class object.**

# In[41]:


X = yelp_class['text']
y = yelp_class['stars']

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33 ,random_state=101)


# **Now fit the pipeline to the training data. Remember you can't use the same training data as last time because that data has already been vectorized. We need to pass in just the text and labels**

# In[42]:


pipe.fit(X_train,y_train)


# ### Predictions and Evaluation
# 
# ** Now use the pipeline to predict from the X_test and create a classification report and confusion matrix. You should notice strange results.**

# In[43]:


predctions = pipe.predict(X_test)


# In[44]:


print(confusion_matrix(y_test,predctions))
print('\n')
print(classification_report(y_test,predctions))


# Looks like Tf-Idf actually made things worse! That is it for this project. But there is still a lot more you can play with:
# 
# **Some other things to try....**
# Try going back and playing around with the pipeline steps and seeing if creating a custom analyzer like we did in the lecture helps (note: it probably won't). Or recreate the pipeline with just the CountVectorizer() and NaiveBayes. Does changing the ML model at the end to another classifier help at all?

# # Great Job!
