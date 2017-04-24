
# coding: utf-8

# In[38]:

import pandas as pd
import re
from nltk.corpus import stopwords # Import the stop word list
from autocorrect import spell #import spell checker
import numpy as np
from textstat.textstat import textstat #import vocabulary level grader
from sklearn.cross_validation import train_test_split #for training and testing split
from textblob.classifiers import NaiveBayesClassifier
import csv
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfTransformer
import nltk


# # This code cleans the essays by:
# 
# 1. Removning Stopwords
# 2. Stemming
# 3. removing numerical values
# 4. puts everything in lowercase

# In[2]:

#this block of code is for preprocessing the data you only need to run this once if you don't have the .csv files
def clean_Essay( raw_review ):
    stemmer = nltk.stem.SnowballStemmer('english')
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [spell(stemmer.stem(w)) for w in words if not w in stops]   
    # 6. Doing a spell corrector
    # 7. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))
#this is for setting up the data only if CSV file not avaible
#train_all_sets.csv is the data file from the keggle website for short answer
#after being saved as a csv file using excel.
#it will be uploaded to the git repository 
df = pd.read_csv("train_all_sets.csv", index_col=False)
df.columns
sets = [3,4,7,8,9]
df=df.loc[df["EssaySet"].isin(sets)]#we only keep essays for datasets of enligh subject
df = df[['EssayText','Score1']] #keep on 2 cols
df['EssayText'] = df['EssayText'].apply(lambda x: clean_Essay(x))#check cell above
#this is to check if we have a null value
df[df.isnull().any(axis=1)]
#this is to drop NaN values
df.dropna(axis=0,how='any', inplace=True)
#save the file to CSV
df.to_csv("English_cleaned.csv",index=False)
#English_cleaned.csv will also be uploaded in the github repository


# # main_classifier and main_grade_assign 
# ### are the only two main functions that you need to combine in the pipeline.

# In[5]:

#this is a sample classifier based upon the cleaned english datasets
def main_classifier(text):
#if you just want to load the dataframe and see results then call this
#just make sure that English_clean.csv is in the same directory you are in
    df = pd.read_csv("English_cleaned.csv", index_col=False)
    pipeline = Pipeline([
        ('vectorizer',  CountVectorizer()),
        ('classifier',  MultinomialNB()) ])
    pipeline.fit(df['EssayText'].values, df['Score1'].values)
    return pipeline.predict(text)[0] # produce predicted score


# In[6]:

#give vocabulary grade level
def main_grade_assign(text):
    return textstat.automated_readability_index(text)


# ## The rest of the code below is just to see the accuracy of different classifiers we tried

# In[39]:

#pipeline
pipeline = Pipeline([
        ('vectorizer',  CountVectorizer()),
        ('tfidf_transformer',  TfidfTransformer()),
        ('classifier',  RandomForestClassifier(n_estimators=100)) ])


# In[41]:

#resutls with confusion matrix using kfolds

k_fold = KFold(n=len(df), n_folds=6)
scores = []
confusion = numpy.array([[0,0,0], [0,0,0], [0,0,0]])
for train_indices, test_indices in k_fold:
    train_text = df.iloc[train_indices]['EssayText'].values
    train_y = df.iloc[train_indices]['Score1'].values
    test_text = df.iloc[test_indices]['EssayText'].values
    test_y = df.iloc[test_indices]['Score1'].values
    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)
    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, labels=['0','1','2'],average='macro')
    scores.append(score)
print('Total emails classified:', len(df))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)


# In[40]:

#pretty print for table of confusion matrix with no kfolds
train,test,tr_y,te_y = train_test_split(df['EssayText'],df['Score1'], test_size = 0.2)
pipeline.fit(train, tr_y)
predictions = pipeline.predict(test)
vals = [0,1,2]
y_actu = pd.Categorical(te_y, categories=vals)
y_actu = pd.Series(y_actu,name="Actual")
y_pred = pd.Categorical(predictions, categories=vals)
y_pred = pd.Series(y_pred,name="Prediction")
print('Accuracy Score:', accuracy_score(y_actu, y_pred))
pd.crosstab(y_actu, y_pred,margins = True)


# In[25]:



