
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
import string
from nltk import word_tokenize


# In[2]:
def organization(file_path):
    fil = pd.read_csv(file_path)


# In[3]:

    df= fil.copy()


# In[4]:

    ids = df["essay_id"]
    essays = df["essay"].str.lower()
    sub_ids = df["essay_id"]
    sub_essay = df["essay"].str.lower()


# In[5]:

    from nltk import word_tokenize

    for m in range(len(sub_essay)):
        sub_essay[m] = " ".join(c for c in word_tokenize(sub_essay[m]) if c not in list(string.punctuation))


# In[6]:

    unigram_org= {"begin":1, "first":1, "firstly":1, "initially":1, "foremost":1,"conclusion":1, "conclude":1, "final":1, "finally":1, "last":1, "lastly":1, "ultimately":1, "end":1, "sum":1, "eventually":1, "so":1, "thus":1, "hence":1, "altogether":1, "summarize":1, "summary":1, "therefore":1, "overall":1, "secondly":1, "next":1, "subsequently":1, "before":1, "previously": 1, "afterwards":1, "then":1, "after":1, "so":1, "example":1, "instance":1, "because":1, "consequently":1, "consequence":1, "therefore":1, "result":1, "due":1, "rather":1, "however":1, "moreover":1, "nonetheless":1, "still":1, "yet":1, "nevertheless":1, "although":1, "though":1, "regardless":1, "despite":1, "indeed":1, "importantly":1, "besides":1, "contrast":1, "while":1, "conversely":1, "similarly":1, "likewise":1, "equally":1, "namely":1, "specifically":1, "especially":1, "particularly":1, "illustrated":1, "illustrates":1, "also":1, "and":1, "or":1, "too":1, "addition":1, "furthermore":1, "further":1, "alternatively":1}
    bigram_org ={"i think":1, "in brief":1,"in conclusion":1,"to conclude":1,"to summarize":1,"in sum":1,"in summary":1,"Above all":1,"Coupled with":1, "Whats more":1}
    trigram_org={"in order to":1,"in other words":1, "to that end":1, "as well as":1, "not to mention":1, "in the end":1, "on the whole":1, "to sum up":1, "an additional info":1}


# In[7]:

    scores={}
    for k,j in zip(ids,sub_essay):
        points = 0
        if(type(j)!=float):
            for i in j.split():
                if i in unigram_org:
                    points = points + unigram_org[i]
            scores[k] = points

    for i in range(len(sub_essay)):
        n = nltk.word_tokenize(sub_essay[i])
        bi = ngrams(n,2)
        tri = ngrams(n,3)
        bigrams = ['  '.join(j) for j in list(bi)]
        trigrams =  ['  '.join(j) for j in list(tri)]

            
            


# In[8]:


    for k,j in zip(ids,sub_essay):
        points1=0
        if(type(j)!=float):
            for i in j.split():
                if i in bigram_org:
                    points1 = points1 + bigram_org[i]
            scores[k] += points1
    


# In[9]:

    for k,j in zip(ids,sub_essay):
        points2=0
        if(type(j)!=float):
            for i in j.split():
                if i in trigram_org:
                    points2 = points2 + trigram_org[i]
            scores[k] += points2
    


# In[10]:

    dataframe = pd.DataFrame(list(scores.items()))
    dataframe[2] = (dataframe[1] - dataframe[1].mean()) / (dataframe[1].max() - dataframe[1].min())
    bins = np.linspace(dataframe[2].min(), dataframe[2].max(), 4)



# In[19]:

    low=[]
    mid=[]
    high=[]
    def grader(x):
    
        if x <= bins[1]:
            low.append(x)
            return 'AVERAGE'
        
        elif x > bins[1] or x < bins[2]:
            mid.append(x)
            return 'GOOD'
        
        elif x > bins[2]:
            high.append(x)
            return '2GOOD'
        


# In[20]:

    dataframe["GRADE"] = None
    dataframe["GRADE"] = dataframe[2].map(grader)
return[high,medium, low]



