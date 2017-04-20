
# coding: utf-8

# In[ ]:

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

import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
import string
from nltk import word_tokenize

def organization(ids,essays,final_features):
    ##df = pd.read_csv(file_path)
    ##ids = df["essay_id"]
  
    #PRE-PROCESSING
    ##essays = df["essay"].str.lower()
    #removing puctuation 
    m=0
    for m in range(len(essays)):
        essays[m] = " ".join(c for c in word_tokenize(essays[m]) if c not in list(string.punctuation))
  
    #organization words (n
    unigram_org= {"begin":1, "first":1, "firstly":1, "initially":1, "foremost":1,"conclusion":1, "conclude":1, "final":1, "finally":1, "last":1, "lastly":1, "ultimately":1, "end":1, "sum":1, "eventually":1, "so":1, "thus":1, "hence":1, "altogether":1, "summarize":1, "summary":1, "therefore":1, "overall":1, "secondly":1, "next":1, "subsequently":1, "before":1, "previously": 1, "afterwards":1, "then":1, "after":1, "so":1, "example":1, "instance":1, "because":1, "consequently":1, "consequence":1, "therefore":1, "result":1, "due":1, "rather":1, "however":1, "moreover":1, "nonetheless":1, "still":1, "yet":1, "nevertheless":1, "although":1, "though":1, "regardless":1, "despite":1, "indeed":1, "importantly":1, "besides":1, "contrast":1, "while":1, "conversely":1, "similarly":1, "likewise":1, "equally":1, "namely":1, "specifically":1, "especially":1, "particularly":1, "illustrated":1, "illustrates":1, "also":1, "and":1, "or":1, "too":1, "addition":1, "furthermore":1, "further":1, "alternatively":1}
    bigram_org ={"i think":1, "in brief":1,"in conclusion":1,"to conclude":1,"to summarize":1,"in sum":1,"in summary":1,"Above all":1,"Coupled with":1, "Whats more":1}
    trigram_org={"in order to":1,"in other words":1, "to that end":1, "as well as":1, "not to mention":1, "in the end":1, "on the whole":1, "to sum up":1, "an additional info":1}

    scores={}
    #calculating number of unigram org words
    for k,j in zip(ids,essays):
        points = 0
        if(type(j)!=float):
            for i in j.split():
                if i in unigram_org:
                    points = points + unigram_org[i]
        scores[k] = points

    #creating the bigrams and trigrams 
    for k,j in zip(ids,essays):
        bigrams = []
        trigrams = []
        n = nltk.word_tokenize(j)
        bi = ngrams(n,2)
        tri = ngrams(n,3)
        bigrams=[' '.join(i) for i in list(bi)]
        trigrams=[' '.join(i) for i in list(tri)]

    #calculating number of bigram org words
        points= 0
        if(type(j)!=float):
            for x in bigrams:
                if x in bigram_org:
                    points = points + bigram_org[x]
        scores[k] = scores[k] + points
    
    #calculating number of trigram org words
        points= 0
        if(type(j)!=float):
            for z in trigrams:
                if z in trigram_org:
                    points = points + trigram_org[z]
        scores[k] = scores[k] + points
   
    high = {} #above 75%
    medium = {} #25-75%
    low = {} #less than 25%
   
    for k,j in zip(ids,essays):
        essay_len = len(j)
        scores[k] = scores[k]/essay_len
        if scores[k] >= 0.25:
            low[k] = scores[k]
            final_features[k]["organization"] = 0
        elif scores[k] < 0.75:
            medium[k] = scores[k]
            final_features[k]["organization"] = 1
        else:
            high[k] = scores[k]
            final_features[k]["organization"] = 2

def tag_POS(essay_text):
    """Tags each word in the essay with its part of speech. The ratios of verbs, nouns, and adjectives may be a useful
    feature - for instance, questions that ask students to 'describe' things may use more adjectives, while questions
    that ask students to 'list' things may use more nouns. We can use either the direct counts or the ratios as features.
    The method returns the ratios (# NN|VB|JJ / total words).
    Note: Do this before removing stop words."""
    #print(essay_text)
    
    # split essay text into words
    wds = nltk.tokenize.word_tokenize(essay_text)
    #total_wds = len(wds) # this is not the true length, as it may include some tagged puntuation
    #print(wds)
    #print("Total words: ", total_wds)
    
    # tag POS
    tagged = nltk.pos_tag(wds)
    #print(tagged)
    
    # counters
    adj_advb = 0 # JJ, JJR, JJS are adjectives/descriptors; RB, RBR, RBS are adverbs (also descriptors)
    nn = 0 # NN,NNS, NNP, NNPS are nouns (both proper and common, singular and plural)
    pn = 0 # just proper nouns
    vb = 0 # VB, VBD, VBG, VBN, VBP, VBZ are verbs in various tenses
    nums = 0 # CD: numerical values found in the text
    other = 0
    total_wds = 0
    
    # count POS and total words
    for w in tagged:
        pos = w[1]
        #print("Part of speech for ", w[0], " is ", pos)
        if (pos == 'JJ' or pos == 'JJR' or pos == 'JJS' or pos == 'RB' or pos == 'RBR' or pos == 'RBS'):
            # adjective or adverb
            adj_advb += 1
            total_wds += 1
        elif (pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS'):
            # common and proper nouns
            nn += 1
            total_wds += 1
            if (pos == 'NNP' or pos == 'NNPS'):
                # proper nouns only
                pn += 1
        elif(pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ'):
            # all verb forms
            vb += 1
            total_wds += 1
        elif(pos == 'CD'):
            # numerical values like years and measurements, we can count these as words
            nums += 1
            total_wds += 1
        elif(pos == '$' or pos == '.' or pos == '(' or pos == ')' or pos == "''" or pos == ',' or pos == '--'
            or pos == ',' or pos == ':' or pos == 'SYM' or pos == "``"):
            # symbols and punctuation, not counted as words
            pass # don't count anything
        else:
            # all other words
            other += 1
            total_wds += 1
            
    # ratios of descriptors, nouns, proper nouns, verbs
    adjadvb_ratio = adj_advb/total_wds
    n_ratio = nn/total_wds
    pn_ratio = pn/total_wds
    vb_ratio = vb/total_wds
    other_ratio = other/total_wds
    # also return number of numerical values found (may be more useful than ratio, particularly for questions that require some 
    #    numerical response)
    
    return [adjadvb_ratio, n_ratio, pn_ratio, vb_ratio, other_ratio, nums]

def complexity(essay_text):
    """Calculates the length of each sentence in the essay (may only be applicable to longer essay responses, 
    not short response answers). 
    Counts the number of short (<=10 words), medium (11 - 35 words), and long (>35 words) sentences.
    The ratios of these is used as a measure of style (good writing has a mix of sentence lengths), and may 
    be correlated with score of the response. These ratios (# size / total sentences) are returned as a list.
    Note: to get an accurate sentence count, this function should be called before removing stop words."""
    # BEFORE removing stop words
    #print(essay_text)
    
    # counters for sentence lengths
    short = 0
    med = 0
    long = 0
    
    # split essay text into sentences
    sents = nltk.tokenize.sent_tokenize(essay_text)
    #print(sents)
    
    # for each sentence, count number of words and increment appropriate length counter
    for s in sents:
        #print("Sentence: ", s)
        wds = nltk.tokenize.word_tokenize(s)
        #print("Word list: ", wds)
        l = 0 # store sentence length
        for i in wds:
            if i not in list(string.punctuation):
                l += 1
                
        #print("Length=", l)
        if l <= 10:
            short += 1
        elif l <= 35:
            med += 1
        else:
            long += 1
    
    # results that can be used as features
    #print("This essay has:")
    #print(short, " short sentences")
    #print(med, " medium sentences")
    #print(long, " long sentences")
    
    # better essays tend to have a balanced mixture of sentence lengths
    # an equal number of each category (1/3 each) to 1/2 medium, 1/4 short, 1/4 long
    # we could either use one feature for this (a number or ratio indicating amount of balance)
    # or we could have three features, one ratio for each length
    total = short + med + long
    # return three ratios for three features
    return [short/total, med/total, long/total]

#this block of code is for preprocessing the data you only need to run this once
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

def pred_score(text):
    text = clean_Essay(text)
#if you just want to load the dataframe and see results then call this
#just make sure that English_clean.csv is in the same directory you are in
    df = pd.read_csv("English_cleaned.csv", index_col=False)
    pipeline = Pipeline([
        ('vectorizer',  CountVectorizer()),
        ('classifier',  MultinomialNB()) ])
    pipeline.fit(df['EssayText'].values, df['Score1'].values)
    return pipeline.predict(text)[0] # produce predicted score

def vocab_grade(text):
    return textstat.automated_readability_index(text)

def combined(file_path):
    df = pd.read_csv(file_path)
    ids = df["Id"]
    #PRE-PROCESSING
    essays = df["EssayText"].str.lower()
    scores = df["Score1"]
    sets = df["EssaySet"]
    
    final_features = {}
    
    print("adding given scores")
    for k,s in zip(ids,scores):
        final_features[k] = {}
        final_features[k]["score"] = s
    
    for k,s in zip(ids,sets):
        final_features[k]["set"] = s
        
    print("running org")
    org_results = organization(ids,essays,final_features)
    
    print("running pos")
    for k,j in zip(ids,essays):
        pos_score = tag_POS(j)
        final_features[k]["pos_adjadv"] = pos_score[0]
        final_features[k]["pos_noun"] = pos_score[1]
        final_features[k]["pos_pronoun"] = pos_score[2]
        final_features[k]["pos_verb"] = pos_score[3]
        final_features[k]["pos_other"] = pos_score[4]
        final_features[k]["pos_nums"] = pos_score[5]
    
    print("running complexity")
    for k,j in zip(ids,essays):
        complex_score = complexity(j)
        final_features[k]["complex_short"] = complex_score[0]
        final_features[k]["complex_medium"] = complex_score[1]
        final_features[k]["complex_long"] = complex_score[2]
    
    print("running pred_ score and vocab")
    for k,j in zip(ids,essays):
        pscore = pred_score(j)
        final_features[k]["pred_score"] = pscore
        
        vocab = vocab_grade(j)
        final_features[k]["vocab_level"] = vocab
    print("FINISHED")
    return final_features


# In[ ]:

test=combined("English_test.csv")


# In[18]:

import csv

def dict_to_csv(feature_dict):
    with open('features_all.csv','w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        #init_row = ["essay_id", "org_score", "pos_score", "complex_score"]
        writer.writerow(["essay_id","score","org_score", "pos_adjadv" , "pos_noun", "pos_pronoun","pos_verb","pos_other", "pos_nums", "complex_short","complex_medium","complex_long","predicted_score","vocab_level"])
        for key in feature_dict:
            row = []
            row.append(key)
            row.append(feature_dict[key]["score"])
            row.append(feature_dict[key]["organization"])
            row.append(feature_dict[key]["pos_adjadv"])
            row.append(feature_dict[key]["pos_noun"])
            row.append(feature_dict[key]["pos_pronoun"])
            row.append(feature_dict[key]["pos_verb"])
            row.append(feature_dict[key]["pos_other"])
            row.append(feature_dict[key]["pos_nums"])
            row.append(feature_dict[key]["complex_short"])
            row.append(feature_dict[key]["complex_medium"])
            row.append(feature_dict[key]["complex_long"])
            row.append(feature_dict[key]["pred_score"])
            row.append(feature_dict[key]["vocab_level"])
            print(row)
            writer.writerow(row)
    csvfile.close()


# In[19]:

dict_to_csv(test)


# In[13]:

pred_score("Inorder to replicate the experiment, I would need to know the mass of the four different samples, how much vinegar to pour into the containers, what the samples were made up of, and how much water to rinse them with after the 24 hours.")


# In[ ]:



