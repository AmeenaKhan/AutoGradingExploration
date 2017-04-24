import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.util import ngrams
import string
from nltk import word_tokenize

#Parameter: file path as a string
#Returns: brackets of high, medium, low levels of occurance in an array of three dictionaries

def organization(file_path):
  df = pd.read_csv(file_path)
  ids = df["essay_id"]
  
  #PRE-PROCESSING
  essays = df["essay"].str.lower()
  #removing puctuation 
  m=0
  for m in range(len(essays)):
    essays[m] = " ".join(c for c in word_tokenize(essays[m]) if c not in list(string.punctuation))
  
  #organization words (n
  unigram_org= {"begin":1, "first":1, "firstly":1, "initially":1, "foremost":1,"conclusion":1, "conclude":1, "final":1, "finally":1, "last":1, "lastly":1, "ultimately":1, "end":1, "sum":1, "eventually":1, "so":1, "thus":1, "hence":1, "altogether":1, "summarize":1, "summary":1, "therefore":1, "overall":1, "secondly":1, "next":1, "subsequently":1, "before":1, "previously": 1, "afterwards":1, "then":1, "after":1, "so":1, "example":1, "instance":1, "because":1, "consequently":1, "consequence":1, "therefore":1, "result":1, "due":1, "rather":1, "however":1, "moreover":1, "nonetheless":1, "still":1, "yet":1, "nevertheless":1, "although":1, "though":1, "regardless":1, "despite":1, "indeed":1, "importantly":1, "besides":1, "contrast":1, "while":1, "conversely":1, "similarly":1, "likewise":1, "equally":1, "namely":1, "specifically":1, "especially":1, "particularly":1, "illustrated":1, "illustrates":1, "also":1, "and":1, "or":1, "too":1, "addition":1, "furthermore":1, "further":1, "alternatively":1}
  bigram_org ={"i think":1, "in brief":1,"in conclusion":1,"to conclude":1,"to summarize":1,"in sum":1,"in summary":1,"Above all":1,"Coupled with":1, "Whats more":1}
  trigram_org={"in order to":1,"in other words":1, "to that end":1, "as well as":1, "not to mention":1, "in the end":1, "on the whole":1, "to sum up":1, "an additional info":1}
  
  #calculating number of unigram org words
  for k,j in zip(ids,essays):
    points = 0
    for i in j.split():
        if i in org_words:
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
    for x in bigrams:
      if x in addition2:
        points = points + bigram_org[x]
    scores[k] = scores[k] + points
    
    #calculating number of trigram org words
    points= 0
    for z in trigrams:
      if z in addition3:
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
    elif scores[k] < 0.75:
        medium[k] = scores[k]
    else:
        high[k] = scores[k]
        
    
  return [high, medium, low]
