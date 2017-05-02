def essay_length(ids,essays,final_features):
    for m in range(len(essays)):
        essays[m] = " ".join(c for c in word_tokenize(essays[m]) if c not in list(string.punctuation))
    
    for k,j in zip(ids,essays):
        length = 0
        for x in j.split():
            length += 1 
        final_features[k]["total_length"]=length
