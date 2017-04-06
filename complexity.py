# function to get sentence lengths ratio to judge complexity of response
# see http://www.aims.edu/student/online-writing-lab/process/sentence-length
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
        l = len(wds)
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