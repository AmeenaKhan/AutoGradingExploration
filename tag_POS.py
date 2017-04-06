# part of speech tagging
# http://www.nltk.org/book/ch05.html
# note: Amer might be able to use the .similar() function to look for words with same POS as those found in rubric
# http://textminingonline.com/dive-into-nltk-part-iii-part-of-speech-tagging-and-pos-tagger
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