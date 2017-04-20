def split_by_set(features):
    """Returns a dictionary of sets mapped to a dictionary of essays in the set mapped to a dictionary of their features"""
    set = {}
    for k in features:
        #print(features[k])
        qset = features[k]['set']
        #print(k, " is in set ", qset)
        d = {k : features[k]}
        if qset in set:
            set[qset].update(d)
        else:
            set.update({qset : d})
    return set