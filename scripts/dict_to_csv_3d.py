import csv

def dict_to_csv(feature_dict):
    for k in feature_dict:
        filename = 'long_set' + str(k) + '.csv'
        with open(filename,'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["essay_id", "set", "score", "org_score", "pos_adjadv", "pos_noun", "pos_pronoun", "pos_verb", "pos_other", "pos_nums", "complex_short", "complex_medium", "complex_long", "vocab_level"])
            for key in feature_dict[k]:
                row = []
                row.append(key)
                row.append(k)
                row.append(feature_dict[k][key]["score"])
                row.append(feature_dict[k][key]["organization"])
                row.append(feature_dict[k][key]["pos_adjadv"])
                row.append(feature_dict[k][key]["pos_noun"])
                row.append(feature_dict[k][key]["pos_pronoun"])
                row.append(feature_dict[k][key]["pos_verb"])
                row.append(feature_dict[k][key]["pos_other"])
                row.append(feature_dict[k][key]["pos_nums"])
                row.append(feature_dict[k][key]["complex_short"])
                row.append(feature_dict[k][key]["complex_medium"])
                row.append(feature_dict[k][key]["complex_long"])
                row.append(feature_dict[k][key]["vocab_level"])
                row.append(feature_dict[k][key]["predicted_score"])
                writer.writerow(row)
    
    csvfile.close()