import csv

def dict_to_csv(feature_dict):
	with open('features.csv','w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #init_row = ["essay_id", "org_score", "pos_score", "complex_score"]
        writer.writerow([["essay_id","org_score", "pos_adjadv" , "pos_noun", "pos_pronoun","pos_verb","pos_other", "pos_nums", "complex_short","complex_medium","complex_long"]])
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
            row = [row]
            print(row)
            writer.writerow(row)
