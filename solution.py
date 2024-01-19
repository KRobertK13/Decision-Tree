import numpy as np #(működik a Moodle-ben is)


######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    prob_1 = n_cat1/(n_cat1+n_cat2)
    prob_2 = n_cat2/(n_cat1+n_cat2)
    if n_cat1 == 0 or n_cat2 == 0:
        return 0
    entropy = - (prob_1 * np.log2(prob_1) + prob_2 * np.log2(prob_2))
    return entropy
###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(features: list, labels: list) -> (int, int):
    best_separation_feature, best_separation_value = 0, 0
    min = float("inf")
    #TODO számítsa ki a legjobb szeparáció tulajdonságát és értékét!
    features = np.array(features)
    for j in range(len(features[0])-1):
        actual = features[0][j]
        en = 0
        ep = 0
        fn = 0
        fp = 0
        ln = 0
        lp = 0
        for i in range(len(features)-1):
            for k in features[i]:
                if k <= actual:
                    if labels[i] == 1: 
                        ep += 1
                        lp += 1
                    else: 
                        en += 1
                        ln += 1
                else:
                    if labels[i] == 1: 
                        fp += 1
                        lp += 1
                    else: 
                        fn += 1
                        ln += 1
        l = ep+en+fp+fn
        efficiency = get_entropy(l,l) - ((ep+en) / l * get_entropy(ep, l) + (fp+fn) / l *get_entropy(fp,l))
        if efficiency < min: 
            min = efficiency
            best_separation_feature = features[0][j]
            best_separation_value = j
    return best_separation_feature, best_separation_value        
################### 3. feladat, döntési fa implementációja ####################
def main():
    #TODO: implementálja a döntési fa tanulását!
    return 0
if __name__ == "__main__":
    main()