import numpy as np

######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    if n_cat1 == 0 or n_cat2 == 0:
        return 0
    val_n_cat1 = n_cat1 / (n_cat1 + n_cat2)
    val_n_cat2 = n_cat2 / (n_cat1 + n_cat2)
    return -val_n_cat1 * np.log2(val_n_cat1) - val_n_cat2 * np.log2(val_n_cat2)

###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(features: list, labels: list) -> (int, int):
    best_separation_feature, best_separation_value = 0, 0
    min_ent = float('inf')

    for fi in range(len(features[0])):
        features_sublist = [s[fi] for s in features]
        distinct = set(features_sublist)
        for fv in distinct:
            smaller = [labels[i] for i in range(len(labels)) if features[i][fi] <= fv]
            bigger = [labels[i] for i in range(len(labels)) if features[i][fi] > fv]
            ent = (len(smaller) / len(labels)) * get_entropy(smaller.count(0), smaller.count(1)) + \
                  (len(bigger) / len(labels)) * get_entropy(bigger.count(0), bigger.count(1))
            if ent < min_ent:
                min_ent = ent
                best_separation_feature = fi
                best_separation_value = fv
    return best_separation_feature, best_separation_value

################### 3. feladat, döntési fa implementációja ####################
def main():
    tdat = np.loadtxt('train.csv', delimiter=',')
    tlab = tdat[:, 8]
    tfea = tdat[:, :7]
    tree = tree_train(tfea, tlab)
    cdat = np.loadtxt('test.csv', delimiter=',')
    results = []
    for i in cdat:
        results.append(evaluate(tree, i))
    np.savetxt('results.csv', results, fmt='%d')
    return 0

def tree_train(features: list, labels: list):
    if len(set(labels)) == 1: return labels[0]
    if len(features[0]) == 0: return max(set(labels), key=labels.count)
    bv = get_best_separation(features, labels)
    lf = []
    for i in features:
        if i[bv[0]] <= bv[1]:
            lf.append(i)
    rf = []
    for i in features:
        if i[bv[0]] > bv[1]:
            rf.append(i)
    ll = []
    for i in range(len(labels)):
        if features[i][bv[0]] <= bv[1]:
            ll.append(labels[i])
    rl = []
    for i in range(len(labels)):
        if features[i][bv[0]] > bv[1]:
            rl.append(labels[i])
    return bv[0], bv[1], tree_train(lf, ll), tree_train(rf, rl)

def evaluate(tree, value) -> int:
    if tree == 0 or tree == 1:
        return tree;
    if value[tree[0]] <= tree[1]:
        return evaluate(tree[2], value)
    else:
        return evaluate(tree[3], value)

if __name__ == "__main__":
    main()