import numpy as np
import os


# 1
def tanimoto(x, y):
    f1 = np.dot(x, y)
    f2 = np.sum(np.square(x)) + np.sum(np.square(y)) - f1
    return f1 / f2


# 2
def cos_sim(x, y):
    f1 = np.dot(x, y)
    f2 = np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y)))
    return f1/ f2


# 3
def ssim(x, y):
    f1 = np.mean(x) * np.mean(y)
    f2 = np.square(np.mean(x)) + np.square(np.mean(y))
    f3 = np.dot((x - np.mean(x)), (y - np.mean(y))) / len(x)
    f4 = np.var(x) + np.var(y)
    # print(f1, f2, f3, f4)
    return 4 * f1 * f3 / f2 / f4

# 4
def spearman(x, y):
    sort_x_index = sorted(range(len(x)), key=lambda i: x[i], reverse=True)
    sort_y_index = sorted(range(len(y)), key=lambda i: y[i], reverse=True)
    f1 = np.sum(np.square(np.array(sort_x_index) - np.array(sort_y_index)))
    return 1 - 6 * f1 / (len(x) * (np.square(len(x)) - 1))


def positive_dt():
    dt_matrix = np.load(data_path + "new_label.npy")
    dt_matrix = dt_matrix.T
    drug = np.load(data_path + "drug_data.npy")
    target_p = np.load(data_path + "tp_data.npy")
    drug_target_matrix = np.zeros(shape=(len(drug), 2 * len(target_p)))
    positive_dt_tuple = []
    d_positive_target = []
    for d in range(dt_matrix.shape[0]):
        positive_target = []
        for t in range(dt_matrix.shape[1]):
            if dt_matrix[d][t] == 1.0:
                positive_dt_tuple.append((d, t))
                positive_target.append(t)
                drug_target_matrix[d][t] = 1
        d_positive_target.append(positive_target) 
    np.save("d_tp", positive_dt_tuple)
    np.save("d_t_pn_matrix", drug_target_matrix)
    np.save("d_p_t", d_positive_target)
    print("positive sample done...")


def negtive_dt():
    target_p = np.load(data_path + "tp_data.npy")
    target_n = np.load(data_path + "rngene_data.npy")
    target_feature = np.zeros(shape=(2 * len(target_p), 978))
    target_feature[:len(target_p)] = target_p
    tn_index_list = list(np.arange(len(target_n)))
    np.random.shuffle(tn_index_list)
    tn_list = []
    d_tn = []
    positive_t_index = len(target_p)
    for i in range(len(target_p)):
        t_n = tn_index_list.pop()
        tn_list.append(t_n)
        d_tn.append(i + positive_t_index)
        target_feature[i + positive_t_index] = target_n[t_n]
    np.save("tn", tn_list) 
    np.save("d_tn", d_tn) 
    np.save("target_feature", target_feature) 
    print("negative sample done...")


def similarity_matrix(check=False):
    if check:
        if os.path.exists("./d_similarity_matrix.npy"):
            return
    drug = np.load(data_path + "drug_data.npy")
    target = np.load("./target_feature.npy")

    d_s = np.zeros((drug.shape[0], drug.shape[0]))
    t_s = np.zeros((target.shape[0], target.shape[0]))

    for i in range(d_s.shape[0]):
        for j in range(d_s.shape[0]):
            if i == j:
                d_s[i][j] = 1
            elif i > j:
                d_s[i][j] = d_s[j][i]
            else:
                # 1
                d_s[i][j] = tanimoto(drug[i], drug[j])
        if i % 100 == 0:
            print("sim drug %.2f done..."%(i/d_s.shape[0]))

    for i in range(t_s.shape[0]):
        for j in range(t_s.shape[0]):
            if i == j:
                t_s[i][j] = 1
            elif i > j:
                t_s[i][j] = t_s[j][i]
            else:
                # 2
                t_s[i][j] = tanimoto(target[i], target[j])
        if i % 20 == 0:
            print("sim target %.2f done..." % (i / t_s.shape[0]))

    np.save("d_similarity_matrix", d_s)
    np.save("t_similarity_matrix", t_s)


def get_train_data(neg_size=50):
    train_data = []
    d_tp = np.load("./d_tp.npy")
    d_tn = np.load("./d_tn.npy")
    count = 0
    for dtp in d_tp:
        d, ti = dtp
        tem = d_tn
        np.random.shuffle(tem)
        tem = tem[:neg_size]
        for tj in tem:
            train_data.append((d, ti, tj))
        count += 1
        if count % 50 == 0:
            print("train data %.2f done..." % (count / d_tp.shape[0]))
    return train_data


cell = "VCAP"
data_path = "../after_merge/{ce}/".format(ce=cell)
positive_dt()
negtive_dt()
similarity_matrix()
train_data = get_train_data(2)
np.save("train_data", train_data)
