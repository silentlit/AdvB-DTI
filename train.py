import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import math

# CUDA_VISIBLE_DEVICES="-1"
d_sim_func = 1
t_sim_func = 1
alpha = 0.8
lambda_g = 1
gamma = 1
lambda_c = 1.25
start_learning_rate = 0.06
eps = 0.1
adv_reg = 0.3
factor = 25
# k_nn = 4
k_fold = 5
calc_top_k = 15
mini_batch_size = 64

cell = "VCAP"
want_to_show = "%s" %(cell)
save_path = "./sess_saver/0403_%d_UAC_sort_sr%.2f_adv%.2f_eps%.2f_d%d_t%d_fold%d/%s"%(factor, lambda_c, adv_reg, eps, d_sim_func, t_sim_func, k_fold, cell)
data_path = "../after_merge/{ce}/".format(ce=cell)
# train_data = np.load("./train_labels.npy", allow_pickle=True)
all_data = np.load("./train_data.npy", allow_pickle=True)
# test_data = np.load("./sort_test_data.npy", allow_pickle=True)
d_similarity_matrix = np.load("./d_similarity_matrix.npy", allow_pickle=True)
t_similarity_matrix = np.load("./t_similarity_matrix.npy", allow_pickle=True)
# Y = np.load(data_path + "new_label.npy", allow_pickle=True)
Y = np.load("d_t_pn_matrix.npy", allow_pickle=True)
# Y = Y.T
# d_t_matrix = Y.T
d_t_matrix = Y
d_positive_target = np.load("./d_p_t.npy", allow_pickle=True)
d_negative_target = np.load("./d_tn.npy", allow_pickle=True)
# d_t_matrix = np.load("./d_t_matrix.npy", allow_pickle=True)
print(Y.shape)
print("train#{c}".format(c=all_data.shape[0]))

if not os.path.exists(save_path):
    os.makedirs(save_path)
print("save path:{p}".format(p=save_path))
# f = open(save_path + "/%d%d_%d_%.2f_hr_10.txt"%(d_sim_func, t_sim_func, factor, lambda_c), 'a+')
# f.write("hit rate top 10:\n")
# f.write("sr:{sr}, lr:{lr}, eps:{eps}, adv_reg:{adr}, df:{d}, tf:{t}, factor:{fac}, k:{k}\n".format(sr=lambda_c, lr=start_learning_rate, adr=adv_reg, eps=eps, d=d_sim_func, t=t_sim_func, fac=factor, k=k_nn))
# f.close()


def random_train_batches(example_n, size=512):
    permutation = list(np.random.permutation(example_n))
    num_complete_mini_batches = math.floor(example_n / size)

    mini_batches = []
    for k in range(num_complete_mini_batches):
        mini_batch = permutation[size * k: size * (k + 1)]
        mini_batches.append(mini_batch)
    if example_n % size != 0:
        mini_batch = permutation[num_complete_mini_batches * size:]
        mini_batches.append(mini_batch)
    return mini_batches


def calc_sr(elems):
    d_idx, t_i, t_j = elems

    d_sim_lookup = tf.cast(x=tf.nn.embedding_lookup(d_similarity_matrix, d_idx), dtype=tf.float32)
    d_U_lookup = tf.nn.embedding_lookup(drug_U, d_idx)
    t_sim_lookup_i = tf.cast(tf.nn.embedding_lookup(t_similarity_matrix, t_i), dtype=tf.float32)
    t_V_lookup_i = tf.nn.embedding_lookup(target_V, t_i)
    t_sim_lookup_j = tf.cast(tf.nn.embedding_lookup(t_similarity_matrix, t_j), dtype=tf.float32)
    t_V_lookup_j = tf.nn.embedding_lookup(target_V, t_j)
    # b_lookup_i = tf.nn.embedding_lookup(bias, t_i)
    # b_lookup_j = tf.nn.embedding_lookup(bias, t_j)
    # print(d_sim_lookup, d_U_lookup, t_sim_lookup_i, t_V_lookup_i, t_sim_lookup_j, t_V_lookup_j, b_lookup_i, b_lookup_j)

    sr1 = tf.reduce_mean(tf.square(
        d_sim_lookup - tf.exp(-gamma * tf.reduce_sum(tf.square(drug_U - d_U_lookup), 1))))
    sr2 = tf.reduce_mean(tf.square(
        t_sim_lookup_i - tf.exp(-gamma * tf.reduce_sum(tf.square(target_V - t_V_lookup_i), 1))))
    sr3 = tf.reduce_mean(tf.square(
        t_sim_lookup_j - tf.exp(-gamma * tf.reduce_sum(tf.square(target_V - t_V_lookup_j), 1))))
    # i是向量 查询到的矩阵 维度不对
    sr_d_i_j = lambda_c * (sr1 + sr2 + sr3)
    return sr_d_i_j


def calc_adv(elems):
    d_idx, t_i, t_j = elems

    d_U_lookup = tf.nn.embedding_lookup(drug_U, d_idx)
    t_V_lookup_i = tf.nn.embedding_lookup(target_V, t_i)
    t_V_lookup_j = tf.nn.embedding_lookup(target_V, t_j)
    # b_lookup_i = tf.nn.embedding_lookup(bias, t_i)
    # b_lookup_j = tf.nn.embedding_lookup(bias, t_j)
    delta_u = tf.nn.embedding_lookup(delta_U, d_idx)
    delta_v_i = tf.nn.embedding_lookup(delta_V, t_i)
    delta_v_j = tf.nn.embedding_lookup(delta_V, t_j)

    d_plus = d_U_lookup + delta_u
    t_i_plus = t_V_lookup_i + delta_v_i
    t_j_plus = t_V_lookup_j + delta_v_j

    # delta_adv = ((tf.reduce_mean(tf.multiply(d_plus, t_i_plus)) + b_lookup_i) -
    #              (tf.reduce_mean(tf.multiply(d_plus, t_j_plus)) + b_lookup_j))
    delta_adv = ((tf.reduce_mean(tf.multiply(d_plus, t_i_plus))) -
                 (tf.reduce_mean(tf.multiply(d_plus, t_j_plus))))
    return delta_adv


def calc_f1(elems):
    d_idx, t_i, t_j = elems

    d_U_lookup = tf.nn.embedding_lookup(drug_U, d_idx)
    t_V_lookup_i = tf.nn.embedding_lookup(target_V, t_i)
    t_V_lookup_j = tf.nn.embedding_lookup(target_V, t_j)
    # b_lookup_i = tf.nn.embedding_lookup(bias, t_i)
    # b_lookup_j = tf.nn.embedding_lookup(bias, t_j)
    # w_d = tf.cast(tf.nn.embedding_lookup(train_data, d_idx), dtype=tf.float32)
    # w_d = tf.nn.embedding_lookup(w_d, 3)
    # w_t = tf.cast(tf.nn.embedding_lookup(train_data, d_idx), dtype=tf.float32)
    # w_t = tf.nn.embedding_lookup(w_t, 4)

    # delta_i_j = ((tf.reduce_sum(tf.multiply(d_U_lookup, t_V_lookup_i)) + b_lookup_i) -
    #              (tf.reduce_sum(tf.multiply(d_U_lookup, t_V_lookup_j)) + b_lookup_j))
    delta_i_j = ((tf.reduce_sum(tf.multiply(d_U_lookup, t_V_lookup_i))) -
                 (tf.reduce_sum(tf.multiply(d_U_lookup, t_V_lookup_j))))

    # delta_i_j = tf.reduce_sum(tf.multiply(d_U_lookup, t_V_lookup_i)) - tf.reduce_sum(tf.multiply(d_U_lookup, t_V_lookup_j))

    # w_delta = ((1 - alpha) * w_d + alpha * w_t) * delta_i_j
    # return w_delta, delta_i_j
    return delta_i_j, delta_i_j


def init_cost():
    d_array = tf.placeholder(tf.int32, [None])
    i_array = tf.placeholder(tf.int32, [None])
    j_array = tf.placeholder(tf.int32, [None])
    # w_d = tf.placeholder(tf.float32, [None])
    # w_t = tf.placeholder(tf.float32, [None])

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100, 0.96, staircase=True)

    # drug_U = tf.get_variable("drug_U", [520, 20], initializer=tf.random_normal_initializer(0, 0.1))
    # target_V = tf.get_variable("target_V", [363, 20], initializer=tf.random_normal_initializer(0, 0.1))
    # bias = tf.get_variable("bias", [363, 1], initializer=tf.random_normal_initializer(0, 0.1))

    # d_sim_lookup = tf.cast(x=tf.nn.embedding_lookup(d_similarity_matrix, d), dtype=tf.float32)
    # d_U_lookup = tf.nn.embedding_lookup(drug_U, d)
    # t_sim_lookup_i = tf.cast(tf.nn.embedding_lookup(t_similarity_matrix, i), dtype=tf.float32)
    # t_V_lookup_i = tf.nn.embedding_lookup(target_V, i)
    # t_sim_lookup_j = tf.cast(tf.nn.embedding_lookup(t_similarity_matrix, j), dtype=tf.float32)
    # t_V_lookup_j = tf.nn.embedding_lookup(target_V, j)
    # b_lookup_i = tf.nn.embedding_lookup(bias, i)
    # b_lookup_j = tf.nn.embedding_lookup(bias, j)
    # print(d_sim_lookup, d_U_lookup, t_sim_lookup_i, t_V_lookup_i, t_sim_lookup_j, t_V_lookup_j, b_lookup_i, b_lookup_j)

    # sr1 = tf.reduce_mean(d_sim_lookup - tf.exp(-gamma * tf.reduce_sum(tf.square([drug_U - yy for yy in d_U_lookup]), 2)))
    # sr2 = tf.reduce_mean(t_sim_lookup_i - tf.exp(-gamma * tf.reduce_sum(tf.square([target_V - yy for yy in t_V_lookup_i]), 2)))
    # sr3 = tf.reduce_mean(t_sim_lookup_j - tf.exp(-gamma * tf.reduce_sum(tf.square([target_V - yy for yy in t_V_lookup_j]), 2)))
    # i是向量 查询到的矩阵 维度不对
    # sr = lambda_c * (sr1 + sr2 + sr3)
    sr_array = tf.map_fn(fn=calc_sr, elems=(d_array, i_array, j_array), dtype=tf.float32)
    f2_sum = tf.reduce_sum(sr_array)

    # f1 = ((1 - alpha) * w_d + alpha * w_t) * \
    #      (tf.matmul(d_U_lookup, tf.transpose(t_V_lookup_i)) + b_lookup_i -
    #       (tf.matmul(d_U_lookup, tf.transpose(t_V_lookup_j)) + b_lookup_j))
    # f1 = tf.reduce_mean(tf.log(tf.sigmoid(f1)))
    # f2 = tf.reduce_sum(tf.square(drug_U)) + tf.reduce_sum(tf.square(target_V)) + tf.reduce_sum(tf.square(bias))
    f1_array, delta_i_j = tf.map_fn(fn=calc_f1, elems=(d_array, i_array, j_array), dtype=(tf.float32, tf.float32))
    # f1_sig = tf.reduce_sum(-tf.log(tf.sigmoid(f1_array)))
    result_f1 = tf.clip_by_value(f1_array, -80.0, 1e8)
    f1_sig = tf.reduce_sum(tf.nn.softplus(-result_f1))

    adv_array = tf.map_fn(fn=calc_adv, elems=(d_array, i_array, j_array), dtype=tf.float32)
    result_adv = tf.clip_by_value(adv_array, -80.0, 1e8)
    adv_sig = tf.reduce_sum(tf.nn.softplus(-result_adv))
    # adv_sig = tf.reduce_sum(-tf.log(tf.sigmoid(adv_array)))

    # loss = (f1 + lambda_g * f2) + 0.02 * (tf.nn.l2_loss(drug_U) + tf.nn.l2_loss(target_V) + tf.nn.l2_loss(bias))
    # loss_1 = (f1_sig + lambda_g * f2_sum) + 0.1 * (
    #           tf.norm(drug_U, ord=2) + tf.norm(target_V, ord=2) + tf.norm(bias))
    loss_1 = (f1_sig + lambda_g * f2_sum) + 0.1 * (
            tf.norm(drug_U, ord=2) + tf.norm(target_V, ord=2))

    auc_set = tf.reduce_mean(tf.to_float(delta_i_j > 0))

    grad_U, grad_V = tf.gradients(f1_sig, [drug_U, target_V])
    grad_U_dense = tf.stop_gradient(grad_U)
    grad_V_dense = tf.stop_gradient(grad_V)
    update_U = delta_U.assign(tf.nn.l2_normalize(grad_U_dense, 1) * eps)
    update_V = delta_V.assign(tf.nn.l2_normalize(grad_V_dense, 1) * eps)

    loss_total = loss_1 + adv_reg * adv_sig

    train_op_set = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_total)

    return d_array, i_array, j_array, loss_total, train_op_set, f1_sig, adv_sig, sr_array, auc_set, update_U, update_V


def get_train_test_data(cv_k, k=5):
    if cv_k == 0:
        np.random.shuffle(all_data)
    sp_size = int(all_data.shape[0] / k)
    cv_list = list(range((cv_k * sp_size), (cv_k + 1) * sp_size))
    train_list = list(range(cv_k * sp_size)) + list(range((cv_k + 1) * sp_size, all_data.shape[0]))
    return all_data[train_list], all_data[cv_list]


# d = tf.placeholder(tf.int32, [None])
# i = tf.placeholder(tf.int32, [None])
# j = tf.placeholder(tf.int32, [None])
# w_d = tf.placeholder(tf.float32, [None])
# w_t = tf.placeholder(tf.float32, [None])

# drug_U = tf.get_variable("drug_U", [520, 20], initializer=tf.random_normal_initializer(0, 0.1))
# target_V = tf.get_variable("target_V", [363, 20], initializer=tf.random_normal_initializer(0, 0.1))
# bias = tf.get_variable("bias", [363, 1], initializer=tf.random_normal_initializer(0, 0.1))

# sr1 = tf.reduce_mean(d_similarity_matrix[d] - tf.exp(-gamma * tf.reduce_sum(tf.square(drug_U - drug_U[d]), 1)))
# sr2 = tf.reduce_mean(t_similarity_matrix[i] - tf.exp(-gamma * tf.reduce_sum(tf.square(target_V - target_V[i]), 1)))
# sr3 = tf.reduce_mean(t_similarity_matrix[j] - tf.exp(-gamma * tf.reduce_sum(tf.square(target_V - target_V[j]), 1)))
# sr = lambda_c * (sr1 + sr2 + sr3)
#
# f1 = ((1 - alpha) * w_d + alpha * w_t) * \
#      (tf.matmul(drug_U[d], tf.transpose(target_V[i])) + bias[i] -
#       (tf.matmul(drug_U[d], tf.transpose(target_V[j])) + bias[j]))
# f1 = tf.reduce_mean(tf.log(tf.sigmoid(f1)))
# f2 = tf.reduce_mean(tf.square(drug_U)) + tf.reduce_mean(tf.square(target_V)) \
#      + tf.reduce_mean(tf.square(bias))
# f2 += sr
# loss = f1 + lambda_g * f2

# global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100, 0.96, staircase=True)
# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# drug_U * tf.transpose(target_V) + tf.transpose(bias) =  (520, 363) + (1, 363)
drug_U = tf.get_variable("drug_U", [Y.shape[0], factor], initializer=tf.random_normal_initializer(0, 0.1))
target_V = tf.get_variable("target_V", [Y.shape[1], factor], initializer=tf.random_normal_initializer(0, 0.1))
# bias = tf.get_variable("bias", [Y.shape[1], 1], initializer=tf.random_normal_initializer(0, 0.1))
delta_U = tf.Variable(tf.zeros(shape=[Y.shape[0], factor]), name='delta_U', dtype=tf.float32, trainable=False)
delta_V = tf.Variable(tf.zeros(shape=[Y.shape[1], factor]), name='delta_V', dtype=tf.float32, trainable=False)
Y_P = tf.matmul(drug_U, tf.transpose(target_V))
# _B = tf.transpose(bias)
with tf.Session() as sess:
    d, i, j, loss, train_op, f1, f2, sr, auc, update_U, update_V = init_cost()
    for cv_k in range(k_fold):
        max_cor = 0 # 1不存模型
        sess.run(tf.global_variables_initializer())
        train_data, test_data = get_train_test_data(cv_k, k_fold)
        test_d_t_list = []
        for d_t_p in test_data:
            if (d_t_p[0], d_t_p[1]) not in test_d_t_list and d_t_matrix[d_t_p[0]][d_t_p[1]] == 1.0:
                test_d_t_list.append((d_t_p[0], d_t_p[1]))
        print("test#%d"%len(test_d_t_list))
        train_mini_batches = random_train_batches(train_data.shape[0], size=mini_batch_size)
        n = math.floor(train_data.shape[0] / mini_batch_size)
        saver = tf.train.Saver(max_to_keep=3)

        if not os.path.exists(save_path + "/%d" % cv_k):
            os.makedirs(save_path + "/%d" % cv_k)
        f = open(save_path + "/%d" % cv_k + "/%d%d_%d_%.2f_hr_10.txt" % (d_sim_func, t_sim_func, factor, lambda_c), 'a+')
        f.write("hit rate top 10:\n")
        f.write("sr:{sr}, lr:{lr}, eps:{eps}, adv_reg:{adr}, df:{d}, tf:{t}, factor:{fac}\n".format(sr=lambda_c,
                                                                                                           lr=start_learning_rate,
                                                                                                           adr=adv_reg,
                                                                                                           eps=eps,
                                                                                                           d=d_sim_func,
                                                                                                           t=t_sim_func,
                                                                                                           fac=factor,
                                                                                                           ))
        f.close()

        for epoch in range(50):
            epoch_cost = 0.
            _epoch = 0
            for batch in train_mini_batches:
                # print(len(batch))
                epoch_cost = 0.
                _, _, _train_op, _batch_cost = sess.run([update_U, update_V, train_op, loss],
                                                                       feed_dict={d: train_data[batch, 0],
                                                                                  i: train_data[batch, 1],
                                                                                  j: train_data[batch, 2]})  # ,
                # w_d: train_data[batch, 3],
                # w_t: train_data[batch, 4]})
                epoch_cost += _batch_cost
                # print("epoch: {ep}, cost: {co}".format(ep=epoch, co=epoch_cost))

                # when test should split drug?
                # auc_sum = 0.0
                # auc_list = []
                # for test_labels in test_data:
                #     _auc = sess.run(auc,
                #                     feed_dict={d: test_labels[:, 0],
                #                                i: test_labels[:, 1],
                #                                j: test_labels[:, 2]})
                #     auc_sum += _auc
                #     auc_list.append(_auc)

                _auc = sess.run(auc,
                                feed_dict={d: test_data[:, 0],
                                           i: test_data[:, 1],
                                           j: test_data[:, 2]})

                _epoch = _epoch + 1
                # Y_P = tf.matmul(drug_U, tf.transpose(target_V))
                # _b = tf.transpose(bias)
                # _y_p, _b = sess.run([Y_P, _B])
                _y_p = sess.run(Y_P)
                # _y_p = sess.run(Y_P)
                # y_p = np.add(_y_p, _b)
                y_p = _y_p
                # Y_P = sess.run(tf.argmax(Y_P, 1))
                # cor = 0
                ap_list = []
                aupr_list = []
                # top_10_list = []
                # precision_list = []
                test_d_t_list_count = 0
                d_top_k_array = []
                # d_top_k_array = np.zeros(shape=(d_t_matrix.shape[0], calc_top_k))
                for n_d, n_d_row in enumerate(y_p):
                    if np.sum(d_t_matrix[n_d]) == 0.0:
                        continue
                    else:
                        top_k_array = np.zeros(shape=15)
                        # n_find = np.sum(Y[n_d])
                        # top_10 = 0.0
                        top_k = sorted(range(len(n_d_row)), key=lambda kk: n_d_row[kk], reverse=True)
                        # top_k = top_k[: np.sum(Y[n_d])]
                        ap = []
                        # ar = []
                        count_k = 0
                        valid_index = 0
                        valid_t_list = d_positive_target[n_d] + list(d_negative_target)
                        t_p_size = len(d_positive_target[n_d])
                        for k_i in top_k:
                            if k_i in valid_t_list:
                                valid_index += 1
                                if d_t_matrix[n_d][k_i] == 1.0:
                                    count_k += 1
                                    ap.append(1.0 * count_k / valid_index)
                                    # ar.append(1.0 * count_k / t_p_size)
                                    if valid_index <= 10 and (n_d, k_i) in test_d_t_list:
                                        test_d_t_list_count += 1
                                if valid_index <= calc_top_k:
                                    top_k_array[valid_index - 1] = 1.0 * count_k / t_p_size
                                    # d_top_k_array[n_d][valid_index - 1] = (count_k * 1.0) / t_p_size
                                if valid_index > calc_top_k and count_k == t_p_size:
                                    break
                            # else:
                            #     continue
                            # if d_t_matrix[n_d][k_i] == 1.0:
                            #     count_k += 1
                            #     ap.append(1.0 * count_k / valid_index)
                            #     if valid_index <= 10 and (n_d, k_i) in test_d_t_list:
                            #         test_d_t_list_count += 1
                            # if valid_index <= calc_top_k:
                            #     d_top_k_array[n_d][valid_index - 1] = (count_k * 1.0) / t_p_size
                                # top_10 = count_k * 1.0
                                # if count_k == n_find:
                                # break
                        # ap.append(1.0 * count_k / (np.sum(Y[n_d])))
                        ap_list.append(np.mean(ap))
                        d_top_k_array.append(top_k_array)
                        # if len(ap) != 0:
                        #     _map = np.sum(ap) / len(ap)
                        #     ap_list.append(_map)
                        # top_list.append(top_10 / min(count_k, 10))
                        # top_10_list.append(top_10 / count_k)
                        # precision_list.append(top_10 / 10.0)


                    # if Y[n_d][n_i] == 1.0:
                    #     cor += 1

                m_auc = _auc
                np.save("./d_top_k_array", d_top_k_array)
                d_top_k_array = np.array(d_top_k_array)
                d_top_k_array = np.mean(d_top_k_array, 0)
                top_k_str = ''
                for idx, top_k_value in enumerate(d_top_k_array):
                    top_k_str += "top_%d:%.4f " % (idx + 1, top_k_value)
                # m_recall = np.mean(top_10_list)
                # m_precision = np.mean(precision_list)
                # m_f = 2.0 * m_recall * m_precision / (m_recall + m_precision)
                # m_auc = auc_sum / len(auc_list)
                # print(auc_list)
                # print(ap_list)
                # print("e%d-%d"%(epoch, _epoch), "fold-%d"%cv_k, "auc:%.4f"%m_auc, "mAP:%.4f"%np.mean(ap_list), "top_10:%.4f"%np.mean(top_10_list), "m_F1:%.4f"%m_f, "top:%.4f"%(test_d_t_list_count / len(test_d_t_list)), want_to_show)
                print("e%d-%d" % (epoch, _epoch), "fold-%d" % cv_k, "auc:%.4f" % m_auc, "mAP:%.4f" % np.mean(ap_list),
                      "top_10:%.4f" % np.mean(d_top_k_array[9]),
                      "top:%.4f" % (test_d_t_list_count / len(test_d_t_list)), want_to_show)
                print(top_k_str)
                epo_file = open(save_path + "/%d" % cv_k + "/%d%d_%d_%.2f_epo.txt" % (d_sim_func, t_sim_func, factor, lambda_c), 'a+')
                # epo_file.write("e{ep}-{_e} {hr} {au} {ap} {f} {t}\n".format(ep=epoch, _e=_epoch, hr=np.mean(top_10_list), au=m_auc, ap=np.mean(ap_list), f=m_f, t=(test_d_t_list_count / len(test_d_t_list))))
                epo_file.write(
                    "e{ep}-{_e} {hr} {au} {ap} {t} {ks}\n".format(ep=epoch, _e=_epoch, hr=np.mean(d_top_k_array[9]),
                                                                  au=m_auc,
                                                                  ap=np.mean(ap_list),
                                                                  t=(test_d_t_list_count / len(test_d_t_list)),
                                                                  ks= top_k_str))
                epo_file.close()
                # print("f1:", _f1, ", f2:", _f2, "correct:", cor, "auc:", auc_sum / len(auc_list)) # , ", sr: ", _sr)
                # print(auc_list)
                if np.mean(ap_list) > max_cor:
                    max_cor = np.mean(ap_list)
                #   saver.save(sess, save_path + "/%d" % cv_k + "/sess_saver")
                    np.save(save_path + "/%d" % cv_k + "/Y.npy", _y_p)
                #     f = open(save_path + "/%d" % cv_k + "/%d%d_%d_%.2f_hr_10.txt"%(d_sim_func, t_sim_func, factor, lambda_c), 'a+')
                #     f.write("e{ep}-{_e} {hr}\n".format(ep=epoch, _e=_epoch, hr=np.mean(top_10_list)))
                #     f.close()
            print("epoch: {e_t}, cost: {avg_cost}".format(e_t=epoch, avg_cost=np.sum(epoch_cost) / n))
