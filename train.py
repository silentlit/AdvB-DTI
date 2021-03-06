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

    sr1 = tf.reduce_mean(tf.square(
        d_sim_lookup - tf.exp(-gamma * tf.reduce_sum(tf.square(drug_U - d_U_lookup), 1))))
    sr2 = tf.reduce_mean(tf.square(
        t_sim_lookup_i - tf.exp(-gamma * tf.reduce_sum(tf.square(target_V - t_V_lookup_i), 1))))
    sr3 = tf.reduce_mean(tf.square(
        t_sim_lookup_j - tf.exp(-gamma * tf.reduce_sum(tf.square(target_V - t_V_lookup_j), 1))))
    
    sr_d_i_j = lambda_c * (sr1 + sr2 + sr3)
    return sr_d_i_j


def calc_adv(elems):
    d_idx, t_i, t_j = elems

    d_U_lookup = tf.nn.embedding_lookup(drug_U, d_idx)
    t_V_lookup_i = tf.nn.embedding_lookup(target_V, t_i)
    t_V_lookup_j = tf.nn.embedding_lookup(target_V, t_j)

    delta_u = tf.nn.embedding_lookup(delta_U, d_idx)
    delta_v_i = tf.nn.embedding_lookup(delta_V, t_i)
    delta_v_j = tf.nn.embedding_lookup(delta_V, t_j)

    d_plus = d_U_lookup + delta_u
    t_i_plus = t_V_lookup_i + delta_v_i
    t_j_plus = t_V_lookup_j + delta_v_j

    delta_adv = ((tf.reduce_mean(tf.multiply(d_plus, t_i_plus))) -
                 (tf.reduce_mean(tf.multiply(d_plus, t_j_plus))))
    return delta_adv


def calc_f1(elems):
    d_idx, t_i, t_j = elems

    d_U_lookup = tf.nn.embedding_lookup(drug_U, d_idx)
    t_V_lookup_i = tf.nn.embedding_lookup(target_V, t_i)
    t_V_lookup_j = tf.nn.embedding_lookup(target_V, t_j)
    delta_i_j = ((tf.reduce_sum(tf.multiply(d_U_lookup, t_V_lookup_i))) -
                 (tf.reduce_sum(tf.multiply(d_U_lookup, t_V_lookup_j))))
    return delta_i_j, delta_i_j


def init_cost():
    d_array = tf.placeholder(tf.int32, [None])
    i_array = tf.placeholder(tf.int32, [None])
    j_array = tf.placeholder(tf.int32, [None])

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100, 0.96, staircase=True)

    sr_array = tf.map_fn(fn=calc_sr, elems=(d_array, i_array, j_array), dtype=tf.float32)
    f2_sum = tf.reduce_sum(sr_array)

    f1_array, delta_i_j = tf.map_fn(fn=calc_f1, elems=(d_array, i_array, j_array), dtype=(tf.float32, tf.float32))

    result_f1 = tf.clip_by_value(f1_array, -80.0, 1e8)
    f1_sig = tf.reduce_sum(tf.nn.softplus(-result_f1))

    adv_array = tf.map_fn(fn=calc_adv, elems=(d_array, i_array, j_array), dtype=tf.float32)
    result_adv = tf.clip_by_value(adv_array, -80.0, 1e8)
    adv_sig = tf.reduce_sum(tf.nn.softplus(-result_adv))

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


drug_U = tf.get_variable("drug_U", [Y.shape[0], factor], initializer=tf.random_normal_initializer(0, 0.1))
target_V = tf.get_variable("target_V", [Y.shape[1], factor], initializer=tf.random_normal_initializer(0, 0.1))
delta_U = tf.Variable(tf.zeros(shape=[Y.shape[0], factor]), name='delta_U', dtype=tf.float32, trainable=False)
delta_V = tf.Variable(tf.zeros(shape=[Y.shape[1], factor]), name='delta_V', dtype=tf.float32, trainable=False)
Y_P = tf.matmul(drug_U, tf.transpose(target_V))
with tf.Session() as sess:
    d, i, j, loss, train_op, f1, f2, sr, auc, update_U, update_V = init_cost()
    for cv_k in range(k_fold):
        max_cor = 0 
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
                epoch_cost += _batch_cost

                _auc = sess.run(auc,
                                feed_dict={d: test_data[:, 0],
                                           i: test_data[:, 1],
                                           j: test_data[:, 2]})

                _epoch = _epoch + 1
                _y_p = sess.run(Y_P)
                y_p = _y_p
                ap_list = []
                aupr_list = []
                test_d_t_list_count = 0
                d_top_k_array = []
                for n_d, n_d_row in enumerate(y_p):
                    if np.sum(d_t_matrix[n_d]) == 0.0:
                        continue
                    else:
                        top_k_array = np.zeros(shape=15)
                        top_k = sorted(range(len(n_d_row)), key=lambda kk: n_d_row[kk], reverse=True)
                        ap = []
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
                                    if valid_index <= 10 and (n_d, k_i) in test_d_t_list:
                                        test_d_t_list_count += 1
                                if valid_index <= calc_top_k:
                                    top_k_array[valid_index - 1] = 1.0 * count_k / t_p_size
                                if valid_index > calc_top_k and count_k == t_p_size:
                                    break

                        ap_list.append(np.mean(ap))
                        d_top_k_array.append(top_k_array)

                m_auc = _auc
                np.save("./d_top_k_array", d_top_k_array)
                d_top_k_array = np.array(d_top_k_array)
                d_top_k_array = np.mean(d_top_k_array, 0)
                top_k_str = ''
                for idx, top_k_value in enumerate(d_top_k_array):
                    top_k_str += "top_%d:%.4f " % (idx + 1, top_k_value)
                    
                print("e%d-%d" % (epoch, _epoch), "fold-%d" % cv_k, "auc:%.4f" % m_auc, "mAP:%.4f" % np.mean(ap_list),
                      "top_10:%.4f" % np.mean(d_top_k_array[9]),
                      "top:%.4f" % (test_d_t_list_count / len(test_d_t_list)), want_to_show)
                print(top_k_str)
                epo_file = open(save_path + "/%d" % cv_k + "/%d%d_%d_%.2f_epo.txt" % (d_sim_func, t_sim_func, factor, lambda_c), 'a+')
                
                epo_file.write(
                    "e{ep}-{_e} {hr} {au} {ap} {t} {ks}\n".format(ep=epoch, _e=_epoch, hr=np.mean(d_top_k_array[9]),
                                                                  au=m_auc,
                                                                  ap=np.mean(ap_list),
                                                                  t=(test_d_t_list_count / len(test_d_t_list)),
                                                                  ks= top_k_str))
                epo_file.close()

                if np.mean(ap_list) > max_cor:
                    max_cor = np.mean(ap_list)
                    saver.save(sess, save_path + "/%d" % cv_k + "/sess_saver")
                    np.save(save_path + "/%d" % cv_k + "/Y.npy", _y_p)
            print("epoch: {e_t}, cost: {avg_cost}".format(e_t=epoch, avg_cost=np.sum(epoch_cost) / n))
