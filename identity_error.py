import pickle
import numpy as np


def sum_over_j(avg_desc, synth_desc):
    # sum_1_32 [1 - csim(R(T_k(I_i_j)), r_k)]

    # synth_desc : 32 x 1024
    # avg_desc : 1 x 1024
    dot_prod = np.dot(synth_desc, avg_desc.T) # 32 x 1
    
    avg_desc_norm = np.linalg.norm(avg_desc) # 1
    synth_desc_norm = np.linalg.norm(synth_desc, axis=-1) # 32 x 1
    norm_prod = avg_desc_norm * synth_desc_norm # 32 x 1

    cos_sim = 1. - dot_prod / norm_prod # 32 x 1

    sum_cos_sim = np.sum(cos_sim)[0] # 1

    return sum_cos_sim


def sum_over_i(idx, idx_path):
    # idx_path: ~/descriptor_files/idx/
    # idx: original id of the current celebrity

    # sum_1_30, i neq k
    sum_1_29 = 0.0

    avg_desc_path = os.path.join(idx_path, idx + '.pkl')
    with open(avg_desc_path, 'rb') as handle:
        avg_desc = pickle.load(handle)
    
    avg_desc = np.reshape(avg_desc, [1, 1024]) 

    sub_ids = os.listdir(idx_path)
    synth_desc_ids = list(set(sub_ids) - set(idx))

    for sub_idx in synth_desc_ids:
        sub_idx_path = os.path.join(idx_path, sub_idx + '.pkl')
        with open(sub_idx_path, 'rb') as handle:
            synth_desc = pickle.load(handle)

        # avg_desc: 1 x 1024
        # synth_desc: 32 x 1024

        sum_1_29 += sum_over_j(avg_desc, synth_desc)

    return sum_1_29


def sum_over_k(DESC_DIR, ids):
    # ids: [idx_1, idx_2, ..., idx_n]
    # DESC_DIR = /home/ubuntu/descriptor_files

    # sum_1_30, k
    sum_1_30 = 0.0

    for idx in ids:
        idx_path = os.path.join(DESC_DIR, idx)
        sum_1_30 += sum_over_i(idx, idx_path)

    return sum_1_30


def get_identity_error(DESC_DIR, ids):

    i_t = sum_over_k(DESC_DIR, ids)
    i_t = i_t / (30. * 29. * 32.)

    return i_t

i_e = get_identity_error(ids)
print('Identity error:{}'.format(i_e))
DESC_DIR = '/home/ubuntu/descriptors_files'
ids = sorted(os.listdir(DESC_DIR))
identity_error = get_identity_error(DESC_DIR, ids)
print('Identity error:{}'.format(identity_error))


