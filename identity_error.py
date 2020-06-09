import pickle

BASE_DIR = 'path to ids'
ids = sorted(os.listdir(BASE_DIR))
ids = [os.path.join(BASE_DIR, idx) for idx in ids]


def get_cosine_similarity(avg_desc, synth_desc):
    # sum_1_32 [1 - csim(R(T_k(I_i_j)), r_k)]

    # synth_desc : 32 x 1024
    # avg_desc : 1 x 1024
    dot_prod = np.dot(synth_desc, avg_desc.T) # 32 x 1
    
    avg_desc_norm = np.linalg.norm(avg_desc) # 1
    synth_desc_norm = np.linalg.norm(synth_desc, axis=-1) # 32 x 1
    norm_prod = avg_desc_norm * synth_desc_norm # 32 x 1

    cos_sim = 1 - dot_prod / norm_prod # 32 x 1

    return np.sum(cos_sim)


def sum_over_i(ids, avg_desc_idx, avg_desc):
    # sum_1_30, i neq k

    sum_1_30 = 0.0
    synth_desc_ids = list(set(ids) - set(avg_desc_idx))
    for idx in synth_desc_ids:
        with open(idx, 'rb') as handle:
            synth_desc = pickle.load(handle)

        # avg_desc: 1 x 1024
        # synth_desc: 32 x 1024

        sum_1_30 += get_cosine_similarity(avg_desc, synth_desc)

    return sum_1_30


def sum_over_k(ids):
    # sum_1_30, k

    sum_1_30 = 0.0
    for idx in ids:
        with open(idx, 'rb') as handle:
            avg_desc = pickle.load(handle)

        # avg_desc: 1 x 1024

        sum_1_30 += sum_over_i(ids, idx, avg_desc)

    return sum_1_30


def get_identity_error(ids):

    i_t = sum_over_k(ids)
    i_t = i_t / (30 * 29 * 32)

    return i_t

i_e = get_identity_error(ids)
print('Identity error:{}'.format(i_e))

