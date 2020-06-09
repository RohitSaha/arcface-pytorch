import numpy as np
import pickle

def l2_distance(features_1, features_2):
    # features_1: 68 x 2
    # features_2: 68 x 2

    diff = features_1 - features_2
    diff = diff ** 2 # 68 x 2
    diff = np.reduce_sum(diff, axis=-1) # 68 x 1
    diff = diff ** 0.5
    diff = np.mean(diff) # [1]

    return diff


def sum_over_j(real_landmark_paths, synthesized_landmark_paths):
    # sum j_33_64
    sum_j = 0.0

    # Frame id should be the same in both cases.
    # Equivalent frames will be compared.
    # There should be at most 32 frames.
    real_landmark_paths = sorted(real_landmark_paths)
    synthesized_landmark_paths = sorted(synthesized_landmark_paths)

    for real_path, synth_path in zip(real_landmark_paths, synthesized_landmark_paths):
        with open(real_path, 'rb') as handle:
            features_1 = pickle.load(handle)
        with open(synth_path, 'rb') as handle:
            features_2 = pickle.load(handle)

        sum_j += l2_distance(features_1, features_2)

    return sum_j


def sum_over_k(REAL_DIR, FAKE_DIR):
    # gather real facial and fake facial pickle files
    # corresponding to each ID
    sum_k = 0.0

    ids = os.listdir(REAL_DIR) # same ids should be in FAKE_DIR

    # gather paths to pickle files of ids
    for idx in ids:
        # list pickle files
        real_landmark_paths = sorted(os.listdir(REAL_DIR, idx)) 
        real_landmark_paths = [
            os.path.join(REAL_DIR, idx, path)
            for path in real_landmark_paths]
        fake_landmark_paths = sorted(os.listdir(FAKE_DIR, idx))
        fake_landmark_paths = [
            os.path.join(FAKE_DIR, idx, path)
            for path in fake_landmark_paths]

        # make sure the indexes in both paths correspond to the same
        # frame numbers in proper order

        sum_k += sum_over_j(real_landmark_paths, fake_landmark_paths)

    return sum_k


def get_pose_rec_error():
    REAL_DIR = 'path to directory that has real images' 
    FAKE_DIR = 'path to directory that has fake images'

    pose_error = sum_over_k(REAL_DIR, FAKE_DIR)

    return pose_error / (30. * 32)


pose_error = get_pose_rec_error()
print('Pose reconstruction error:{}'.format(pose_error))



