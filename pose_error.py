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

    return diff[0]


def sum_over_j(driver_path, driven_path):
    # driver_path: full path to driver folder
    # driven_path: full path to driven folder

    # sum j_33_64
    sum_j = 0.0

    driver_pickle_files = sorted(os.listdir(driver_path))
    driven_pickle_files = sorted(os.listdir(driven_path))

    for driver_pi, driven_pi in zip(driver_pickle_files, driven_pickle_files):
        driver_pi_path = os.path.join(driver_path, driver_pi)
        driven_pi_path = os.path.join(driven_path, driven_pi)

        with open(driver_pi_path, 'rb') as handle:
            features_1 = pickle.load(handle)
        with open(driven_pi_path, 'rb') as handle:
            features_2 = pickle.load(handle)

        sum_j += l2_distance(features_1, features_2)


    return sum_j


def sum_over_k(LANDMARK_DIR, ids):
    # LANDMARK_DIR = /home/ubuntu/landmark_files
    # ids = [idx_1, idx_2, ..., idx_n]

    sum_k = 0.0

    for idx in ids:
        idx_path = os.path.join(LANDMARK_DIR, idx)
        driver_path = os.path.join(idx_path, 'driver')
        driven_path = os.path.join(idx_path, 'driven')

        sum_k += sum_over_j(driver_path, driven_path)

    return sum_k


def get_pose_rec_error(LANDMARK_DIR, ids):

    pose_error = sum_over_k(LANDMARK_DIR, ids)

    return pose_error / (30. * 32.)


LANDMARK_DIR = '/home/ubuntu/landmark_files'
ids = sorted(os.listdir(LANDMARK_DIR))
pose_error = get_pose_rec_error(LANDMARK_DIR, ids)
print('Pose reconstruction error:{}'.format(pose_error))



