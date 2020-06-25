# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
import pickle
from models.resnet import *
import torch
import numpy as np
import time
from config.config import Config
from torch.nn import DataParallel
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, 0)

    # resizing because ArcNet requires images of 128 x 128
    image = cv2.resize(image, (128, 128))
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_features(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc

def voxceleb2_test(model, img_paths, batch_size):
    features, cnt = get_features(model, img_paths, batch_size=batch_size)
    # print(features.shape)
    return features


if __name__ == '__main__':

    opt = Config()
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
        # model = resnet18()
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model = DataParallel(model)
    # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path))
    model.to(torch.device("cuda"))
    print('Loaded model checkpoint')
    model.eval()

    '''Directory:
    real images: ~/modidatasets/VoxCeleb2/preprocessed_data/test/ids/identity'
    synthesized images: 
        ~/synthesized_images/
            <id_1>/
                <id_1_2>/
                    img_n
                <id_1_3>/
                    img_n
            <id_2>/
                <id_2_1>/
                    img_n
                <id_2_3>/
                    img_n
    descriptors_files:
        ~/descriptors_files/
            <id_1>/
                id_1.pkl
                id_1_2.pkl
                id_1_3.pkl
            <id_2>/
                id_2.pkl
                id_2_1.pkl
                id_2_3.pkl


    Read from original images
    Read from synthesized directory
    Save to pickle folder
    '''
    DEST_DIR = '/home/ubuntu/descriptors'
    REAL_DIR = '/home/ubuntu/modidatasets/VoxCeleb2/preprocessed_data/test/ids/identity'

    ids = os.listdir(REAL_DIR)

    for idx in ids:
        idx_path = os.path.join(REAL_DIR, idx)
        codes = os.listdir(idx_path)
        for code in codes:
            code_path = os.path.join(idx_path, code)
            mp4s = os.listdir(code_path)
            for mp4 in mp4s:
                mp4_path = os.path.join(code_path, mp4)
                frames = sorted(os.listdir(mp4_path))

                frames = [os.path.join(mp4_path, frame) for frame in frames]
                frames = [frame for frame in frames if 'random_frame' in frame]

                # there should be at most 32 frames
                features = voxceleb2_test(model, frames, 32) # 32 x 1024
                features = np.mean(features, axis=0)
                save_dir = os.path.join(DEST_DIR, '{}'.format(idx))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                with open(os.path.join(save_dir, '{}.pkl'.format(idx)), 'wb') as handle:
                    pickle.dump(features, handle)

    print('Averaged decriptors extracted')

    SYNTH_DIR = '/home/ubuntu/synthesized_images' 
    ids = os.listdir(SYNTH_DIR)

    for idx in ids:
        idx_path = os.path.join(SYNTH_DIR, idx)
        sub_ids = os.listdir(idx_path)
        sub_ids = [sub_idx for sub_idx in sub_ids if not sub_idx == idx]

        for sub_idx in sub_ids:
            sub_idx_path = os.path.join(idx_path, sub_idx)
            frames = sorted(os.listdir(sub_idx_path))
            frames = [os.path.join(sub_idx_path, frame) for frame in frames]

            features = voxceleb2_test(model, frames, len(frames))
            
            save_dir = os.path.join(DEST_DIR, '{}'.format(idx))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with open(os.path.join(save_dir, '{}.pkl'.format(sub_idx)), 'wb') as handle:
                pickle.dump(features, handle)

    print('Descriptors computed for synthesized images')    

