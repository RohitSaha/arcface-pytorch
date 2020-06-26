import os
import numpy as np
import face_alignment
import pickle
import cv2

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D,
    face_detector="sfd",
    flip_input=False,
    device="cpu"
)

def get_features(path_to_images):

    for image in path_to_images:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            preds, boxes = fa.get_landmarks(img)
        except:
            continue

        preds = preds[0] # 68 x 2
        with open(SAVE_DIR + 'path', 'wb') as handle:
            pickle.dump(preds, handle)        


'''Directory:
    real image: ~/modidatasets/VoxCeleb2/preprocessed_data/test/ids/pose'
    synthesized images:
        ~/synthesized_images_pose/
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
    landmark files:
        ~/landmark_files/
            <id_1>/
                <driver>/
                    id_1_frame-number.pkl # driver - frame 32 -> 64
                    id_1-frame-number.pkl
                <driven>/
                    id_1_1_frame-number.pkl # synthesized using driver - frame 32 -> 64
                    id_1_1_frame-number.pkl
            <id_2>/
                <driver>/
                    id_2_frame-number.pkl
                <driven>/
                    id_2_2_frame-number.pkl

    Read from original images
    Read from synthesized directory
    Save to pickle folder
'''
LANDMARK_DIR = '/home/ubuntu/landmark'
REAL_DIR = '/home/ubuntu/modidatasets/VoxCeleb2/preprocessed_data/test/ids/pose'

ids = os.listdir(REAL_DIR)

for idx in ids:
    idx_path = os.path.join(REAL_DIR, idx)
    codes = os.listdir(idx_path)
    for code in codes:
        code_path = os.path.join(idx_path, code)
        mp4s = os.listdir(code_path)
        for mp4 in mp4s:
            mp4_path = os.path.join(code_path, mp4)
            images = sorted(os.listdir(mp4_path))
            images = [os.path.join(mp4_path, image) for image in images]
            frames = [image for image in images if 'random_frame' in image]
            masks = [image for image in images if 'random_mask' in image]
            
            for frame in frames:
                frame_number = frame.split('/')[-1].split('_')[0]
                mask_image = [
                    mask
                    for mask in masks
                    if frame_number in mask][0]

                img = cv2.imread(frame)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 128))
                mask = cv2.imread(mask_image)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                mask = cv2.resize(mask, (128, 128))
                mask = np.mean(mask, axis=-1, keepdims=True)
                mask = np.where(mask > 0.001, 1., 0.)
                img = img * mask
                try:
                    preds, boxes = fa.get_landmarks(img)
                except:
                    continue

                preds = preds[0] # 68 x 2
                save_dir = os.path.join(LANDMARK_DIR, '{}'.format(idx), 'driver')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                frame_number = frame.split('/')[-1].split('_')[0]
                with open(os.path.join(save_dir, '{}_{}.pkl'.format(idx, frame_number)), 'wb') as handle:
                    pickle.dump(preds, handle)        

print('Landmark for real images computed')

SYNTH_DIR = '/home/ubuntu/synthesized_images'
ids = os.listdir(SYNTH_DIR)

for idx in ids:
    idx_path = os.path.join(SYNTH_DIR, idx)
    sub_idx_path = os.path.join(idx_path, idx)

    frames = sorted(os.listdir(sub_idx_path))
    frames = [os.path.join(sub_idx_path, frame) for frame in frames]
    frames = [frame for frame in frames if 'random_frame' in frame]

    for frame in frames:
        img = cv2.imread(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            preds, boxes = fa.get_landmarks(img)
        except:
            continue

        preds = preds[0]
        save_dir = os.path.join(LANDMARK_DIR, '{}'.format(idx), 'driven')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        frame_number = frame.split('/')[-1].split('_')[0]
        with open(os.path.join(save_dir, '{}_{}.pkl'.format(idx, frame_number)), 'wb') as handle:
            pickle.dump(preds, handle)


print('Landmark for synthesized images computed')

