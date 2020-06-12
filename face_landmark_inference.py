import os
import face_alignment
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
                id_1.pkl # driver - frame 32 -> 64
                id_1_1.pkl # synthesized using driver - frame 32 -> 64
            <id_2>/
                id_2.pkl
                id_2_2.pkl

    Read from original images
    Read from synthesized directory
    Save to pickle folder
'''
LANDMARK_DIR = '/home/ubuntu/landmark_files'
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
            frames = sorted(os.listdir(mp4_path))
            frames = [os.path.join(mp4_path, frame) for frame in frames]
            frames = [frame for frame in frames if 'random_frame' in frame]

            for frame in frames:
                img = cv2.imread(frame)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                try:
                    preds, boxes = fa.get_landmarks(img)
                except:
                    continue

                preds = preds[0] # 68 x 2
                save_dir = os.path.join(LANDMARK_DIR, '{}'.format(idx))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                with open(os.path.join(save_dir, '{}.pkl'.format(idx)), 'wb') as handle:
                    pickle.dump(preds, handle)        

print('Landmark for real images computed')

SYNTH_DIR = '/home/ubuntu/synthesized_images_pose'
ids = os.listdir(SYNTH_DIR)

for idx in ids:
    idx_path = os.path.join(SYNTH_DIR, idx)
    sub_ids = os.listdir(idx_path)

    for sub_idx in sub_ids:
        sub_idx_path = os.path.join(idx_path, sub_idx)
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
            save_dir = os.path.join(LANDMARK_DIR, '{}.pkl'.format(idx))
            if not os.path.join(save_dir):
                os.makedirs(save_dir)

            with open(os.path.join(save_dir, '{}.pkl'.format(sub_idx)), 'wb') as handle:
                pickle.dump(preds, handle)


print('Landmark for synthesized images computed')

