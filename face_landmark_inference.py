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


ROOT_DIR = 'path to synthesized images'
path_to_images = []
