import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

# === Завантаження моделі ===
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    num_faces=10,
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

def is_eye_open(landmarks, eye_idxs, threshold=0.03):
    upper = landmarks[eye_idxs[0]]
    lower = landmarks[eye_idxs[1]]
    return abs(upper.y - lower.y) > threshold

def detect_faces_and_eyes(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = face_landmarker.detect(mp_image)

    face_data = []
    for idx, face_landmarks in enumerate(results.face_landmarks):
        left_eye_open = is_eye_open(face_landmarks, [159, 145])
        right_eye_open = is_eye_open(face_landmarks, [386, 374])
        face_data.append({
            "face_index": idx,
            "left_eye_open": left_eye_open,
            "right_eye_open": right_eye_open,
            "landmarks": face_landmarks
        })

    return face_data

def create_eye_mask(image, landmarks, eye_idxs):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = image.shape[:2]
    points = []
    for idx in eye_idxs:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
    points = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, points, 255)
    return mask


# === Вхідне зображення ===
img_path = "/Users/lilianamirchuk/Desktop/Mirchuk-task4/data/inputs/Image_3.png"
img = cv2.imread(img_path)
results = detect_faces_and_eyes(img)

print(f"Знайдено облич: {len(results)}")
for i, face in enumerate(results):
    print(f"Обличчя {i}: Left eye open: {face['left_eye_open']}, Right eye open: {face['right_eye_open']}")

# === Створення маски та збереження зображень ===
for face in results:
    if not (face["left_eye_open"] and face["right_eye_open"]):
        landmarks = face["landmarks"]
        LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144, 145]
        RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380, 374]

        eye_mask_left = create_eye_mask(img, landmarks, LEFT_EYE_IDX)
        eye_mask_right = create_eye_mask(img, landmarks, RIGHT_EYE_IDX)
        full_mask = cv2.bitwise_or(eye_mask_left, eye_mask_right)


        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        init_image = Image.fromarray(img_rgb).convert("RGB")
        mask_image = Image.fromarray(full_mask).convert("RGB")

        img = Image.open("init_image.png")
        img.thumbnail((512, 512))
        img.save("init_image_small.png")
        mask_image.save("mask_image.png")
        print("Зображення та маска збережені: init_image.png, mask_image.png")
        break 
