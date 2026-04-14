import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# ===============================
# LOAD TRAINED MODEL
# ===============================
model = load_model("models/medical_model.h5")

IMG_SIZE = 224

# ===============================
# CLASS LABELS
# ===============================
classes = ["Normal", "Pneumonia"]

# ===============================
# PREPROCESS IMAGE
# ===============================
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(" Image not found. Check dataset path!")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ===============================
# PREDICT FUNCTION
# ===============================
def predict(img_path):
    img = preprocess_image(img_path)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    return classes[class_index], confidence

# ===============================
# AUTO SELECT REAL IMAGE (FIXED)
# ===============================
if __name__ == "__main__":

    test_dir = "data/chest_xray/test/NORMAL"

    # get all images
    image_list = os.listdir(test_dir)

    if len(image_list) == 0:
        raise ValueError(" No images found in test folder!")

    # pick first image automatically
    test_image = os.path.join(test_dir, image_list[0])

    print("Using image:", test_image)

    # prediction
    label, conf = predict(test_image)

    # output
    print("\n AI DIAGNOSIS RESULT")
    print("------------------------")
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.2f}%")

    # display image
    img = cv2.imread(test_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(f"{label} ({conf:.2f}%)")
    plt.axis("off")
    plt.show()