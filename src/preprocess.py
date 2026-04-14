import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_dir = os.path.join(BASE_DIR, "data", "chest_xray", "train")
val_dir   = os.path.join(BASE_DIR, "data", "chest_xray", "val")
test_dir  = os.path.join(BASE_DIR, "data", "chest_xray", "test")

IMG_SIZE = 224
BATCH_SIZE = 32

def load_data():
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    return train_data, val_data, test_data

def normalize_data(train, val, test):
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train = train.map(lambda x, y: (normalization_layer(x), y))
    val   = val.map(lambda x, y: (normalization_layer(x), y))
    test  = test.map(lambda x, y: (normalization_layer(x), y))

    return train, val, test
