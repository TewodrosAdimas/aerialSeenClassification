import numpy as np
from tensorflow.keras.preprocessing import image


UC_CLASSES = [
    "agricultural",
    "airplane",
    "baseball_diamond",
    "beach",
    "buildings",
    "chaparral",
    "dense_residential",
    "forest",
    "freeway",
    "golf_course",
    "harbor",
    "intersection",
    "medium_residential",
    "mobile_home_park",
    "overpass",
    "parking_lot",
    "river",
    "runway",
    "sparse_residential",
    "storage_tanks",
    "tennis_court"
]


def preprocess_image(img, size=(224,224)):

    img = img.resize(size)

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    return img_array