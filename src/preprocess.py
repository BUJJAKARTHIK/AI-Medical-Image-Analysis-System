import cv2
import numpy as np
import os

IMG_SIZE = 224

def load_data(data_dir):
    data = []
    labels = []

    categories = ["NORMAL", "PNEUMONIA"]

    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)

        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(class_num)
            except:
                pass

    return np.array(data), np.array(labels)