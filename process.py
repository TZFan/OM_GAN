import pandas as pd
import os

class_names = ["fear", "angry", "happy", "sad", "neutral"]
base_path = os.getcwd()
class_to_label = {class_name: i for i, class_name in enumerate(class_names)}

for class_name in class_names:
    images = []
    labels = []

    for folder in ['train', 'test']:
        folder_path = os.path.join(base_path, folder)
        class_file = os.path.join(folder_path, class_name)

        for filename in os.listdir(class_file):
            if filename.endswith(".jpg"):
                image_path = os.path.join(class_file, filename)
                images.append(image_path)
                labels.append(class_to_label[class_name])

    df = pd.DataFrame(data={"image": images, "label": labels})
    df.to_csv(f"{class_name}.csv", sep=",", index=False)