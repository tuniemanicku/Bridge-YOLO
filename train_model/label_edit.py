import os
import random
from pathlib import Path
import numpy as np
from ultralytics import YOLO

def create_learn_and_test_sets(
        img_source_dir: str = "train",
        lbl_source_dir: str = "train",
        img_test_dir: str = "val",
        lbl_test_dir: str = "val", 
        train_ratio: float = 0.8,) -> None:
    
    img_in_dir = Path(img_source_dir)
    img_in_dir.mkdir(parents=True, exist_ok=True)
    img_test = Path(img_test_dir)
    img_test.mkdir(parents=True, exist_ok=True)
    label_in_dir = Path(lbl_source_dir)
    label_in_dir.mkdir(parents=True, exist_ok=True)
    lbl_test = Path(lbl_test_dir)
    lbl_test.mkdir(parents=True, exist_ok=True)

    images = Path(img_source_dir).glob("*")
    images = list(images)
    labels = Path(lbl_source_dir).glob("*")
    labels = list(labels)

    test_labels = random.sample(labels, int(np.ceil(len(labels) * (1 - train_ratio))))
    # print(test_labels)
    test_images = []
    for lbl in test_labels:
        img_name = lbl.name.split("__")[-1]
        test_images.append(img_in_dir / Path(img_name.split(".")[0] + ".jpg"))

        
    # print(test_images)

    for image in test_images:
        new_img_path = img_test / f"{image.name}"
        os.rename(image, new_img_path)
    for label in test_labels:
        new_label = label.name.split("__")[-1]
        new_lbl_path = lbl_test / f"{new_label}"
        os.rename(label, new_lbl_path)

    for image in images:
        if image not in test_images:
            new_img_path = img_in_dir / f"{image.name}"
            os.rename(image, new_img_path)
    for label in labels:
        if label not in test_labels:
            new_label = label.name.split("__")[-1]
            new_lbl_path = label_in_dir / f"{new_label}"
            os.rename(label, new_lbl_path)

def main():
    create_learn_and_test_sets(
        img_source_dir="images/train", 
        lbl_source_dir="labels/train", 
        img_test_dir="images/val", 
        lbl_test_dir="labels/val")

if __name__ == "__main__":
    main()