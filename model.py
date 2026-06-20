from ultralytics import YOLO

def train_yolo_model():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,

        # Generate 2 augmented outputs per training example
        augment=True,

        # Disable augmentations you don't want
        mosaic=0.0,
        fliplr=0.0,
        flipud=0.0,
        translate=0.0,
        scale=0.0,
        perspective=0.0,

        # Rotation
        degrees=20.0,   # random rotation between -20° and +20°

        # Color adjustments
        hsv_s=0.05,     # saturation ±5%
        hsv_v=0.10,     # brightness/value ±10%

        # Exposure approximation
        # YOLO doesn't expose separate "exposure";
        # hsv_v is the closest equivalent
        hsv_h=0.0,

        batch=8,
        cache=False,
    )

    # print(type(results))
    # print("\n\n\n\n\n")
    # print(dir(results))
    # print("\n\n\n\n\n")
    # print(results)
    # print("\n\n\n\n\n")

    """
    from ultralytics import YOLO

    model = YOLO("runs/detect/train/weights/last.pt")
    model.train(resume=True)
    """

def main():
    train_yolo_model()

if __name__ == "__main__":
    main()