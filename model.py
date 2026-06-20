from ultralytics import YOLO

def train_yolo_model():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        mosaic=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=20.0,
        translate=0.05,
        scale=0.3,

        hsv_h = 0.01,
        hsv_s = 0.3,
        hsv_v = 0.4,

        perspective = 0.001,

        batch=8,
        cache=False
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