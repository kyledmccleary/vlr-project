from ultralytics import YOLO

model = YOLO('yolov8x.pt',task='detect')
model.to('cuda')






if __name__ == '__main__':
    results = model.train(
        data='yolov8x_fl.yaml',
        degrees = 180,
        scale = 0.1,
        fliplr = 0.0,
        mosaic = 0.0,
        perspective = 0.0001,
        imgsz=1024,
        epochs = 300,
        batch = 4,
        name = 'yolov8x_fl')