from ultralytics import YOLO

model = YOLO('yolov8n.pt', task='detect')
model.to('cuda')

if __name__ == '__main__':
    results = model.train(
        data='yolov8n_cc5.yaml',
        degrees=180,
        scale=0.5,
        fliplr=0.0,
        mosaic=0.0,
        perspective=0.0001,
        imgsz=1024,
        epochs=300,
        batch=16,
        name='yolov8n_rgb23c')
