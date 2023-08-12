from ultralytics import YOLO

model = YOLO('yolov8s.pt', task='detect')
model.to('cuda')

if __name__ == '__main__':
    results = model.train(
        data='yolov8n_cc5.yaml',
        degrees=180,
        scale=0.1,
        fliplr=0.0,
        imgsz=1312,
        epochs=100,
        batch=4,
        mosaic = 0,
        perspective = 0,
        resume=False,
        # hsv_h=0.055,
        name='33S_s')
