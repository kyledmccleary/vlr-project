from ultralytics import YOLO

regions = ['33T', '52S', '53S', '54S', '54T']

def train_model(region):
    model = YOLO('yolov8s.pt', task='detect')
    model.to('cuda')
    results = model.train(
        data=region + '.yaml',
        degrees=180,
        scale=0.1,
        fliplr=0.0,
        imgsz=1312,
        epochs=100,
        batch=4,
        mosaic = 0.0,
        perspective = 0,
        resume=False,
        hsv_h=0.005,
        name=region + '_s')
    return results

def train_models(regions):
    for region in regions:
        train_model(region)
    return 

def main():
    train_models(regions)


if __name__ == '__main__':
    main()
