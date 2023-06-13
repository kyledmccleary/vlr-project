from ultralytics import YOLO

model = YOLO('best.pt')

if __name__ == '__main__':
    results = model.val(data='yolov8n_cc5.yaml')
    # results = model.train(
    #     data='yolov8n_it_hand_val.yaml',
    #     degrees = 180,
    #     scale = 0.1,
    #     fliplr = 0.0,
    #     mosaic = 0.0,
    #     perspective = 0.0001,
    #     imgsz=1024,
    #     epochs = 100,
    #     batch = 32,
    #     name = 'yolov8n_it_grid')