from yolo_custom_trainer import CustomYOLO
from ultralytics import YOLO
import torch

# Set default tensor type
torch.set_default_tensor_type('torch.FloatTensor')

# Now you can use set_printoptions
torch.set_printoptions(precision=10, sci_mode=False)

def train_custom():
    print("Training YOLO with Custom Loss\n")
    args = dict(
        model = 'yolov8m.pt',
        task = 'detect',
        data='53S.yaml',
        degrees=180,
        scale=0.1,
        fliplr=0.0,
        imgsz=1312,
        epochs=20,
        batch=4,
        mosaic = 0,
        perspective = 0,
        resume=False,
        hsv_h=0.055,
        name='53S')

    model = CustomYOLO(args)
    results = model.train()
    
    return results

def train_base():
    model = YOLO('yolov8m.pt', task='detect')
    results = model.train(
        data='53S.yaml',
        degrees=180,
        scale=0.1,
        fliplr=0.0,
        imgsz=1312,
        epochs=20,#100,
        batch=4,
        mosaic = 0.0,
        perspective = 0,
        resume=False,
        hsv_h=0.055,
        name='53S')
    return results

if __name__ == '__main__':
    # model = get_model()
    # results = train_custom()
    results = train_base()
    print(results)


