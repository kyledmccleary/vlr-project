from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model',required=True)
parser.add_argument('-s','--size',default=512,type=int)
parser.add_argument('-f','--format', default='onnx',type=str)
args = parser.parse_args()


model = YOLO(args.model)
model.export(format=args.format,imgsz=args.size, int8=True, device='cpu')
