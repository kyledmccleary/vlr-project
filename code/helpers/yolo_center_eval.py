from ultralytics import YOLO
import os
import random

def calculate_center(coords, height, width):
    x,y,w,h = coords
    cx = x + w / 2
    cy = y + h / 2
    return cx / width, cy / height

baseline_weight = "runs/detect/53S8/weights/best.pt"
new_weight = "runs/detect/53S7/weights/best.pt"

baseline = YOLO(baseline_weight)
new = YOLO(new_weight)

def eval(label_vals, image):
    values = []

    for label_val in label_vals:
        cx, cy = label_val[1:3]
        values.append((cx, cy))

    base_preds = baseline(image, verbose=False)
    height, width =  base_preds[0].orig_shape
    base_pred_values = []

    for boxes in base_preds[0].boxes.xywh:
        cx,cy = calculate_center(boxes, height, width)
        base_pred_values.append((cx, cy))

    new_preds = new(image, verbose=False)
    new_pred_values = []

    for boxes in new_preds[0].boxes.xywh:
        cx,cy = calculate_center(boxes, height, width)
        new_pred_values.append((cx, cy))

    def mse(target_center, gt_center):
        return (target_center[0] - gt_center[0])**2 + (target_center[1] - gt_center[1])**2

    base_mse_list = []
    new_mse_list = []

    pred_len_base = min(len(base_pred_values), len(values))
    pred_len_new = min(len(new_pred_values), len(values))

    if pred_len_new == 0 or pred_len_base == 0:
        return -1, -1

    for i in range(pred_len_base):
        base_mse_list.append(mse(base_pred_values[i], values[i]))

    for i in range(pred_len_new):
        new_mse_list.append(mse(new_pred_values[i], values[i]))

    avg_base_mse = sum(base_mse_list) / len(base_mse_list)
    avg_new_mse = sum(new_mse_list) / len(new_mse_list)

    return avg_base_mse, avg_new_mse

root = "datasets/53S/val/"
files = os.listdir("datasets/53S/val/images/")
random.shuffle(files)

base_mse_vals = []
new_mse_vals = []

for file in files:
    label = root + "labels/" + file.split(".")[0] + ".txt"
    image = root + "images/" + file

    with open(label) as f:
        label_vals = f.readlines()
        label_vals = [list(map(float, vals.strip().split())) for vals in label_vals if vals != "\n"]
    
    if label_vals != []:
        base_mse, new_mse = eval(label_vals, image)
        if base_mse != -1 and new_mse != -1:
            base_mse_vals.append(base_mse)
            new_mse_vals.append(new_mse)
            print("Processing", file)
            print(f"Base MSE: {base_mse}, New MSE: {new_mse}")

print(f"Average Base MSE: {sum(base_mse_vals) / len(base_mse_vals)}")
print(f"Average New MSE: {sum(new_mse_vals) / len(new_mse_vals)}")