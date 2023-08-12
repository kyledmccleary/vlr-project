import classify
import opts
import os
import locate

IM_PATH = '../data2'

MODEL_PATH = opts.MODEL_PATH
model = classify.loadModel(MODEL_PATH)
model.eval()

test_files = os.listdir(IM_PATH)

lon_diffs = []
lat_diffs = []
ccoeffs = []
preds = []

for file in test_files:
    filepath = os.path.join(IM_PATH,file)
    pred = classify.makePrediction(filepath, model)
    if pred != 'elsweyr':
        avgdiff_lons, avgdiff_lats, ccoeff = locate.locateInTile(filepath, pred)
        
    else:
        avgdiff_lons, avgdiff_lats, ccoeff = None,None,None
    preds.append(pred)
    lon_diffs.append(avgdiff_lons)
    lat_diffs.append(avgdiff_lats)
    ccoeffs.append(ccoeff)
    