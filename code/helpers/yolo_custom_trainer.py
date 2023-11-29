from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from yolo_custom_loss import v8DetectionLoss

class CustomModel(DetectionModel):
    def init_criterion(self):
        return v8DetectionLoss(self)
    
class CustomTrainer(DetectionTrainer):
    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        self.loss_names = 'ctr_loss', 'cls_loss', 'dfl_loss'
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = CustomModel(cfg, nc=self.data['nc'], verbose=verbose)
        if weights:
            model.load(weights)
        return model

def CustomYOLO(args):
    trainer = CustomTrainer(overrides=args)
    return trainer