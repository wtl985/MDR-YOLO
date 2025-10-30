import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO(model=r'/CSF-YOLO/cfg/models/MDR-YOLO/MDR-YOLO.yaml')\
        #.load('/MDR-YOLO/best.pt')  # build from YAML and transfer weights
    model.train(data=r'/CSF-YOLO/cfg/models/MDR-YOLO/MDR-YOLO.yaml/GT data.yaml',
                imgsz=640,
                epochs=300,
                batch=16,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                amp=False
                )
