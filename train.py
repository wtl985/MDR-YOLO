# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == "__main__":
    # 使用自己的YOLOv8.yamy文件搭建模型并加载预训练权重训练模型
    model = YOLO(r"D:\PyCharm\yolo11\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml") \
        #.load(r'D:\PyCharm\yolo11\ultralytics-main\yolo11n.pt')  # build from YAML and transfer weights
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml')   #  原始yolo11模型
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-MSCAN.yaml')  #  添加MSCAN的模型
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-smallhead.yaml')  # 添加小目标检测头模型
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-Down_wt.yaml')  #引入Haar小波下采样Down_wt卷积
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-RCS.yaml')  #引入通道混洗的重参数化卷积RCS
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-MSCAN+smallhead.yaml')#  添加MSCAN+小目标检测头的模型
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-MSCAN+Down.yaml')#  添加MSCAN+Haar小波下采样Down_wt卷积
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-MSCAN+RCS.yaml')  # 添加MSCAN+RCS
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-Down_wt+RCS.yaml')  # 添加Haar小波下采样Down_wt+RCS
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-MSCAN+Down+RCS.yaml')  # 添加MSCAN+小波下采样Down_wt+RCS
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\ultralytics\cfg\models\v9\yolov9c.yaml')   #  原始yolo11模型
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-SAFM.yaml') #添加空间自适应特征调制网络SAFMN 增强neck部分上采样分辨率
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-WTCon.yaml') #引入小波卷积WTConv    (一定要改head)
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-HRAMI.yaml') #引入分层互补注意力混合器HRAMi
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-WTCon+SAFM.yaml')
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-WTCon+HRAMI.yaml')
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-HRAMI+SAFM.yaml')
    #model = YOLO(model=r'D:\PyCharm\yolo11\ultralytics-main\yolo11-WTCon+HRAMI+SAFM.yaml')  # 引入WTCon+HRAMI+SAFM   (一定要改head)
    model.train(data=r'D:\PyCharm\yolo11\ultralytics-main\jiazawu.yaml',
                imgsz=640,
                epochs=2,
                batch=8,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train2',
                name='exp',
                single_cls=False,
                cache=False,
                #amp=False       #(引入Haar小波下采样Down_wt卷积时，要把混合精度关了 amp=False）
                )
