#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp
#from pycocotools.cocoeval import COCOeval



class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/"
        self.train_ann = "train_annotations.coco.json"
        self.val_ann = "valid_annotations.coco.json"
        self.test_ann = "test_annotations.coco.json"

        self.num_classes = 3

        self.max_epoch = 10
        self.data_num_workers = 4


#TRAIN#
### python tools/train.py -f datasets\wp\yolox_s.py -d 1 -b 8 -o -c datasets\wp\yolox_s.pth ###

#TEST#
### python tools/demo.py image -f datasets\wp\yolox_s.py -c YOLOX_outputs\yolox_s\best_ckpt.pth --path assets/wp3.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu] ###

### python tools/demo.py video -f datasets\wp\yolox_s.py -c YOLOX_outputs\latest_ckpt.pth --path datasets/ninja_17.mp4 --conf 0.5 --nms 0.45 --tsize 640 --save_result --device gpu ###