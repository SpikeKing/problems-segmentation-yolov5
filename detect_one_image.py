#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 28.12.20
"""
import copy
import os
import itertools
import cv2
import onnxruntime
import numpy as np
import torch

from myutils.cv_utils import *
from myutils.project_utils import *

from models.experimental import attempt_load
from root_dir import DATA_DIR
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


class ImgDetector(object):
    """
    图像检测
    """
    def __init__(self):
        # self.weights = "mydata/models/best-20201224.pt"
        # self.weights = "mydata/models/best-20210118.pt"
        self.weights = "mydata/models/best-m-20210121.pt"

        self.img_size = 640
        self.conf_thres = 0.15
        self.iou_thres = 0.45

        self.device = select_device()  # 自动选择环境
        self.is_half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model, self.img_size = self.load_model()  # 加载模型

        self.onnx_path = os.path.join(DATA_DIR, 'models', 'best-m-20210121.onnx')
        self.onnx_sess = self.load_onnx_model()

    def load_model(self):
        """
        加载模型
        """
        # Load model
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        if self.is_half:
            model.half()  # to FP16
        img_size = check_img_size(self.img_size, s=model.stride.max())  # check img_size

        # 设置Img Half
        img_tmp = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = model(img_tmp.half() if self.is_half else img_tmp) if self.device.type != 'cpu' else None  # run once

        return model, img_size

    def preprocess_data(self, img_bgr):
        """
        图像预处理
        """
        # Padded resize
        img_bgr = letterbox(img_bgr, new_shape=self.img_size, auto=False, scaleFill=True)[0]

        # Convert
        img_rgb = img_bgr[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_rgb = np.ascontiguousarray(img_rgb)

        # 正则化
        img = torch.from_numpy(img_rgb).to(self.device)
        img = img.half() if self.is_half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def detect_problems(self, img_bgr):
        """
        检测逻辑
        """
        img_ori = copy.copy(img_bgr)

        img = self.preprocess_data(img_bgr)  # 预处理数据

        pred = self.model(img, augment=False)[0]  # 预测图像

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)  # NMS后处理

        box_list = []  # 最终输出
        for i, det in enumerate(pred):  # detections per image
            # 回复图像尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_ori.shape).round()
            det = det.tolist()

            for *xyxy, conf, cls in reversed(det):  # 绘制图像
                xyxy = [int(i) for i in xyxy]
                conf = round(conf, 4)
                cls = int(cls)
                # output_list.append([xyxy, conf, cls])
                box_list.append(xyxy)

        return box_list

    def load_onnx_model(self):
        """
        加载onnx模型
        """
        onnx_sess = onnxruntime.InferenceSession(self.onnx_path)
        onnx_sess.get_modelmeta()

        inputs_meta = onnx_sess.get_inputs()
        for in_meta in inputs_meta:
            print('[Info] in_meta name: {}'.format(in_meta.name))

        outputs_meta = onnx_sess.get_outputs()
        for out_meta in outputs_meta:
            print('[Info] out_meta name: {}'.format(out_meta.name))

        return onnx_sess

    def detect_problems_by_onnx(self, img_bgr):
        """
        预测题目
        """
        img_ori = copy.copy(img_bgr)

        img = self.preprocess_data(img_bgr)  # 预处理数据

        img_np = img.numpy()

        pred_onnx = self.onnx_sess.run(["output"], {"images": img_np})[0]
        pred_onnx = torch.from_numpy(pred_onnx).to(self.device)
        pred = non_max_suppression(pred_onnx, self.conf_thres, self.iou_thres)  # NMS后处理

        box_list = []  # 最终输出
        for i, det in enumerate(pred):  # detections per image
            # 回复图像尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_ori.shape).round()
            det = det.tolist()

            for *xyxy, conf, cls in reversed(det):  # 绘制图像
                xyxy = [int(i) for i in xyxy]
                conf = round(conf, 4)
                cls = int(cls)
                # output_list.append([xyxy, conf, cls])
                box_list.append(xyxy)

        return box_list


def filer_boxes_by_size(boxes, r_thr=0.5):
    """
    根据是否重叠过滤包含在内部的框
    """
    if not boxes:
        return boxes

    size_list = []
    idx_list = []
    for idx, box in enumerate(boxes):
        size_list.append(get_box_size(box))
        idx_list.append(idx)

    size_list, sorted_idxes, sorted_boxes = \
        sort_three_list(size_list, idx_list, boxes, reverse=True)

    n_box = len(sorted_boxes)  # box的数量
    flag_list = [True] * n_box

    for i in range(n_box):
        if not flag_list[i]:
            continue
        x_boxes = [sorted_boxes[i]]
        for j in range(i+1, n_box):
            box1 = sorted_boxes[i]
            box2 = sorted_boxes[j]
            r_iou = min_iou(box1, box2)
            if r_iou > r_thr:
                flag_list[j] = False
                x_boxes.append(box2)
        print('[Info] i: {}, x_boxes: {}'.format(i, x_boxes))
        sorted_boxes[i] = merge_boxes(x_boxes)

    new_boxes = []
    for i in range(n_box):
        if flag_list[i]:
            new_boxes.append(sorted_boxes[i])

    return new_boxes


def draw_problems_boxes(img_bgr, box_list):
    box_list = filer_boxes_by_size(box_list)
    sorted_boxes, sorted_idxes, num_row = sorted_boxes_by_col(box_list)
    new_boxes = []
    for col_boxes in sorted_boxes:
        new_boxes += col_boxes
    img_out = draw_box_list(img_bgr, new_boxes, is_text=False)
    return img_out


def normal_vs_onnx():
    ido = ImgDetector()

    img_bgr = cv2.imread(os.path.join(DATA_DIR, 'test', 'test1.jpg'))

    onnx_box_list = ido.detect_problems_by_onnx(img_bgr)

    img_onnx = draw_problems_boxes(img_bgr, onnx_box_list)
    onnx_path = os.path.join(DATA_DIR, 'test', "test1.onnx.jpg")
    cv2.imwrite(onnx_path, img_onnx)
    print('[Info] onnx输出路径: {}'.format(onnx_path))

    box_list = ido.detect_problems(img_bgr)
    img_out = draw_problems_boxes(img_bgr, box_list)
    out_path = os.path.join(DATA_DIR, 'test', "test1.out.jpg")
    cv2.imwrite(out_path, img_out)
    print('[Info] out输出路径: {}'.format(out_path))


def process():
    file_path = os.path.join(DATA_DIR, 'page_dataset_raw',
                             'sanghu.zj_question_cut_sampled_kousuanpigai_url_3w_0104')

    out_dir = os.path.join(DATA_DIR, 'zhengyepai-old')
    mkdir_if_not_exist(out_dir)

    data_lines = read_file(file_path)
    data_lines = data_lines[:201]
    print('[Info] 样本数: {}'.format(len(data_lines)))
    ido = ImgDetector()

    for idx, data_line in enumerate(data_lines):
        img_name = data_line.split('/')[-1]
        url = data_line
        is_ok, img_bgr = download_url_img(url)

        box_list = ido.detect_problems(img_bgr)

        img_out = draw_problems_boxes(img_bgr, box_list)
        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, img_out)
        print('[Info] {} 处理完成: {}'.format(idx, url))


def main():
    normal_vs_onnx()


if __name__ == '__main__':
    main()
