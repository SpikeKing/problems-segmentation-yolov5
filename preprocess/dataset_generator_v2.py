#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 6.1.21
"""

import os
import sys
import json
import cv2
import collections

from myutils.cv_utils import draw_box
from myutils.project_utils import *
from root_dir import DATA_DIR


class DatasetGeneratorV2(object):
    """
    数据集生成
    """
    def __init__(self):
        self.url_format = "http://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/problems_segmentation/" \
                          "datasets/prelabeled-20201224/{}.jpg"
        self.out_dataset_dir = os.path.join(DATA_DIR, 'ps_datasets_v2')

        self.out_images_dir = os.path.join(self.out_dataset_dir, 'images')
        self.out_images_train_dir = os.path.join(self.out_images_dir, 'train')
        self.out_images_val_dir = os.path.join(self.out_images_dir, 'val')

        self.out_labels_dir = os.path.join(self.out_dataset_dir, 'labels')
        self.out_labels_train_dir = os.path.join(self.out_labels_dir, 'train')
        self.out_labels_val_dir = os.path.join(self.out_labels_dir, 'val')

        mkdir_if_not_exist(self.out_dataset_dir)
        mkdir_if_not_exist(self.out_images_dir)
        mkdir_if_not_exist(self.out_images_train_dir)
        mkdir_if_not_exist(self.out_images_val_dir)
        mkdir_if_not_exist(self.out_labels_dir)
        mkdir_if_not_exist(self.out_labels_train_dir)
        mkdir_if_not_exist(self.out_labels_val_dir)

    @staticmethod
    def convert(iw, ih, box):
        """
        将标注的xml文件标注转换为darknet形的坐标
        """
        iw = float(iw)
        ih = float(ih)
        dw = 1. / iw
        dh = 1. / ih
        x = (box[0] + box[2]) / 2.0 - 1
        y = (box[1] + box[3]) / 2.0 - 1
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def generate_file(self, file_path):
        print('[Info] 处理数据开始: {}'.format(file_path))
        data_line = read_file(file_path)[0]
        data_dict = json.loads(data_line)
        print('[Info] keys: {}'.format(data_dict.keys()))
        images = data_dict['images']

        id_name_dict = {}
        for idx, img in enumerate(images):
            img_id = img['id']
            image_name = img['file_name'].split('.')[0]
            height = img['height']
            width = img['width']

            print('[Info] img: {}'.format(img))
            print('[Info] img_id: {}, file_name: {}'.format(img_id, image_name))
            id_name_dict[img_id] = [image_name, height, width]
            # if idx == 20:
            #     break

        annotations = data_dict["annotations"]

        image_dict = collections.defaultdict(list)
        for idx, anno in enumerate(annotations):
            print('[Info] idx: {}'.format(idx))
            image_id = anno['image_id']
            image_name, ih, iw = id_name_dict[image_id]
            wh_box = anno['bbox']
            bbox = [wh_box[0], wh_box[1], wh_box[0] + wh_box[2], wh_box[1] + wh_box[3]]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            # image_url = self.url_format.format(image_name)
            # is_ok, img_bgr = download_url_img(image_url)
            # draw_box(img_bgr, bbox, is_show=True)  # bbox绘制正确
            bbox_yolo = self.convert(iw, ih, bbox)
            bbox_yolo = [str(round(i, 6)) for i in bbox_yolo]
            print('[Info] image_id: {}, ih: {}, iw: {}, bbox: {}, bbox_yolo: {}'
                  .format(image_name, ih, iw, bbox, bbox_yolo))

            image_dict[image_id].append(bbox_yolo)

            # img_path = os.path.join(self.out_img_dir, '{}.jpg'.format(image_name))
            # cv2.imwrite(img_path, img_bgr)  # 写入图像

            # lbl_path = os.path.join(self.out_lbl_dir, '{}.txt'.format(image_name))
            # data_line = " ".join(["0", *bbox_yolo])
            # write_line(lbl_path, data_line)
            # print('[Info] ' + "-" * 50)
            # if idx == 20:
            #     break
        print('[Info] 样本数: {}'.format(len(image_dict.keys())))

        image_name_list = list(image_dict.keys())
        gap = len(image_name_list) // 10
        image_train_list = image_name_list[:gap*9]
        image_val_list = image_name_list[gap*9:]
        print('[Info] 训练: {}, 验证: {}'.format(len(image_train_list), len(image_val_list)))


def main():
    file_path = os.path.join(DATA_DIR, 'ps_datasets_v2_raw', 'instances_default_1_01.json')
    dg2 = DatasetGeneratorV2()
    dg2.generate_file(file_path=file_path)


if __name__ == '__main__':
    main()