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

from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.cv_utils import draw_box
from myutils.project_utils import *
from root_dir import DATA_DIR


class DatasetGeneratorV2(object):
    """
    数据集生成
    """
    def __init__(self):
        pass

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

    @staticmethod
    def generate_file(file_path, file_name):
        print('[Info] file_path: {}'.format(file_path))

        url_format = "http://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/problems_segmentation/" \
                          "datasets/prelabeled-20201224/{}.jpg"
        out_dataset_dir = os.path.join(DATA_DIR, 'ps_datasets_v2')

        out_images_dir = os.path.join(out_dataset_dir, 'images')
        out_images_train_dir = os.path.join(out_images_dir, 'train')
        out_images_val_dir = os.path.join(out_images_dir, 'val')

        out_labels_dir = os.path.join(out_dataset_dir, 'labels')
        out_labels_train_dir = os.path.join(out_labels_dir, 'train')
        out_labels_val_dir = os.path.join(out_labels_dir, 'val')

        mkdir_if_not_exist(out_dataset_dir)
        mkdir_if_not_exist(out_images_dir)
        mkdir_if_not_exist(out_images_train_dir)
        mkdir_if_not_exist(out_images_val_dir)
        mkdir_if_not_exist(out_labels_dir)
        mkdir_if_not_exist(out_labels_train_dir)
        mkdir_if_not_exist(out_labels_val_dir)

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

            # print('[Info] img: {}'.format(img))
            # print('[Info] img_id: {}, file_name: {}'.format(img_id, image_name))
            id_name_dict[img_id] = [image_name, height, width]
            # if idx == 20:
            #     break

        annotations = data_dict["annotations"]

        image_dict = collections.defaultdict(list)
        for idx, anno in enumerate(annotations):
            image_id = anno['image_id']
            image_name, ih, iw = id_name_dict[image_id]
            wh_box = anno['bbox']
            bbox = [wh_box[0], wh_box[1], wh_box[0] + wh_box[2], wh_box[1] + wh_box[3]]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            bbox_yolo = DatasetGeneratorV2.convert(iw, ih, bbox)
            bbox_yolo = [str(round(i, 6)) for i in bbox_yolo]
            # print('[Info] image_id: {}, ih: {}, iw: {}, bbox: {}, bbox_yolo: {}'
            #       .format(image_name, ih, iw, bbox, bbox_yolo))

            image_dict[image_name].append(" ".join(["0", *bbox_yolo]))

        print('[Info] 样本数: {}'.format(len(image_dict.keys())))

        image_name_list = list(image_dict.keys())
        gap = len(image_name_list) // 10
        image_train_list = image_name_list[:gap*9]
        image_val_list = image_name_list[gap*9:]
        print('[Info] 训练: {}, 验证: {}'.format(len(image_train_list), len(image_val_list)))

        for idx, image_name in enumerate(image_train_list):
            print('[Info] idx: {}'.format(idx))
            bbox_yolo_list = image_dict[image_name]

            image_url = url_format.format(image_name)
            is_ok, img_bgr = download_url_img(image_url)
            img_path = os.path.join(out_images_train_dir, '{}.jpg'.format(image_name))
            cv2.imwrite(img_path, img_bgr)  # 写入图像

            lbl_path = os.path.join(out_labels_train_dir, '{}.txt'.format(image_name))
            write_list_to_file(lbl_path, bbox_yolo_list)
            print('[Info] ' + "-" * 100)
            if idx == 20:
                break

        for idx, image_name in enumerate(image_val_list):
            print('[Info] idx: {}'.format(idx))
            bbox_yolo_list = image_dict[image_name]

            image_url = url_format.format(image_name)
            is_ok, img_bgr = download_url_img(image_url)
            img_path = os.path.join(out_images_val_dir, '{}.jpg'.format(image_name))
            cv2.imwrite(img_path, img_bgr)  # 写入图像

            lbl_path = os.path.join(out_labels_val_dir, '{}.txt'.format(image_name))
            write_list_to_file(lbl_path, bbox_yolo_list)
            print('[Info] ' + "-" * 100)
            if idx == 20:
                break
        print('[Info] 处理完成! {}'.format(file_path))


def process():
    dir_path = os.path.join(DATA_DIR, 'ps_datasets_v2_raw_mini')
    paths_list, names_list = traverse_dir_files(dir_path)

    pool = Pool(processes=10)

    for path, name in zip(paths_list, names_list):
        # DatasetGeneratorV2.generate_file(path)
        print('[Info] path: {}'.format(path))
        pool.apply_async(DatasetGeneratorV2.generate_file, (path, name))

    pool.close()
    pool.join()
    print('[Info] 全部处理完成: {}'.format(dir_path))


def main():
    process()


if __name__ == '__main__':
    main()