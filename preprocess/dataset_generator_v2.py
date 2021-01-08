#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 6.1.21
"""

import os
import sys
import json

from myutils.project_utils import *
from root_dir import DATA_DIR


class DatasetGeneratorV2(object):
    def __init__(self):
        self.url_format = "http://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/problems_segmentation/" \
                          "datasets/prelabeled-20201224/{}"
        self.out_dir = os.path.join(DATA_DIR, 'ps_datasets_v2')
        mkdir_if_not_exist(self.out_dir)
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

    def generate_file(self, file_path):
        print('[Info] 处理数据开始: {}'.format(file_path))
        data_line = read_file(file_path)[0]
        data_dict = json.loads(data_line)
        print('[Info] keys: {}'.format(data_dict.keys()))
        images = data_dict['images']

        id_name_dict = {}
        for idx, img in enumerate(images):
            img_id = img['id']
            file_name = img['file_name'].split('.')[0]
            height = img['height']
            width = img['width']

            print('[Info] img: {}'.format(img))
            print('[Info] img_id: {}, file_name: {}'.format(img_id, file_name))
            id_name_dict[img_id] = [file_name, height, width]
            if idx == 20:
                break

        annotations = data_dict["annotations"]
        for idx, anno in enumerate(annotations):
            image_id = anno['image_id']
            bbox = anno['bbox']
            print(anno)
            if idx == 20:
                break


def main():
    file_path = os.path.join(DATA_DIR, 'ps_datasets_v2_raw', 'instances_default_1_01.json')
    dg2 = DatasetGeneratorV2()
    dg2.generate_file(file_path=file_path)


if __name__ == '__main__':
    main()