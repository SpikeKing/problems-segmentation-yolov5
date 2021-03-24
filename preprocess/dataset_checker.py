#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 23.3.21
"""

import os
import sys

from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR


class DatasetChecker(object):
    def __init__(self):
        self.img_format = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/yjb219735/ori_imgs/{}"

    def check_dataset(self):
        # check_path = os.path.join(DATA_DIR, 'jiabao_dataset', '7_train.txt')  # 未拼接
        # check_path = os.path.join(DATA_DIR, 'jiabao_dataset', '8_train.txt')  # 已经拼接
        check_path = os.path.join(DATA_DIR, 'jiabao_dataset', 'data_check_5k.txt')  # 已经拼接
        data_lines = read_file(check_path)
        for idx, data_line in enumerate(data_lines):
            print('[Info] idx: {}, data_line: {}'.format(idx, data_line))
            data_dict = json.loads(data_line)
            # url = self.img_format.format(data_dict['url'])
            url = data_dict['url']
            is_ok, img_bgr = download_url_img(url)
            # show_img_bgr(img_bgr)
            coord_list = data_dict['coord']
            label_list = data_dict['label']
            # 1 小标题；2 大标题；3 标题；4 杂质；5 题目
            for coord, label in zip(coord_list, label_list):
                if label == 5:
                    coord = [int(x) for x in coord]
                    bbox = [coord[0], coord[1], coord[0]+coord[2], coord[1]+coord[3]]
                    draw_box(img_bgr, bbox, is_show=True)
                    break
            break


def main():
    dc = DatasetChecker()
    dc.check_dataset()


if __name__ == '__main__':
    main()
