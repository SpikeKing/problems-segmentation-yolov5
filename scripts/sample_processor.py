#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 12.1.21
"""

import os
import cv2

from myutils.project_utils import *
from root_dir import DATA_DIR
from x_utils.vpf_utils import get_problem_segmentation_vpf_service


class SampleProcessor(object):
    def __init__(self):
        self.input_file = os.path.join(DATA_DIR, 'url_0108_高航.txt')
        print('[Info] input_file: {}'.format(self.input_file))
        self.out_dir = os.path.join(DATA_DIR, 'url_0108_高航_out')
        mkdir_if_not_exist(self.out_dir)

    def process(self):
        data_lines = read_file(self.input_file)
        paths_list, names_list = traverse_dir_files(self.out_dir)
        print('[Info] 样本数: {}'.format(len(data_lines)))
        for idx, data_line in enumerate(data_lines):
            print('[Info] idx: {}'.format(idx))
            img_url = data_line.split("?")[0]
            img_name = img_url.split('/')[-1]
            if img_name in names_list:
                print('[Info] 已处理: {}'.format(img_name))
                continue
            print('[Info] img_url: {}, img_name: {}'.format(img_url, img_name))
            try:
                res_dict = get_problem_segmentation_vpf_service(img_url)
                oss_url_out1 = res_dict['data']['oss_url_out1']
                print('[Info] out_url: {}'.format(oss_url_out1))
                is_ok, img_bgr = download_url_img(oss_url_out1)
            except Exception as e:
                is_ok, img_bgr = download_url_img(img_url)
                print('[Error] e: {}'.format(e))
            out_path = os.path.join(self.out_dir, img_name)
            print('[Info] out_path: {}'.format(out_path))
            try:
                cv2.imwrite(out_path, img_bgr)
            except Exception as e:
                print('[Error] e: {}'.format(e))
                continue
            print('[Info] ' + '-' * 50)


def main():
    sp = SampleProcessor()
    sp.process()


if __name__ == '__main__':
    main()
