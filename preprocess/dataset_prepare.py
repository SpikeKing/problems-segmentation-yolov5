#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 26.12.20
"""

import os
import sys

from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR


class DatasetPrepare(object):
    def __init__(self):
        pass

    def process(self):
        img_path = os.path.join(DATA_DIR, 'test_400_res.right.e0.csv')

        out_dir = os.path.join(DATA_DIR, 'test_400_out')
        mkdir_if_not_exist(out_dir)

        data_lines = read_file(img_path)
        for idx, data_line in enumerate(data_lines):
            if idx == 0:
                continue
            items = data_line.split(",")
            url = items[0]
            angle = items[1]

            img_name = url.split('/')[-1]
            img_name = os.path.join(out_dir, img_name)
            is_ok, img_bgr = download_url_img(url)
            img_bgr = rotate_img_for_4angle(img_bgr, angle)
            out_path = os.path.join(out_dir, img_name)
            cv2.imwrite(out_path, img_bgr)
            print('[Info] out_path: {}'.format(out_path))


def main():
    dp = DatasetPrepare()
    dp.process()


if __name__ == '__main__':
    main()