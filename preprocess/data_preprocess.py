#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 23.12.20
"""

import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from myutils.cv_utils import *
from root_dir import DATA_DIR
from x_utils.vpf_utils import get_trt_rotation_vpf_service
from x_utils.oss_utils import save_img_2_oss


class DataPreprocess(object):
    """
    处理标注数据
    """
    def __init__(self):
        pass

    @staticmethod
    def process_line(idx, data_line, out_file):
        items = data_line.split("<sep>")
        url = items[1]
        url = url.split('?')[0]
        img_name = url.split('/')[-1]
        # print('[Info] url: {}'.format(url))
        data_dict = get_trt_rotation_vpf_service(url)
        # print('[Info] data_dict: {}'.format(data_dict))
        angle = data_dict['data']['data']['angle']
        # print('[Info] angle: {}'.format(angle))
        is_ok, img_bgr = download_url_img(url)
        out_img = rotate_img_for_4angle(img_bgr, angle)  # 旋转角度
        # show_img_bgr(out_img)
        url = save_img_2_oss(out_img, img_name, "zhengsheng.wcl/problems_segmentation/datasets/prelabeled-20201223")
        write_line(out_file, url)
        print('[Info] idx: {}'.format(idx))

    def process(self):
        data_dir = os.path.join(DATA_DIR, '2020_12_23')
        print('[Info] 数据文件: {}'.format(data_dir))

        out_file = os.path.join(DATA_DIR, '2020_12_23.txt')
        print('[Info] 写出文件: {}'.format(out_file))

        paths_list, names_list = traverse_dir_files(data_dir)
        print('[Info] 文件数: {}'.format(len(paths_list)))

        data_lines = []
        for path, name in zip(paths_list, names_list):
            sub_lines = read_file(path)
            data_lines += sub_lines
        print('[Info] 文本行数: {}'.format(len(data_lines)))

        # pool = Pool(processes=80)
        for idx, data_line in enumerate(data_lines):
            DataPreprocess.process_line(idx, data_line, out_file)
            # pool.apply_async(DataPreprocess.process_line, (idx, data_line, out_file))

        # pool.close()
        # pool.join()
        print('[Info] 处理完成: {}'.format(out_file))


def main():
    dp = DataPreprocess()
    dp.process()


if __name__ == '__main__':
    main()
