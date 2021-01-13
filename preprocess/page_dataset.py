#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 5.1.21
"""

import os
import sys
import random

from myutils.project_utils import *
from root_dir import DATA_DIR


class PageDataset(object):
    def __init__(self):
        self.index = 0
        self.data_raw_dir = os.path.join(DATA_DIR, 'page_dataset_raw')
        self.data_mini_file = os.path.join(DATA_DIR, 'page_dataset_mini_{}.txt'.format(self.index))

    def process(self):
        print('[Info] 待处理文件夹: {}'.format(self.data_raw_dir))
        paths_list, names_list = traverse_dir_files(self.data_raw_dir)
        paths_list = [paths_list[self.index]]
        print('[Info] 数据行数: {}'.format(len(paths_list)))

        data_lines = []
        for path, name in zip(paths_list, names_list):
            sub_lines = read_file(path)
            data_lines += sub_lines

        print('[Info] 总行数: {}'.format(len(data_lines)))

        random.seed(47)
        data_lines = list(set(data_lines))
        random.shuffle(data_lines)

        out_dir = os.path.join(DATA_DIR, 'pages_2021_01_12_split')
        mkdir_if_not_exist(out_dir)

        num = 1000
        out_file_format = os.path.join(out_dir, "cases_{}_{}.txt")
        s_num, e_num = 0, num
        for i, data_line in enumerate(data_lines):
            if s_num <= i < e_num:
                out_file = out_file_format.format(s_num+1, e_num)
            else:
                s_num = e_num
                e_num += num
                e_num = e_num if e_num < len(data_lines) else len(data_lines)
                out_file = out_file_format.format(s_num+1, e_num)
            write_line(out_file, data_line)

        print('[Info] 写入文件完成! {}'.format(self.data_mini_file))


def main():
    pd = PageDataset()
    pd.process()


if __name__ == '__main__':
    main()
