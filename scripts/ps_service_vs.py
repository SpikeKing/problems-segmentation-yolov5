#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 19.1.21
"""

import os
import sys

from multiprocessing.pool import Pool

from myutils.project_utils import *
from root_dir import DATA_DIR
from x_utils.vpf_utils import get_problem_segmentation_vpf_service, get_problem_segmentation_vpf_service_v2


class PsServiceVS(object):
    def __init__(self):
        self.file_path = os.path.join(DATA_DIR, "url_0108_高航.txt")
        self.out1_path = os.path.join(DATA_DIR, "url_0108_高航.out1.txt")
        create_file(self.out1_path)
        self.out2_path = os.path.join(DATA_DIR, "url_0108_高航.out2.txt")
        create_file(self.out2_path)

    @staticmethod
    def process_url(idx, url, out1_path, out2_path):
        print('[Info] url: {}'.format(url))
        try:
            data_dict1 = get_problem_segmentation_vpf_service_v2(url)
            data_dict2 = get_problem_segmentation_vpf_service(url)
            out_img_url1 = data_dict1["data"]["oss_url_out2"]
            out_img_url2 = data_dict2["data"]["oss_url_out2"]
        except Exception as e:
            return
        print("out_img_url1: {}".format(out_img_url1))
        print("out_img_url2: {}".format(out_img_url2))
        write_line(out1_path, out_img_url1)
        write_line(out2_path, out_img_url2)
        print('[Info] 处理完成: {}'.format(idx))

    def process(self):
        data_lines = read_file(self.file_path)

        p = Pool(processes=10)
        for idx, data_line in enumerate(data_lines):
            url = data_line.split("?")[0]
            # self.process_url(url, self.out1_path, self.out2_path)
            p.apply_async(PsServiceVS.process_url, (idx, url, self.out1_path, self.out2_path))
        p.close()
        p.join()
        print('[Info] 处理完成: {}'.format(self.file_path))


def main():
    psv = PsServiceVS()
    psv.process()


if __name__ == '__main__':
    main()
