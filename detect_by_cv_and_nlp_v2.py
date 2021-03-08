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
import base64
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
from multiprocessing.pool import Pool

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR
from x_utils.vpf_utils import get_ocr_vpf_service, get_next_sentences_vpf_service, get_problem_segmentation_vpf_service


class ImgDetector(object):
    """
    图像检测
    """
    def __init__(self):
        self.weights = "mydata/models/best-20210118.pt"  # 当前线上
        # self.weights = "mydata/models/best-20201224.pt"
        # self.weights = "mydata/models/best_20210303.2.pt"

        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45

        self.device = select_device()  # 自动选择环境
        self.is_half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model, self.img_size = self.load_model()  # 加载模型

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
        img_bgr = letterbox(img_bgr, new_shape=self.img_size)[0]

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
        conf_list = []  # 置信度列表

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
                conf_list.append(conf)

        return box_list, conf_list

    @staticmethod
    def filer_boxes_by_size(boxes, r_thr=0.5):
        """
        根据是否重叠过滤包含在内部的框
        """
        if not boxes:
            return boxes, [i for i in range(len(boxes))]

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
            # print('[Info] i: {}, x_boxes: {}'.format(i, x_boxes))
            sorted_boxes[i] = merge_boxes(x_boxes)

        new_boxes, new_idxes = [], []
        for i in range(n_box):
            if flag_list[i]:
                new_boxes.append(sorted_boxes[i])
                new_idxes.append(sorted_idxes[i])

        return new_boxes, new_idxes

    def filter_word_box(self, img_bgr, problem_boxes, word_boxes):
        print('[Info] 题目块: {}, 文字块: {}'.format(len(problem_boxes), len(word_boxes)))
        # draw_box_list(img_bgr, word_boxes)
        n_problem = len(problem_boxes)
        n_word = len(word_boxes)

        flag_list = [True] * n_word

        for i in range(n_problem):
            box1 = problem_boxes[i]
            for j in range(n_word):
                box2 = word_boxes[j]
                r_iou = min_iou(box1, box2)
                if r_iou >= 0.5:
                    flag_list[j] = False

        new_boxes = []
        new_idxes = []
        for i in range(n_word):
            if flag_list[i]:
                new_boxes.append(word_boxes[i])
                new_idxes.append(i)

        # draw_box_list(img_bgr, new_boxes)
        # print('[Info] 输出文字块数: {}'.format(len(new_boxes)))
        return new_boxes, new_idxes

    @staticmethod
    def parse_pos(pos_list):
        """
        处理点
        """
        point_list = []
        for pos in pos_list:
            x = pos['x']
            y = pos['y']
            point_list.append([x, y])
        return point_list

    @staticmethod
    def get_ocr_service_info(url):
        """
        处理URL
        """
        print('[Info] process url: {}'.format(url))
        res_dict = get_ocr_vpf_service(url)
        # print('[Info] res_dict: {}'.format(res_dict))
        data_dict = res_dict['data']['data']
        # print('[Info] data_dict: {}'.format(data_dict))
        word_num = data_dict['wordNum']
        # print('[Info] word_num: {}'.format(word_num))
        words_info = data_dict['wordsInfo']
        angle = data_dict['angle']

        content = data_dict['content']

        # color_list = generate_colors(word_num)
        box_list, word_list, idx_list = [], [], []

        new_words_info = []
        for idx, words_data in enumerate(words_info):
            # print('[Info] words_data: {}'.format(words_data))
            word = words_data["word"]
            rec_classify = words_data["recClassify"]
            # print('[Info] word: {}'.format(word))
            pos = words_data["pos"]
            # print('[Info] pos: {}'.format(pos))
            prob = words_data["prob"]
            if prob == 0:  # 过滤概率等于0的数据
                continue
            # 直接保留公式
            # if rec_classify != 0:
            #     word = u"$${}$$".format(word)
            # if prob == 0:
            #     continue
            word_rec = ImgDetector.parse_pos(pos)
            # print('[Info] point_list: {}'.format(word_rec))
            box_list.append(word_rec)
            word_list.append(word)
            idx_list.append(idx)
            new_words_info.append(words_data)

        return new_words_info, box_list, word_list, idx_list, data_dict

    @staticmethod
    def analyze_layout(l2r_boxes, head_banner_boxes, tail_banner_boxes,
                       h, w, is_ud=True, img_bgr=None):
        """
        分析版面，从左到右的box列
        """
        r2l_boxes = copy.copy(l2r_boxes)
        r2l_boxes.reverse()
        col_box_list = []

        start_h, end_h = 0, h

        head_col_box = []
        if head_banner_boxes:
            head_banner = merge_boxes(head_banner_boxes)
            head_col_box = [head_banner[0], 0, head_banner[2], head_banner[3]]
            start_h = head_banner[3]

        tail_col_box = []
        if tail_banner_boxes:
            tail_banner = merge_boxes(tail_banner_boxes)
            tail_col_box = [tail_banner[0], tail_banner[1], tail_banner[2], h]
            end_h = tail_banner[1]

        for boxes in r2l_boxes:
            x_list, y_list = [], []
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                x_list.append(x_min)
                x_list.append(x_max)
                y_list.append(y_min)
                y_list.append(y_max)
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            if not is_ud:
                col_box = [x_min, y_min, x_max, y_max]
            else:
                col_box = [x_min, start_h, x_max, end_h]
            # draw_box(img_bgr, col_box, is_show=True)
            print('[Info] col_box: {}'.format(col_box))
            col_box_list.append(col_box)

        col_box_list.reverse()
        col_box_list, _ = ImgDetector.filer_boxes_by_size(col_box_list)

        # 版面统一左右排列
        sub_col_box_list, _, _ = sorted_boxes_by_row(col_box_list)
        col_box_list = []
        for sub_col_box in sub_col_box_list:
            col_box_list += sub_col_box

        if head_banner_boxes:
            col_box_list = [head_col_box] + col_box_list

        if tail_banner_boxes:
            col_box_list = col_box_list + [tail_col_box]

        return col_box_list

    def filer_boxes_by_col(self, col_boxes, sub_boxes):
        """
        根据列的范围过滤boxes
        """
        col_sub_boxes = []
        col_sub_idxes = []
        for col_box in col_boxes:
            one_col_boxes = []
            one_col_idxes = []
            for idx, word_box in enumerate(sub_boxes):
                v_iou = min_iou(word_box, col_box)
                if v_iou >= 0.5:
                    one_col_boxes.append(word_box)
                    one_col_idxes.append(idx)
            col_sub_boxes.append(one_col_boxes)
            col_sub_idxes.append(one_col_idxes)
        return col_sub_boxes, col_sub_idxes

    def get_box_h_dist_ratio(self, box1, box2, img_h):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        diff = y2_min - y1_max
        r_diff = safe_div(diff, img_h)
        # print('[Info] diff: {}, r_diff: {}'.format(diff, r_diff))
        return r_diff

    def get_box_w_dist_ratio(self, box1, box2, img_w):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        diff = max(x2_min, x1_min) - min(x1_max, x2_max)
        r_diff = safe_div(diff, img_w)
        # print('[Info] diff: {}, r_diff: {}'.format(diff, r_diff))
        return r_diff

    def merge_problem_by_sentences(self, img_bgr, x_box_list, x_str_list):
        """
        根据上下句服务合并问题
        """
        # draw_box_list(img_bgr, x_box_list, is_show=True, save_name="problem_sentences_lines.jpg")
        print('[Info] str: {}'.format("\n".join(x_str_list)))
        img_h, img_w, _ = img_bgr.shape
        n_box = len(x_box_list)
        rh_thr = 0.02
        rw_thr = 0.6
        thr_idx_list = []
        query_list = []
        for i in range(0, n_box-1):
            str1 = x_str_list[i]
            str2 = x_str_list[i+1]
            box1 = x_box_list[i]
            box2 = x_box_list[i + 1]
            rh_diff = self.get_box_h_dist_ratio(box1, box2, img_h)
            rw_diff = self.get_box_w_dist_ratio(box1, box2, img_w)
            print('[Info] str1: {}, str2: {}'.format(len(str1), len(str2)))
            if rh_diff < rh_thr and rw_diff < rw_thr and len(str1) > 8 and len(str2) > 8:
                thr_idx_list.append(i)
                str1 = str1.replace("'", "")
                str2 = str2.replace("'", "")
                query_list.append({"sentence1": "{}".format(str1),
                                   "sentence2": "{}".format(str2)})

        next_flags = [False] * n_box
        if query_list:
            print('[Info] query_list: {}'.format(query_list))
            query_list_str = json.dumps(query_list).replace('\n', '').replace("'", '"')
            print(query_list_str)
            service_res = get_next_sentences_vpf_service(query_list_str)
            print('[Info] service_res: {}'.format(service_res))
            prob_list = service_res['data']['query_result']

            for x, i in enumerate(thr_idx_list):
                prob = prob_list[x]
                if prob > 0.5:
                    next_flags[i] = True
                    next_flags[i+1] = True

        s_idx, e_idx = -1, 0
        p_idx_list = []

        for idx, n_flag in enumerate(next_flags):
            if idx != 0:  # 判断是否Close
                box1 = x_box_list[idx - 1]
                box2 = x_box_list[idx]
                rh_diff = self.get_box_h_dist_ratio(box1, box2, img_h)
                rw_diff = self.get_box_w_dist_ratio(box1, box2, img_w)
                is_close = rh_diff < rh_thr and rw_diff < rw_thr
            else:
                is_close = True

            if not is_close:  # 非Close, 直接添加
                if e_idx - s_idx > 1:
                    idx_list = [i for i in range(s_idx + 1, e_idx + 1)]
                    p_idx_list.append(idx_list)
                s_idx = idx - 1
                e_idx = idx

            if n_flag:
                e_idx = idx
            else:
                if e_idx - s_idx > 1:
                    idx_list = [i for i in range(s_idx+1, e_idx+1)]
                    p_idx_list.append(idx_list)
                    p_idx_list.append([idx])
                else:
                    p_idx_list.append([idx])
                s_idx = idx
                e_idx = idx

        if e_idx - s_idx > 1:
            idx_list = [i for i in range(s_idx + 1, e_idx + 1)]
            p_idx_list.append(idx_list)

        print('[Info] next_flags: {}'.format(next_flags))
        print('[Info] p_idx_list: {}'.format(p_idx_list))

        sentence_problem = []  # 至少两个框，认为是一个题
        l_box_list = []
        for p_list in p_idx_list:
            box_list = filter_list_by_idxes(x_box_list, p_list)
            l_box = merge_boxes(box_list)
            l_box_list.append(l_box)
            if len(box_list) >= 2:
                sentence_problem.append(l_box)  # 文本题目框

        # draw_box_list(img_bgr, l_box_list, is_show=True, save_name="problem_sentences.jpg")
        return l_box_list, sentence_problem

    @staticmethod
    def add_tag_to_wordinfo_v2(col_boxes, problem_boxes, boxes_confs, words_info, img_bgr=None):
        # 初始化
        for words_data in words_info:
            words_data['columnIdx'] = -1
            words_data['blockIdx'] = -1
            words_data['lineIdx'] = -1

        # ---------- 先切分题目框 ----------#
        prb_idx_dict = collections.defaultdict(list)
        rank_prb_tag_list = []
        word_boxes = []
        for idx, words_data in enumerate(words_info):
            # pos = words_data["pos"]
            # word_rec = ImgDetector.parse_pos(pos)
            # word_box = rec2box(word_rec)
            # word_boxes.append(word_box)
            word_box = words_data['rotated_box']
            word_boxes.append(word_box)
            current_idx = -1
            is_used = False
            for prb_idx, prb_box in enumerate(problem_boxes):
                prb_iou = min_iou(prb_box, word_box)
                if prb_iou >= 0.35 and not is_used:
                    current_idx = prb_idx
                    p_tag = "p_{}".format(current_idx)
                    prb_idx_dict[p_tag].append(idx)
                    is_used = True

            if current_idx == -1:
                rank_prb_tag_list.append("{}".format(idx))
            else:
                p_tag = "p_{}".format(current_idx)
                if p_tag not in rank_prb_tag_list:
                    rank_prb_tag_list.append(p_tag)
        # 测试
        # draw_box_list(img_bgr, word_boxes, is_show=True, save_name="tmp-{}.jpg".format(get_current_time_str()))

        # 切分栏
        col_prb_box_dict = collections.defaultdict(list)
        rank_col_tag_list = []
        col_block_info_dict = collections.defaultdict(list)

        for item in rank_prb_tag_list:
            current_idx = -1
            for col_tag, col_box in enumerate(col_boxes):
                if item.startswith("p"):
                    p_id = int(item.split('_')[-1])
                    prb_box = problem_boxes[p_id]
                    prb_conf = boxes_confs[p_id]
                    col_iou = min_iou(col_box, prb_box)
                    if col_iou >= 0.5:
                        current_idx = col_tag
                        c_tag = "c_{}".format(col_tag)
                        col_prb_box_dict[c_tag].append(item)
                        if prb_conf == -1:
                            conf = 1.0
                            b_type = "text"
                        else:
                            conf = prb_conf
                            b_type = "vision"
                        data_dict = {
                            "box": prb_box,
                            "conf": conf,
                            "type": b_type
                        }
                        col_block_info_dict[col_tag].append(data_dict)
                else:
                    b_id = int(item)
                    w_box = word_boxes[b_id]
                    col_iou = min_iou(col_box, w_box)
                    if col_iou >= 0.5:
                        current_idx = col_tag
                        c_tag = "c_{}".format(col_tag)
                        col_prb_box_dict[c_tag].append(item)

            if current_idx == -1:
                rank_col_tag_list.append(item)
            else:
                c_tag = "c_{}".format(current_idx)
                if c_tag not in rank_col_tag_list:
                    rank_col_tag_list.append(c_tag)
        col_block_info_list = sort_dict_by_key(col_block_info_dict, reverse=False)
        res_blocks_info = []
        for _, col_blocks_info in col_block_info_list:
            res_blocks_info.append(col_blocks_info)

        # 根据位置优化列顺序
        re_rank_col_tag_list, other_rank_col_tag_list = [], []
        c_idx = 0
        for item in rank_col_tag_list:
            if item.startswith('c'):
                re_rank_col_tag_list.append("c_{}".format(c_idx))
                c_idx += 1
            else:
                other_rank_col_tag_list.append(item)
        re_rank_col_tag_list = re_rank_col_tag_list + other_rank_col_tag_list
        rank_col_tag_list = re_rank_col_tag_list

        rank_col_idx_list = []  # 纯idx列表
        for col_tag in rank_col_tag_list:
            if col_tag.startswith("c"):
                col_tag_list = col_prb_box_dict[col_tag]
                tmp_col_idx_list = []
                for prb_tag in col_tag_list:
                    if not prb_tag.startswith("p"):
                        tmp_col_idx_list.append([int(prb_tag)])
                    else:
                        prb_tag = prb_idx_dict[prb_tag]
                        tmp_col_idx_list.append(prb_tag)
                rank_col_idx_list.append(tmp_col_idx_list)
            elif col_tag.startswith("p"):
                prb_idx = col_tag.split('_')[-1]
                prb_tag = prb_idx_dict[prb_idx]
                rank_col_idx_list.append([prb_tag])
            else:
                rank_col_idx_list.append([[int(col_tag)]])

        # ---------- 切分行块 ----------#
        rank_idx_list = []  # 重排idx
        for col_idxes in rank_col_idx_list:
            tmp_col_idx_list = []
            for prb_idxes in col_idxes:
                if len(prb_idxes) == 1:
                    tmp_col_idx_list.append([prb_idxes])
                    continue
                boxes_list = [word_boxes[i] for i in prb_idxes]
                # draw_box_list(img_bgr, boxes_list, save_name="tmp-{}.jpg".format(get_current_time_str()))
                sorted_boxes, sorted_idxes, num_row = sorted_boxes_by_row(boxes_list, img_bgr)
                rank_prb_idxes = filter_list_by_idxes(prb_idxes, sorted_idxes)
                tmp_col_idx_list.append(rank_prb_idxes)
            rank_idx_list.append(tmp_col_idx_list)

        # 添加标签
        ranked_words_info, other_words_info = [], []  # 优质加杂质
        # diff_col_num = 0
        col_blocks_list = []
        ranked_content, other_content = "", ""

        for col_tag in rank_col_tag_list:
            if col_tag.startswith("c"):
                prb_tag_list = col_prb_box_dict[col_tag]
                prb_num = len(prb_tag_list)  # 问题块数
                col_idx = int(col_tag.split("_")[-1])
                col_blocks_list.append(prb_num)
            else:
                if col_tag.startswith("p"):  # 不在栏中的题目，需要重新解析
                    prb_tag = col_tag
                    prb_idx_list = prb_idx_dict[prb_tag]  # 直接是idx
                    for word_idx in prb_idx_list:
                        word_data = words_info[word_idx]
                        other_content += word_data["word"] + " "
                        other_words_info.append(word_data)
                else:
                    word_idx = int(col_tag)
                    word_data = words_info[word_idx]
                    other_content += word_data["word"] + " "
                    other_words_info.append(word_data)
                continue

            prb_idx = 0  # 每一栏单独计数
            for prb_tag in prb_tag_list:
                if prb_tag.startswith("p"):
                    prb_idx_list = prb_idx_dict[prb_tag]  # 直接是idx
                    boxes_list = [word_boxes[i] for i in prb_idx_list]
                    sorted_boxes, sorted_idxes, num_row = sorted_boxes_by_row(boxes_list)
                    nested_line_idx_list = filter_list_by_idxes(prb_idx_list, sorted_idxes)
                else:
                    word_idx = int(prb_tag)
                    word_data = words_info[word_idx]
                    other_content += word_data["word"] + " "
                    other_words_info.append(word_data)
                    continue

                for line_idx, line_idx_list in enumerate(nested_line_idx_list):
                    for idx in line_idx_list:
                        word_data = words_info[idx]
                        word_data['lineIdx'] = line_idx
                        word_data['blockIdx'] = prb_idx
                        word_data['columnIdx'] = col_idx
                        ranked_content += word_data["word"] + " "
                        ranked_words_info.append(word_data)
                prb_idx += 1  # 题目编号依次排列

        ranked_words_info = ranked_words_info + other_words_info
        ranked_content = ranked_content.strip() + " " + other_content.strip()

        column_num, block_num_list = len(col_boxes), col_blocks_list
        return ranked_words_info, column_num, block_num_list, res_blocks_info, ranked_content

    @staticmethod
    def filter_banner_box(boxes, conf_list, w):
        """
        输出通栏boxes
        """
        if len(boxes) <= 5:
            return [], [], boxes, [], [], conf_list

        x_min_list, y_min_list = [], []

        # 从左到右(lr)、从上到下(ud)排序
        for box in boxes:
            x_min_list.append(box[0])
            y_min_list.append(box[1])

        box_lr_idxes = np.argsort(x_min_list)
        box_ud_idxes = np.argsort(y_min_list)

        banner_thr = 0.7  # 通栏的宽度 大于 总宽度 的 70%
        none_banner_thr = 0.4

        head_banner_boxes, tail_banner_boxes = [], []
        head_confs, tail_confs = [], []

        res_ud_idxes = box_ud_idxes  # 用于添加或删除boxes

        head_box = boxes[box_ud_idxes[0]]
        head_w = head_box[2] - head_box[0]
        hw_ratio = safe_div(head_w, w)

        head_box_1 = boxes[box_ud_idxes[1]]
        head_w_1 = head_box_1[2] - head_box_1[0]
        hw_ratio_1 = safe_div(head_w_1, w)

        if hw_ratio > banner_thr and hw_ratio_1 < none_banner_thr:
            head_banner_boxes.append(head_box)
            head_confs.append(conf_list[res_ud_idxes[0]])

            res_ud_idxes = res_ud_idxes[1:]

        tail_box = boxes[box_ud_idxes[-1]]
        tail_w = tail_box[2] - tail_box[0]
        tw_ratio = safe_div(tail_w, w)

        tail_box_1 = boxes[box_ud_idxes[-2]]
        tail_w_1 = tail_box_1[2] - tail_box_1[0]
        tw_ratio_1 = safe_div(tail_w_1, w)

        if tw_ratio > banner_thr and tw_ratio_1 < none_banner_thr:
            tail_banner_boxes.append(tail_box)
            tail_confs.append(conf_list[res_ud_idxes[-1]])

            res_ud_idxes = res_ud_idxes[:-1]

        normal_confs = []
        normal_boxes = []
        for idx in res_ud_idxes:
            normal_boxes.append(boxes[idx])
            normal_confs.append(conf_list[idx])

        return head_banner_boxes, tail_banner_boxes, normal_boxes, \
               head_confs, tail_confs, normal_confs

    @staticmethod
    def draw_blocks_info(img_bgr, blocks_info):
        """
        绘制块位置和概率
        """
        for col_idx, col_bi_list in enumerate(blocks_info):
            for idx, bi_dict in enumerate(col_bi_list):
                img_copy = copy.copy(img_bgr)
                b_type = bi_dict["type"]
                b_box = bi_dict["box"]
                b_conf = bi_dict["conf"]

                if b_type == "vision":
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                draw_box(img_bgr, b_box, color=color, tk=-1, is_show=False, is_new=False)
                cv2.addWeighted(img_bgr, 0.3, img_copy, 0.7, 0, img_bgr)  # 添加色块
                draw_text(img_bgr, "n: {}-{}, c: {}".format(str(col_idx), str(idx), str(b_conf)),
                          org=get_box_center(b_box), color=color)

        return img_bgr

    @staticmethod
    def draw_word_info(img_bgr, words_info):
        # 绘制图像3
        img_bgr_x = copy.copy(img_bgr)
        for idx, words_data in enumerate(words_info):
            word_box = words_data['rotated_box']
            draw_box(img_bgr_x, word_box, color=(0, 255, 0), is_show=False, is_new=False)
            draw_text(img_bgr_x, str(idx), org=get_box_center(word_box), color=(0, 0, 255))
        return img_bgr_x

    def draw_all_boxes(self, img_bgr, rank_word_info, words_info, large_col_boxes, problem_boxes, blocks_info):
        """
        绘制Boxes
        """
        # show_img_bgr(img_bgr)
        # print('[Info] 栏数: {}'.format(len(rank_idx_num_list)))

        res_col_boxes, res_prb_boxes, res_row_boxes = [], [], []

        col_box_dict = collections.defaultdict(list)
        prb_box_dict = collections.defaultdict(list)
        row_box_dict = collections.defaultdict(list)
        for idx, word_info in enumerate(rank_word_info):
            col_idx = word_info['columnIdx']
            block_idx = word_info['blockIdx']
            line_idx = word_info['lineIdx']
            # pos = word_info["pos"]
            # word_rec = ImgDetector.parse_pos(pos)
            # box = rec2box(word_rec)
            box = word_info['rotated_box']
            if col_idx == -1 or block_idx == -1 or line_idx == -1:
                continue
            col_box_dict["{}".format(col_idx)].append(box)
            prb_box_dict["{}_{}".format(col_idx, block_idx)].append(box)
            row_box_dict["{}_{}_{}".format(col_idx, block_idx, line_idx)].append(box)

        for col_key in col_box_dict.keys():
            box_list = col_box_dict[col_key]
            col_one_box = merge_boxes(box_list)
            res_col_boxes.append(col_one_box)

        for prb_key in prb_box_dict.keys():
            box_list = prb_box_dict[prb_key]
            prb_one_box = merge_boxes(box_list)
            res_prb_boxes.append(prb_one_box)

        for row_key in row_box_dict.keys():
            box_list = row_box_dict[row_key]
            row_one_box = merge_boxes(box_list)
            res_row_boxes.append(row_one_box)

        # 绘制图像1
        img_bgr1 = copy.copy(img_bgr)
        color_cols_list = generate_colors(len(res_col_boxes), seed=98)
        color_prbs_list = generate_colors(len(res_prb_boxes), seed=47)
        color_rows_list = generate_colors(len(res_row_boxes), seed=25)

        for idx, xyxy in enumerate(res_col_boxes):
            img_copy = copy.copy(img_bgr1)
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            img_bgr1 = cv2.rectangle(img_bgr1, c1, c2, color_cols_list[idx], thickness=5, lineType=cv2.LINE_AA)
            # cv2.addWeighted(img_bgr1, 0.3, img_copy, 0.7, 0, img_bgr1)  # 添加色块
            # show_img_bgr(img_bgr1)

        for idx, xyxy in enumerate(res_prb_boxes):
            img_copy = copy.copy(img_bgr1)
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            img_bgr1 = cv2.rectangle(img_bgr1, c1, c2, color_prbs_list[idx], thickness=-1, lineType=cv2.LINE_AA)
            cv2.addWeighted(img_bgr1, 0.3, img_copy, 0.7, 0, img_bgr1)  # 添加色块

        for idx, xyxy in enumerate(res_row_boxes):
            img_copy = copy.copy(img_bgr1)
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            img_bgr1 = cv2.rectangle(img_bgr1, c1, c2, color_rows_list[idx], thickness=-1, lineType=cv2.LINE_AA)
            cv2.addWeighted(img_bgr1, 0.3, img_copy, 0.7, 0, img_bgr1)  # 添加色块

        for idx, words_data in enumerate(rank_word_info):
            # pos = words_data["pos"]
            # word_rec = ImgDetector.parse_pos(pos)
            # word_box = rec2box(word_rec)
            # word_box = rank_rotated_boxes[idx]
            word_box = words_data['rotated_box']
            draw_box(img_bgr1, word_box, color=(128, 0, 128), tk=1, is_show=False, is_new=False)
            # draw_text(img_bgr1, str(idx), org=get_box_center(word_box), color=(0, 0, 255))
            word_tag_str = "{} {}_{}_{}".format(idx, words_data['columnIdx'], words_data['blockIdx'], words_data['lineIdx'])
            draw_text(img_bgr1, word_tag_str, org=get_box_center(word_box), color=(0, 0, 255))

        # 绘制图像2
        color_cols_list = generate_colors(len(large_col_boxes), seed=72)
        color_prbs_list = generate_colors(len(problem_boxes))

        img_bgr2 = copy.copy(img_bgr)
        for idx, xyxy in enumerate(large_col_boxes):
            img_copy = copy.copy(img_bgr2)
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            img_bgr2 = cv2.rectangle(img_bgr2, c1, c2, color_cols_list[idx], thickness=5, lineType=cv2.LINE_AA)
            # cv2.addWeighted(img_bgr2, 0.3, img_copy, 0.7, 0, img_bgr2)  # 添加色块

        for idx, xyxy in enumerate(problem_boxes):
            img_copy = copy.copy(img_bgr2)
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            img_bgr2 = cv2.rectangle(img_bgr2, c1, c2, color_prbs_list[idx], thickness=-1, lineType=cv2.LINE_AA)
            cv2.addWeighted(img_bgr2, 0.3, img_copy, 0.7, 0, img_bgr2)  # 添加色块
            draw_text(img_bgr2, str(idx), org=get_box_center(xyxy), color=(0, 0, 255))

        # 绘制图像3
        if blocks_info:
            img_bgr3 = copy.copy(img_bgr)
            img_bgr3 = self.draw_blocks_info(img_bgr3, blocks_info)
        else:
            img_bgr3 = copy.copy(img_bgr)

        # 绘制图像4
        img_bgr4 = copy.copy(img_bgr)
        img_bgr4 = self.draw_word_info(img_bgr4, rank_word_info)

        return img_bgr1, img_bgr2, img_bgr3, img_bgr4

    @staticmethod
    def split_boxes_by_dist(boxes, word_strings, w, thr=0.45):
        """
        避免相距较远的块组成行
        """
        idx_split, is_split = -1, False
        for i in range(len(boxes)-1):
            box1 = boxes[i]
            box2 = boxes[i+1]
            diff_w = box2[0] - box1[2]
            r_w = safe_div(diff_w, w)
            print('[Info] r_w: {}'.format(r_w))
            if r_w > thr:
                is_split = True
                idx_split = i
                break

        if is_split:
            boxes_head = boxes[:idx_split+1]
            boxes_end = boxes[idx_split+1:]
            boxes = [boxes_head, boxes_end]

            strings_head = word_strings[:idx_split+1]
            strings_end = word_strings[idx_split+1:]
            word_strings = [strings_head, strings_end]

        return is_split, boxes, word_strings

    @staticmethod
    def generate_boxes_and_confs(col_boxes, problem_boxes, box_confs):
        """
        生产输出的数据，框和置信度
        """
        res_list = []
        for large_col_box in col_boxes:
            tmp_boxes = []
            tmp_confs = []
            for box, conf in zip(problem_boxes, box_confs):
                x_iou = min_iou(large_col_box, box)
                if x_iou < 0.5:
                    continue
                tmp_boxes.append(box)
                tmp_confs.append(conf)

            sorted_boxes, sorted_idxes, _ = sorted_boxes_by_col(tmp_boxes)
            sorted_confs = filter_list_by_idxes(tmp_confs, sorted_idxes)

            tmp_res_list = []
            for x_boxes, x_confs in zip(sorted_boxes, sorted_confs):
                for box, conf in zip(x_boxes, x_confs):
                    if conf == -1:
                        conf = 1.0
                        b_type = "text"
                    else:
                        conf = conf
                        b_type = "vision"
                    data_dict = {
                        "box": box,
                        "conf": conf,
                        "type": b_type
                    }
                    tmp_res_list.append(data_dict)
            res_list.append(tmp_res_list)
        return res_list

    @staticmethod
    def rotate_bound(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        h, w = image.shape[:2]
        cX, cY = w // 2, h // 2

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int(h * sin + w * cos)
        nH = int(h * cos + w * sin)

        # adjust the rotation matrix to take into account translation
        M[0, 2] += nW / 2 - cX
        M[1, 2] += nH / 2 - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH)), M

    @staticmethod
    def warp_point(x, y, M):
        Q = np.dot(M, np.array(((x,), (y,), (1,))))
        return int(Q[0][0]), int(Q[1][0])

    @staticmethod
    def warp_data(words_info, M):
        new_boxes = []
        for word_info in words_info:
            pos = word_info['pos']
            word_rec = ImgDetector.parse_pos(pos)

            # word_box = rec2box(word_rec)
            # x1, y1, x2, y2 = word_box
            # x1, y1 = ImgDetector.warp_point(x1, y1, M)
            # x2, y2 = ImgDetector.warp_point(x2, y2, M)
            # new_box = [x1, y1, x2, y2]
            new_rec = []
            for pnt in word_rec:
                x, y = pnt
                x, y = ImgDetector.warp_point(x, y, M)
                new_rec.append([x, y])
            new_box = rec2box(new_rec)
            new_boxes.append(new_box)
            word_info['rotated_box'] = new_box
        return new_boxes

    @staticmethod
    def rotate_img(img_bgr, words_info):
        """
        根据words_info的子块旋转
        """
        sum_angle = 0
        n_words = len(words_info)
        for word_info in words_info:
            sum_angle += word_info['angle']

        avg_angle = sum_angle // n_words

        img_rotated, M = ImgDetector.rotate_bound(img_bgr, avg_angle)
        # show_img_bgr(img_rotated)  # 测试

        rotated_boxes = ImgDetector.warp_data(words_info, M)
        # draw_box_list(img_rotated, rotated_boxes, is_show=True)  # 测试

        return img_rotated, rotated_boxes

    @staticmethod
    def sorted_words_info(img_bgr, rotated_boxes, words_info):
        sorted_boxes, sorted_idxes, num_row = sorted_boxes_by_row(rotated_boxes, img_bgr=img_bgr)
        res_rotated_boxes = []
        res_words_info = []
        for sub_idx_list in sorted_idxes:
            for idx in sub_idx_list:
                res_rotated_boxes.append(rotated_boxes[idx])
                res_words_info.append(words_info[idx])
        return res_rotated_boxes, res_words_info

    @staticmethod
    def module_rotated_input(img_bgr, words_info):
        """
        旋转wordInfo
        """
        img_rotated, rotated_boxes = ImgDetector.rotate_img(img_bgr, words_info)  # 旋转角度
        _, words_info_rotated = ImgDetector.sorted_words_info(img_rotated, rotated_boxes, words_info)
        return img_rotated, words_info_rotated

    @staticmethod
    def module_split_banners(problem_boxes, boxes_confs, img_w):
        """
        拆分Banner
        """
        problem_boxes, new_idxes = ImgDetector.filer_boxes_by_size(problem_boxes)   # 合并重叠的检测框
        boxes_confs = filter_list_by_idxes(boxes_confs, new_idxes)  # 同步过滤置信度

        # 增加上下分栏的适配
        head_banner_boxes, tail_banner_boxes, sub_cv_problem_boxes, head_confs, tail_confs, sub_cv_confs = \
            ImgDetector.filter_banner_box(problem_boxes, boxes_confs, img_w)

        # 重新排序问题框，从上到下
        col_problem_boxes, col_sorted_idxes, num_col = sorted_boxes_by_col(sub_cv_problem_boxes)

        # 合并首栏和尾栏的框
        problem_boxes = []  # 去掉分栏信息
        problem_boxes += head_banner_boxes
        for pcb in col_problem_boxes:
            problem_boxes += pcb
        problem_boxes += tail_banner_boxes

        # 合并首栏和尾栏的置信度
        boxes_confs = []  # 去掉分栏信息
        boxes_confs += head_confs
        for line_idxes in col_sorted_idxes:
            boxes_confs += [sub_cv_confs[i] for i in line_idxes]
        boxes_confs += tail_confs
        return problem_boxes, boxes_confs, col_problem_boxes, head_banner_boxes, tail_banner_boxes

    @staticmethod
    def module_analyze_columns(col_problem_boxes, head_banner_boxes, tail_banner_boxes, img_h, img_w):
        """
        分栏逻辑
        """
        large_col_boxes = ImgDetector.analyze_layout(col_problem_boxes, head_banner_boxes, tail_banner_boxes, img_h, img_w)
        return large_col_boxes

    @staticmethod
    def module_rank_wordinfo(large_col_boxes, problem_boxes, boxes_confs, words_info_rotated, img_bgr=None):
        """
        重排序wordinfo
        """
        rank_words_info, column_num, block_num_list, blocks_info, ranked_content = \
            ImgDetector.add_tag_to_wordinfo_v2(
                large_col_boxes, problem_boxes, boxes_confs, words_info_rotated, img_bgr=img_bgr)
        return rank_words_info, column_num, block_num_list, blocks_info, ranked_content

    @staticmethod
    def module_add_info_2_data(data_dict, rank_words_info, ranked_content, column_num, block_num_list, blocks_info):
        """
        添加新的数据内容
        """
        rank_words_info = copy.copy(rank_words_info)
        for r_word_info in rank_words_info:
            if "rotated_box" in r_word_info.keys():
                r_word_info.pop("rotated_box")

        data_dict["content"] = ranked_content
        data_dict["wordsInfo"] = rank_words_info
        data_dict["columnNum"] = column_num
        data_dict["blockNumList"] = block_num_list
        data_dict["blocksInfo"] = blocks_info

        return data_dict

    @staticmethod
    def check_words_info(img_bgr, words_info, is_rank=False):
        """
        检查排序
        """
        word_boxes = []
        for word_info in words_info:
            if not is_rank:
                pos = word_info['pos']
                word_rec = ImgDetector.parse_pos(pos)
                word_box = rec2box(word_rec)
                word_boxes.append(word_box)
            else:
                word_box = word_info['rotated_box']
                word_boxes.append(word_box)

        draw_box_list(img_bgr, word_boxes, is_show=True, save_name="tmp-{}.jpg".format(get_current_time_str()))
        print('[Info] 验证')

    def process(self, img_url, is_sentence=True):
        """
        核心处理逻辑
        """
        # 第0步, 检测题目框
        is_ok, img_bgr = download_url_img(img_url)
        img_h, img_w, _ = img_bgr.shape

        # 第0步，获取OCR的文本框
        words_info, word_rec_list, word_str_list, idx_list, data_dict \
            = self.get_ocr_service_info(img_url)
        word_boxes = []
        for rec in word_rec_list:
            word_boxes.append(rec2box(rec))
        if not word_boxes:
            print('[Info] 文字框为空')
            return img_bgr, img_bgr, img_bgr

        # -----------  第0步 旋转重排序wordInfo  ---------------- #
        # self.check_words_info(img_bgr, words_info)
        img_rotated, words_info_rotated = ImgDetector.module_rotated_input(img_bgr, words_info)
        # self.check_words_info(img_rotated, words_info_rotated, is_rank=True)
        # -----------  第0步 旋转重排序wordInfo  ---------------- #

        # -----------  第1步 检测题目框  ---------------- #
        cv_problem_boxes, cv_confs = self.detect_problems(img_rotated)
        if not cv_problem_boxes:
            print('[Info] 题目框为空')
            return img_bgr, img_bgr, img_bgr
        # -----------  第1步 检测题目框  ---------------- #

        # ----------- 第2步 增加处理上下分栏逻辑 ---------------- #
        problem_boxes, boxes_confs, col_problem_boxes, head_banner_boxes, tail_banner_boxes = \
            ImgDetector.module_split_banners(problem_boxes=cv_problem_boxes, boxes_confs=cv_confs, img_w=img_w)
        # ----------- 第2步 增加处理上下分栏逻辑 ---------------- #

        # ----------- 第3步 使用banner，中间题目框，合并成栏 ---------------- #
        large_col_boxes = ImgDetector.module_analyze_columns(
            col_problem_boxes, head_banner_boxes, tail_banner_boxes, img_h, img_w)
        # draw_box_list(img_bgr, large_col_boxes, is_show=True)
        # ----------- 第3步 使用banner，中间题目框，合并成栏 ---------------- #

        # ----------- 第4步 添加信息和重排, 生成块信息  ---------------- #
        rank_words_info, column_num, block_num_list, blocks_info, ranked_content = \
            ImgDetector.module_rank_wordinfo(large_col_boxes, problem_boxes, boxes_confs, words_info_rotated, img_rotated)
        print('[Info] 文字框数: {}, 新文字框数: {}'.format(len(words_info_rotated), len(rank_words_info)))
        # ----------- 第4步 添加信息和重排, 生成块信息  ---------------- #

        # ----------- 第5步 合并信息 ---------------- #
        # data_dict = ImgDetector.module_add_info_2_data(
        #     data_dict, rank_words_info, ranked_content, column_num, block_num_list, blocks_info)
        # ----------- 第5步 合并信息 ---------------- #

        img_out1, img_out2, img_out3, img_out4 = \
            self.draw_all_boxes(img_rotated, rank_words_info, words_info_rotated,
                                large_col_boxes, problem_boxes, blocks_info)

        show_img_bgr(img_out1, save_name="test1.jpg".format(get_current_time_str()))
        show_img_bgr(img_out2, save_name="test2.jpg".format(get_current_time_str()))
        show_img_bgr(img_out3, save_name="test3.jpg".format(get_current_time_str()))
        show_img_bgr(img_out4, save_name="test4.jpg".format(get_current_time_str()))

        return img_out1, img_out2, img_out3


def evaluate_urls():
    urls_file = os.path.join(DATA_DIR, 'test_400_res.right.e0.csv')
    data_lines = read_file(urls_file)

    out_dir = os.path.join(DATA_DIR, 'cases-400')
    mkdir_if_not_exist(out_dir)
    _, img_names = traverse_dir_files(out_dir)

    ido = ImgDetector()
    for idx, data_line in enumerate(data_lines):
        if idx == 0:
            continue
        items = data_line.split(',')
        url = items[0]
        img_name = url.split('/')[-1]
        if img_name in img_names:
            print('[Info] 已经处理!')
            continue
        url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/tmp/test_400_out/{}".format(img_name)
        print('[Info] url: {}'.format(url))
        img_out1, img_out2, img_out3 = ido.process(url, is_sentence=False)
        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, img_out2)
        print('[Info] 处理完成: {}'.format(out_path))


def process_url(out_dir, url, img_name, idx):
    print('[Info] idx: {}, url: {}'.format(idx, url))
    data_dict = get_problem_segmentation_vpf_service(url)
    try:
        oss_url_out2 = data_dict['data']['oss_url_out2']
    except Exception as e:
        print('[Info] error: {}'.format(url))
        oss_url_out2 = url
    is_ok, img_bgr = download_url_img(oss_url_out2)
    out_path = os.path.join(out_dir, img_name)
    try:
        cv2.imwrite(out_path, img_bgr)
    except Exception as e:
        pass
    print('[Info] 处理完成: {}'.format(url))


def evaluate_urls_v2():
    file_path = os.path.join(DATA_DIR, 'url_0108_高航.txt')
    out_dir = os.path.join(DATA_DIR, 'url_0108_高航')
    _, img_names = traverse_dir_files(out_dir)

    mkdir_if_not_exist(out_dir)
    data_lines = read_file(file_path)

    # pool = Pool(processes=2)
    for idx, data_line in enumerate(data_lines):
        url = data_line.split("?")[0]
        img_name = url.split('/')[-1]
        if img_name in img_names:
            print('[Info] 已处理: {}'.format(img_name))
            continue
        process_url(out_dir, url, img_name, idx)

    # pool.close()
    # pool.join()

    print('[Info] 已处理完成!')


def evaluate_case():
    from root_dir import DATA_DIR
    # img_path = os.path.join(DATA_DIR, 'labeled_mini', '9f4c773acbfb2f58cf4e0d7eb2099fa6.jpg')
    # img_path = os.path.join(DATA_DIR, 'test', 'fcf3445e2f6b1cfce533ba272818a24f.jpg')
    # img_bgr = cv2.imread(img_path)
    # show_img_bgr(img_bgr)
    # img_url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/tmp/test1.jpg"
    # img_url = "https://img.alicdn.com/imgextra/i3/6000000000052/O1CN01EvpIvD1CFrEQZjHyI_!!6000000000052-0-quark.jpg"
    # img_url = "https://img.alicdn.com/imgextra/i1/6000000005190/O1CN013t3IMJ1oD4DVlTj7b_!!6000000005190-0-quark.jpg"
    # img_url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/tmp/test_400_out/O1CN01KseCRE1FsjhJD7ivM_!!6000000000543-0-quark.jpg"

    # img_url = "http://vpf-test.oss-cn-hangzhou.aliyuncs.com/tif/images/a62ed370-54aa-11eb-94be-98039b9cea08.jpg"

    # 下Banner通栏, 略有Bug
    # img_url = "http://vpf-test.oss-cn-hangzhou.aliyuncs.com/tif/images/6fae1306-54aa-11eb-9302-98039b9cea08.jpg"
    # 上Banner通栏
    # img_url = "http://vpf-test.oss-cn-hangzhou.aliyuncs.com/tif/images/c60030cc-54aa-11eb-aeb7-98039b9cea08.jpg"

    # 多块合并Case
    # img_url = "https://img.alicdn.com/imgextra/i3/6000000002579/O1CN01NheOr01UvE6MWKHvE_!!6000000002579-0-quark.jpg"

    # 文本块异常Case
    # img_url = "http://vpf-test.oss-cn-hangzhou.aliyuncs.com/tif/images/75642088-54aa-11eb-8507-98039b9cea08.jpg"

    # img_url = "https://img.alicdn.com/imgextra/i3/6000000007622/O1CN01h99TKk26AvSayz5R4_!!6000000007622-0-quark.jpg"

    # Bug, 额外小块
    # img_url = "https://img.alicdn.com/imgextra/i2/6000000001979/O1CN01lZhWa71QUQSgyIoOT_!!6000000001979-0-quark.jpg"

    # 左右杂质
    # img_url = "https://img.alicdn.com/imgextra/i4/6000000003919/O1CN01xQaU641eowpHvaVoW_!!6000000003919-0-quark.jpg"

    # 倾斜整页
    # img_url = "https://img.alicdn.com/imgextra/i4/6000000004532/O1CN01WLsqmX1jLhcIjSf8F_!!6000000004532-0-quark.jpg"

    # Bug
    # img_url = "https://img.alicdn.com/imgextra/i1/6000000003432/O1CN01PAXDXL1bDtwaz45cM_!!6000000003432-0-quark.jpg"

    # 文字框空
    # img_url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/tmp/test_400_out/O1CN01lQPBvt1t2bFDWslTp_!!6000000005844-0-quark.jpg"

    # 小块文字
    # img_url = "https://img.alicdn.com/imgextra/i2/6000000003335/O1CN01aNTJ931aVTVuytHXc_!!6000000003335-0-quark.jpg"

    # 缺少块
    # img_url = "http://vpf-test.oss-cn-hangzhou.aliyuncs.com/qcut_rotate_skew_image/82a6c550-5a2e-11eb-a57e-98039b7cdc56.jpg"

    # 弯曲Case
    # img_url = "https://img.alicdn.com/imgextra/i4/6000000006439/O1CN017PxjwH1xR6vA1Ismg_!!6000000006439-0-quark.jpg"
    # img_url = "https://img.alicdn.com/imgextra/i1/6000000004084/O1CN01eCFl7L1g2WBzoGfgT_!!6000000004084-0-quark.jpg"
    # img_url = "https://img.alicdn.com/imgextra/i4/6000000003145/O1CN01kkiAyB1Z6SH4xDC6j_!!6000000003145-0-quark.jpg"
    img_url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/tmp/test_400_out/O1CN01g2qgo51NkVVboaXj1_!!6000000001608-0-quark.jpg"

    ido = ImgDetector()
    # ido.detect_with_draw(img_bgr)
    img_out = ido.process(img_url, is_sentence=True)


def main():
    # evaluate_urls()
    # evaluate_urls_v2()
    evaluate_case()


if __name__ == '__main__':
    main()
