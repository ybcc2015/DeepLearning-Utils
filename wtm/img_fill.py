import numpy as np
import cv2
import json


class ImgFill(object):
    def __init__(self, json_file):
        self.box = self.get_box(json_file)
    
    @staticmethod
    def get_box(json_file):
        """
        获取json文件中的box_b的坐标
        Args:
            json_file (str): json文件路径
        Returns:
            [list]: box_b的坐标，格式为[left, top, right, bottom]
        """
        res = []
        
        with open("./boxes.json") as f:
            json_data = json.load(f)
        
        for box in json_data["boxes"]:
            if box["name"] == "box_b":
                print(box["rectangle"])
                res = box["rectangle"]["left_top"]
                res.extend(box["rectangle"]["right_bottom"])
        
        return res
    
    def is_box_valid(self, img):
        """
        判断box_b指定的区域是否超出img的边界
        Args:
            img (numpy array): 目标图像
        Returns:
            [bool]: ture or false
        """
        left, top = self.box[:2]
        h = self.box[3] - self.box[1]
        w = self.box[2] - self.box[0]
        h_img, w_img = img.shape[:2]
        
        cond1 = left >= 0 and (left + w) <= w_img
        cond2 = top >= 0 and (top + h) <= h_img
        return cond1 and cond2

    def fill(self, dst_img, src_img, mode="stretch"):
        """
        图像填充函数
        Args:
            dst_img (numpy array): 目标图像
            src_img (numpy array): 源图像 (待填充的图像)
            mode (str): 填充模式, "stretch"指拉伸填充, "keep"指保持比例填充
        Returns:
            [numpy array]: 填充后的图像
        """
        ok = self.is_box_valid(dst_img)
        if not ok:
            return
        
        # 得到填充区域的左上角顶点, 以及宽和高
        left, top = self.box[:2]
        h = self.box[3] - self.box[1]
        w = self.box[2] - self.box[0]
        
        assert mode in ["stretch", "keep"], "当前仅支持'stretch'和'keep'填充模式!"
        
        if mode == "stretch":
            src_img = cv2.resize(src_img, (w, h))
            dst_img[top: top + h, left: left + w] = src_img

        if mode == "keep":
            # 基于源图的长边得到缩放比例
            h_img, w_img = src_img.shape[:2]
            ratio = h / h_img if h_img >= w_img else w / w_img
            
            # 源图等比例缩放
            h_new = int(round(h_img * ratio))
            w_new = int(round(w_img * ratio))
            src_img = cv2.resize(src_img, (w_new, h_new))
            
            # 如果缩放后的高小于填充区域的高, 则沿y轴方向进行pad
            if h_new < h:
                pad = h - h_new
                pad_size = (pad // 2, pad - pad // 2)
                np.pad(src_img, (pad_size, (0, 0)))
                h_new = h
                
            # 如果缩放后的宽小于填充区域的高宽, 则沿x轴方向进行pad
            if w_new < w:
                pad = w - w_new
                pad_size = (pad // 2, pad - pad // 2)
                np.pad(src_img, ((0, 0), pad_size))
                w_new = w

            dst_img[top: top + h_new, left: left + w_new] = src_img

        return dst_img
