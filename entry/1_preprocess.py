# -*- coding: utf-8 -*-
# @Time : 2023/5/17 15:56
# @Author :
# @Email :
# @File : 1_preprocess.py
"""
@Description: 预处理函数
"""
import logging

from utils.setConfig import setup_config
from preprocess.pcap2session import getPcapIPLength
from preprocess.split_all_data import split_data

def main():
    cfg = setup_config() # 获取 config 文件
    logging.info("开始预处理")
    # pay, label = getPcapPayLoad(cfg.preprocess.traffic_path,cfg.preprocess.packet_num,cfg.preprocess.byte_num)
    pay, seq, label = getPcapIPLength(cfg.preprocess.traffic_path,cfg.preprocess.threshold, cfg.preprocess.ipLength, cfg.preprocess.packet_num, cfg.preprocess.byte_num)
    split_data(pay,seq,label,cfg.preprocess.train_size,cfg.preprocess.npy_path)

if __name__=="__main__":
    main()