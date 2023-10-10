
import torch
import numpy as np

from traffic_log.setLog import logger


def get_tensor_data(pcap_file, seq_file, label_file):
    # 载入 npy 数据
    pcap_data = np.load(pcap_file)  # 获得 pcap 文件
    seq_data = np.load(seq_file)
    label_data = np.load(label_file)  # 获得 label 数据

    # 将 npy 数据转换为 tensor 数据
    pcap_data = torch.from_numpy(pcap_data.reshape(-1,1,pcap_data.shape[1])).float()
    # (batch_size, seq_len, input_size)
    seq_data = torch.from_numpy(seq_data.reshape(-1,seq_data.shape[1],1)).float()
    label_data = torch.from_numpy(label_data).long()
    logger.info('pcap 文件大小, {}; seq文件大小:{}; label 文件大小: {}'.format(pcap_data.shape, seq_data.shape,
                                                                               label_data.shape))

    return pcap_data, seq_data, label_data