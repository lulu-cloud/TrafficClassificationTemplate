
import os
import yaml
from easydict import EasyDict

def setup_config():
    """获取配置信息
    """
    with open(r"../entry/traffic_classification.yaml", encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg