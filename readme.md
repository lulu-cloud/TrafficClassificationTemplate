# 1 流量预处理
安装test_date文件夹仿照就可以了
~~~
原始pcap流量路径
--- traffic_data
     --- 类别1的标签名
        -- 类别1的第一个pcap
        -- 类别1的第二个pcap
        -- ..
     --- 类别2的标签名
        -- 类别2的第一个pcap
        -- 类别2的第二个pcap
        -- ..
     ...
     ...
     --- 类别n的标签名
~~~
- 运行`entry/1_preprocess.py`即可

# 2 训练

- 运行`entry/2_train_test_script.py`即可
- ``config.yaml``是训练版本的超参数配置文件，可修改改变各个loss的占比
