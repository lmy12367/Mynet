# 项目概述

这是一个基于 RNN（循环神经网络）的模型项目，包含数据预处理、网络构建、模型训练和预测等相关功能模块。

## 文件结构

-   data_preprocessing.py ：负责数据的预处理工作，包括数据加载、清洗、归一化、序列化等操作，为模型训练准备合适的数据格式。
-   net.py ：定义了 RNN 网络的结构，搭建了神经网络的各层架构，包括输入层、隐藏层和输出层等，确定了网络的连接方式和参数。
-   train.py ：包含模型训练的主程序，设置了训练过程中的超参数（如学习率、迭代次数等），定义了损失函数和优化器，并通过循环迭代对模型进行训练，同时会保存训练好的模型参数。
-   predict.py ：利用训练好的 RNN 模型进行预测，加载测试数据，调用模型进行推理，输出预测结果