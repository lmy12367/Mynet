# Kaggle-House-Prices-Advanced-Regression

本项目用于参加 Kaggle **House Prices – Advanced Regression Techniques** 竞赛。  
核心逻辑在 `main.py` 中完成：  

1. 通过 `data_read.py` 读取并预处理数据；  
2. 用 `MyNet.py` 中定义的模型 `MyNet`；  
3. 调用 `train.py` 完成训练与预测；  
4. 最终生成可供提交的 `submission.csv`。

---

## 目录结构

├─ data 

│  ├─ data_description.txt

│  ├─ sample_submission.csv 

│  ├─ test.csv 

│  └─ train.csv 

├─ img 

│  ├─ training_loss.png 

│  └─ training_loss_last.png 

├─ model 

│  ├─ model.pth 

│  └─ model_last.pth 

├─ data_read.py      # 数据读取与预处理 

├─ MyNet.py          # 网络结构定义 

├─ train.py          # 训练、验证、预测 

├─ main.py           # 一键运行入口（组合上述三个脚本） 

└─ submission.csv    # 最终提交文件