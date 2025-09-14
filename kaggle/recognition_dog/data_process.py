# data_process.py
import os
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

CSV_PATH = './data/labels.csv'     # 标签文件
IMG_DIR = './data/train'           # 训练图文件夹
JSON_PATH = './class_to_num.json'  # 输出映射

def build_class_mapping():
    """生成 类别名 -> 数字 的映射并保存"""
    df = pd.read_csv(CSV_PATH)
    class_names = sorted(df['breed'].unique())   # 按字母排序
    class_to_num = {cls: idx for idx, cls in enumerate(class_names)}

    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_to_num, f, indent=2)
    print(f"✅ 映射已保存 → {JSON_PATH}")
    print(f"📊 共 {len(class_names)} 类")
    return class_to_num

def show_first_image():
    """可视化第一张图"""
    df = pd.read_csv(CSV_PATH)
    img_id = df.iloc[0]['id']          # 第 0 行
    breed = df.iloc[0]['breed']
    img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")

    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)
    plt.title(breed)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    build_class_mapping()
    show_first_image()