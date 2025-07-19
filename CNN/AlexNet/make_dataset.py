import os
import torch
from torch.utils.data import random_split
from shutil import copy, rmtree
import random
import tarfile
import urllib.request


def download_and_extract(url, dst_dir):
    tgz_path = os.path.join(dst_dir, "flower_photos.tgz")
    if not os.path.exists(tgz_path):
        print("Downloading flower_photos.tgz ...")
        os.makedirs(dst_dir, exist_ok=True)
        urllib.request.urlretrieve(url, tgz_path)
    if not os.path.exists(os.path.join(dst_dir, "flower_photos")):
        print("Extracting ...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=dst_dir)


def ensure_dirs(root, classes, split):
    split_dir = os.path.join(root, split)
    if os.path.exists(split_dir):
        rmtree(split_dir)
    for cls in classes:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)


def split_dataset(src_root, dst_root, classes, train_ratio=0.8, seed=42):
    random.seed(seed)
    for cls in classes:
        src_cls_dir = os.path.join(src_root, cls)
        imgs = [f for f in os.listdir(src_cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(imgs)

        n_train = int(len(imgs) * train_ratio)
        train_imgs = imgs[:n_train]
        test_imgs = imgs[n_train:]

        
        for img in train_imgs:
            copy(os.path.join(src_cls_dir, img),
                 os.path.join(dst_root, "train", cls, img))
       
        for img in test_imgs:
            copy(os.path.join(src_cls_dir, img),
                 os.path.join(dst_root, "test", cls, img))


def main():
    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = os.path.join(os.getcwd(), "data/flower")
    raw_dir = os.path.join(data_dir, "flower_photos")
    processed_dir = os.path.join(data_dir, "flower_split")  


    try:
        download_and_extract(url, data_dir)
    except Exception as e:
        print("Download failed:", e)
        print("请手动下载并解压 flower_photos.tgz 到 data/flower_photos 目录，然后继续。")
        if not os.path.exists(raw_dir):
            return


    classes = [d for d in os.listdir(raw_dir)
               if os.path.isdir(os.path.join(raw_dir, d))]


    ensure_dirs(processed_dir, classes, "train")
    ensure_dirs(processed_dir, classes, "test")


    split_dataset(raw_dir, processed_dir, classes, train_ratio=0.8, seed=42)
    print("Dataset split done! Check:", processed_dir)


if __name__ == "__main__":
    main()
