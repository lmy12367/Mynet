import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from net import TinySSD


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BananaDataset(Dataset):
    def __init__(self, data_dir, is_train=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        dataset_type = 'bananas_train' if is_train else 'bananas_val'
        self.image_dir = os.path.join(data_dir, dataset_type, 'images')
        label_file = os.path.join(data_dir, dataset_type, 'label.csv')
        
        self.labels = pd.read_csv(label_file).set_index('img_name')
        self.all_boxes = self.labels[['xmin', 'ymin', 'xmax', 'ymax']].values
        self.all_wh = self.all_boxes[:, 2:] - self.all_boxes[:, :2]
        
        print(f"{dataset_type} - 平均宽高: {np.mean(self.all_wh, axis=0)}")
        print(f"{dataset_type} - 宽高范围: [min={np.min(self.all_wh, axis=0)}, max={np.max(self.all_wh, axis=0)}]")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.index[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        
        width, height = img.size
        bbox = self.labels.iloc[idx][['xmin', 'ymin', 'xmax', 'ymax']].values
        bbox = torch.tensor(bbox, dtype=torch.float32) / torch.tensor([width, height, width, height])
        
        
        bbox = torch.clamp(bbox, 0.0, 1.0)
        label = torch.tensor([1], dtype=torch.long)
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.cat([label.float(), bbox]).unsqueeze(0)


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    c_anc = (anchors[:, 2:] + anchors[:, :2]) / 2
    c_assigned = (assigned_bb[:, 2:] + assigned_bb[:, :2]) / 2
    
    wh_anc = anchors[:, 2:] - anchors[:, :2] + eps
    wh_assigned = assigned_bb[:, 2:] - assigned_bb[:, :2] + eps
    
    
    offset_xy = 10 * (c_assigned - c_anc) / wh_anc
    offset_wh = 5 * torch.log(torch.clamp(wh_assigned / wh_anc, 1e-4, 1e4))
    
    return torch.cat([offset_xy, offset_wh], dim=1)

def multibox_target(anchors, labels):
    batch_size, num_anchors = anchors.shape[:2]
    device = anchors.device
    
    batch_offset, batch_mask, batch_class_labels = [], [], []
    
    for i in range(batch_size):
       
        sample_anchors = anchors[i].clone() 
        gt_boxes = labels[i][:, 1:5]  
        gt_labels = labels[i][:, 0].long() 
        
        
        sample_anchors = torch.clamp(sample_anchors, 0.0, 1.0)
        gt_boxes = torch.clamp(gt_boxes, 0.0, 1.0)
        
        
        inter_xmin = torch.max(sample_anchors[:, 0], gt_boxes[0, 0])
        inter_ymin = torch.max(sample_anchors[:, 1], gt_boxes[0, 1])
        inter_xmax = torch.min(sample_anchors[:, 2], gt_boxes[0, 2])
        inter_ymax = torch.min(sample_anchors[:, 3], gt_boxes[0, 3])
        
        inter_area = torch.clamp(inter_xmax - inter_xmin, 0) * torch.clamp(inter_ymax - inter_ymin, 0)
        anchors_area = (sample_anchors[:, 2] - sample_anchors[:, 0]) * (sample_anchors[:, 3] - sample_anchors[:, 1])
        gt_area = (gt_boxes[0, 2] - gt_boxes[0, 0]) * (gt_boxes[0, 3] - gt_boxes[0, 1])
        
        iou = inter_area / (anchors_area + gt_area - inter_area + 1e-6)
        
        
        pos_mask = iou >= 0.5
        assigned_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_boxes = torch.zeros_like(sample_anchors)
        
        assigned_labels[pos_mask] = gt_labels[0]
        assigned_boxes[pos_mask] = gt_boxes[0]
        
        
        offsets = offset_boxes(sample_anchors, assigned_boxes) * pos_mask.unsqueeze(1).float()
        
        batch_offset.append(offsets)
        batch_mask.append(pos_mask)
        batch_class_labels.append(assigned_labels)
    
    return (torch.stack(batch_offset),
            torch.stack(batch_mask),
            torch.stack(batch_class_labels))


def cls_eval(cls_preds, cls_labels):
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    bbox_preds = bbox_preds.reshape(bbox_labels.shape)
    bbox_masks = bbox_masks.unsqueeze(-1)
    return float(torch.abs((bbox_labels - bbox_preds) * bbox_masks).sum())


class MultiBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        self.bbox_loss = nn.SmoothL1Loss(reduction='none')  
    
    def forward(self, cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        
        cls = self.cls_loss(
            cls_preds.reshape(-1, cls_preds.shape[-1]),
            cls_labels.reshape(-1)
        ).reshape(cls_labels.shape[0], -1).mean(dim=1)
        
       
        bbox = self.bbox_loss(
            bbox_preds.reshape(bbox_labels.shape) * bbox_masks.unsqueeze(-1),
            bbox_labels * bbox_masks.unsqueeze(-1)
        ).mean(dim=[1, 2])
        
        
        if torch.isnan(cls).any() or torch.isnan(bbox).any():
            print("警告: 检测到NaN值!")
            return torch.tensor(0.0, requires_grad=True)
        
        return (cls + bbox).mean()


def train(net, train_loader, val_loader, num_epochs=20, lr=1e-3):
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    criterion = MultiBoxLoss()
    
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    
    for epoch in range(num_epochs):
        net.train()
        train_metrics = [0.0] * 4
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            
            anchors, cls_preds, bbox_preds = net(images)
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, targets)
            
            
            loss = criterion(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            
            
            train_metrics[0] += cls_eval(cls_preds, cls_labels)
            train_metrics[1] += cls_labels.numel()
            train_metrics[2] += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            train_metrics[3] += bbox_labels.numel()
        
        
        net.eval()
        val_metrics = [0.0] * 4
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                
                anchors, cls_preds, bbox_preds = net(images)
                bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, targets)
                
                loss = criterion(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
                val_loss += loss.item()
                
                val_metrics[0] += cls_eval(cls_preds, cls_labels)
                val_metrics[1] += cls_labels.numel()
                val_metrics[2] += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
                val_metrics[3] += bbox_labels.numel()
        
        
        train_loss = loss.item()
        train_cls_err = 1 - train_metrics[0] / train_metrics[1]
        train_bbox_mae = train_metrics[2] / train_metrics[3]
        
        val_cls_err = 1 - val_metrics[0] / val_metrics[1]
        val_bbox_mae = val_metrics[2] / val_metrics[3]
        
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, ClsErr: {train_cls_err:.4f}, BboxMAE: {train_bbox_mae:.4f}")
        print(f"Val - Loss: {val_loss/len(val_loader):.4f}, ClsErr: {val_cls_err:.4f}, BboxMAE: {val_bbox_mae:.4f}")
        
        
        scheduler.step(val_loss)
    
    torch.save(net.state_dict(), 'ssd_model.pth')
    print("训练完成，模型已保存")


def get_transform(train=True):
    transform = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if train:
        transform.insert(1, transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transform)


if __name__ == "__main__":
    
    net = TinySSD(num_classes=1)
    
    
    data_dir = 'D:\\code\\document\\data\\banana-detection'  
    train_dataset = BananaDataset(data_dir, is_train=True, transform=get_transform(True))
    val_dataset = BananaDataset(data_dir, is_train=False, transform=get_transform(False))
    
    
    train_loader = DataLoader(
        Subset(train_dataset, range(min(512, len(train_dataset)))),
        batch_size=64,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        Subset(val_dataset, range(min(100, len(val_dataset)))),
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    
    print(f"训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")
    train(net, train_loader, val_loader, num_epochs=5)