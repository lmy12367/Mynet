import torch
import torch.nn as nn

# --- 工具函数 ---
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)

# --- 网络块定义 ---
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

# --- 锚框生成 ---
def multibox_prior(data, sizes, ratios):
    in_height, in_width = data.shape[-2:]  # 获取特征图高宽
    device = data.device
    num_sizes, num_ratios = len(sizes), len(ratios)
    num_anchors = num_sizes + num_ratios - 1  # 每个位置生成的锚框数
    
    # 计算中心点坐标（归一化到0-1）
    center_x = (torch.arange(in_width, device=device) + 0.5) / in_width
    center_y = (torch.arange(in_height, device=device) + 0.5) / in_height
    center_x, center_y = torch.meshgrid(center_x, center_y, indexing='ij')
    
    # 生成宽高组合
    s = torch.tensor(sizes, device=device)
    r = torch.tensor(ratios, device=device)
    w = torch.cat([s * torch.sqrt(r[0]), s[0] * torch.sqrt(r[1:])])
    h = torch.cat([s / torch.sqrt(r[0]), s[0] / torch.sqrt(r[1:])])
    
    # 生成锚框坐标（左上+右下）
    anchor_manipulations = torch.stack([-w/2, -h/2, w/2, h/2], dim=1)
    anchor_centers = torch.stack([center_x.flatten(), center_y.flatten(),
                                center_x.flatten(), center_y.flatten()], dim=1)
    
    anchors = anchor_centers.unsqueeze(1) + anchor_manipulations.unsqueeze(0)
    anchors = anchors.view(-1, 4).unsqueeze(0)
    
    # 新增：严格限制坐标范围
    anchors = torch.clamp(anchors, min=0.0, max=1.0)
    return anchors.repeat(data.shape[0], 1, 1)

# --- 前向传播块 ---
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


class TinySSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], 
        #              [0.71, 0.79], [0.88, 0.961]]
        # 在TinySSD的__init__中，调整sizes为更小的值
        self.sizes = [[0.05, 0.1], [0.15, 0.2], [0.25, 0.3], 
             [0.35, 0.4], [0.45, 0.5]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = [len(s)+len(r)-1 for s,r in zip(self.sizes, self.ratios)]  # [4,4,4,4,4]
        
        # 初始化网络块
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(64 if i==0 else 128, self.num_anchors[i], num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(64 if i==0 else 128, self.num_anchors[i]))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [], [], []
        for i in range(5):
            X = getattr(self, f'blk_{i}')(X)
            anchors.append(multibox_prior(X, self.sizes[i], self.ratios[i]))
            cls_preds.append(getattr(self, f'cls_{i}')(X))
            bbox_preds.append(getattr(self, f'bbox_{i}')(X))
        
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes+1)
        bbox_preds = concat_preds(bbox_preds)
        
        return anchors, cls_preds, bbox_preds

# --- 测试 ---
if __name__ == "__main__":
    net = TinySSD(num_classes=1)
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)
    print("anchors shape:", anchors.shape)
    print("cls_preds shape:", cls_preds.shape)
    print("bbox_preds shape:", bbox_preds.shape)