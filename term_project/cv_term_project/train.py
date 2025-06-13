import os
import json
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import multiprocessing
from tqdm import tqdm

from model.resnet_50 import ResNet50


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




class ImageDataset(Dataset):
    """自定义数据集类，支持延迟加载"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 延迟加载图片
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_dataset_paths(dataset_json):
    """只加载图片路径和标签，不加载实际图片数据"""
    dataset_name = dataset_json['dataset_name']
    dataset_path = dataset_json['dataset_path']
    classes = dataset_json['classes']
    class_number = len(classes)

    print(f'Loading dataset paths for {dataset_name}...')

    cls2index = {cls: i for i, cls in enumerate(classes)}

    # 收集训练集路径
    train_paths, train_labels = [], []
    train_path = os.path.join(dataset_path, 'train')
    
    for cls in tqdm(os.listdir(train_path), desc="Loading train paths"):
        if cls not in cls2index:
            continue
        cls_idx = cls2index[cls]
        cls_path = os.path.join(train_path, cls)
        
        if not os.path.isdir(cls_path):
            continue
            
        for img in os.listdir(cls_path):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cls_path, img)
                train_paths.append(img_path)
                # 直接使用类别索引而不是one-hot编码
                train_labels.append(cls_idx)
    
    # 收集验证集路径
    val_paths, val_labels = [], []
    val_path = os.path.join(dataset_path, 'val')
    
    for cls in tqdm(os.listdir(val_path), desc="Loading val paths"):
        if cls not in cls2index:
            continue
        cls_idx = cls2index[cls]
        cls_path = os.path.join(val_path, cls)
        
        if not os.path.isdir(cls_path):
            continue
            
        for img in os.listdir(cls_path):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cls_path, img)
                val_paths.append(img_path)
                val_labels.append(cls_idx)
    
    print(f'Found {len(train_paths)} training images and {len(val_paths)} validation images')
    
    return train_paths, train_labels, val_paths, val_labels, class_number


def create_data_loaders(train_paths, train_labels, val_paths, val_labels, batch_size, num_workers=4):
    """创建数据加载器"""
    train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ImageDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


def train(model, train_loader, val_loader, num_epochs, learning_rate=0.001):
    """训练模型并在每个epoch后进行验证"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print(f'Training on device: {device}')
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 添加进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # 计算训练准确率
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算验证指标
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # 计算准确率
        val_acc = 100. * np.sum(all_predictions == all_targets) / len(all_targets)
        avg_val_loss = val_loss / len(val_loader)
        
        # 计算每个类别的Precision和Recall
        num_classes = len(np.unique(all_targets))
        precision_per_class = []
        recall_per_class = []
        
        for class_id in range(num_classes):
            tp = np.sum((all_predictions == class_id) & (all_targets == class_id))
            fp = np.sum((all_predictions == class_id) & (all_targets != class_id))
            fn = np.sum((all_predictions != class_id) & (all_targets == class_id))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
        
        # 计算宏平均Precision和Recall
        macro_precision = np.mean(precision_per_class)
        macro_recall = np.mean(recall_per_class)
        
        # 计算F1-Score
        f1_score = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0.0
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印训练结果
        print(f"\nEpoch [{epoch+1:3d}/{num_epochs}] | LR: {current_lr:.6f}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Precision:  {macro_precision:.4f} | Recall:    {macro_recall:.4f} | F1: {f1_score:.4f}")
        
        # 详细的每类指标
        if num_classes <= 10:
            print(f"  Per-class Precision: {[f'{p:.3f}' for p in precision_per_class]}")
            print(f"  Per-class Recall:    {[f'{r:.3f}' for r in recall_per_class]}")
        
        print("-" * 80)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best.pth')
    
    return model


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="dataset config path",
        default=None
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=20
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="number of workers for data loading"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate"
    )

    return parser


if __name__ == "__main__":
    # 设置多进程启动方式（Windows上可能需要）
    if __name__ == '__main__':
        multiprocessing.set_start_method('spawn', force=True)
    
    parser = build_parser()
    args = parser.parse_args()
    
    if args.dataset:
        with open(args.dataset, 'r') as fp:
            dataset_json = json.load(fp)
        
        # 加载数据集路径（快速）
        train_paths, train_labels, val_paths, val_labels, class_number = load_dataset_paths(dataset_json)
        
        # 创建数据加载器（支持多进程和延迟加载）
        train_loader, val_loader = create_data_loaders(
            train_paths, train_labels, val_paths, val_labels, 
            args.batch_size, args.num_workers
        )
        
        # 创建模型
        model = ResNet50(class_number)
        
        # 开始训练
        train(model, train_loader, val_loader, args.epoch, args.learning_rate)
        
    else:
        print('--dataset must be set.')