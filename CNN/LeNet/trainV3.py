import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset
import torch.optim as optim
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
import psutil
import gc

class TrainingConfig:
    def __init__(self):
        self.data_root = Path("./data/data")
        self.batch_size = 64
        self.num_workers = 4
        self.epochs = 20
        self.lr = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path("./saved_models")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = Path("./logs/training.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.vram_threshold = 7000

def setup_logger(config):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(config.log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def check_vram_usage(config, logger):
    if config.device.type == 'cuda':
        vram_used = torch.cuda.memory_allocated() / 1024**2
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        logger.info(f"VRAM Usage: {vram_used:.1f}MB / {vram_total:.1f}MB")
        if vram_used > config.vram_threshold:
            logger.warning(f"High VRAM usage: {vram_used:.1f}MB > threshold {config.vram_threshold}MB")

def create_data_loaders(config, logger):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        train_dataset = datasets.CIFAR10(
            root=config.data_root,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=config.data_root,
            train=False,
            download=True,
            transform=transform
        )
    except Exception as e:
        logger.error(f"Dataset loading failed: {str(e)}")
        raise
    train_subset = Subset(train_dataset, indices=range(1000))
    test_subset  = Subset(test_dataset,  indices=range(5000))


    train_loader = DataLoader(
        dataset=train_subset,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_subset,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    return train_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, config, logger, epoch):
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(config.device, non_blocking=True)
        labels = labels.to(config.device, non_blocking=True)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        
        if batch_idx % 50 == 0:
            check_vram_usage(config, logger)
            avg_loss = running_loss / total_samples
            logger.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
    
    return running_loss / total_samples

def evaluate_model(model, test_loader, criterion, config):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / total
    return avg_loss, accuracy

def save_model(model, config, accuracy, epoch):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"cifar10_model_ep{epoch+1}_acc{accuracy:.2f}_{timestamp}.pth"
    save_path = config.save_dir / model_name
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'epoch': epoch
    }, save_path)
    return save_path

def main():
    config = TrainingConfig()
    logger = setup_logger(config)
    
    logger.info("==== Training Configuration ====")
    logger.info(f"Device: {config.device}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Learning rate: {config.lr}")
    logger.info(f"VRAM threshold: {config.vram_threshold}MB")
    
    try:
        train_loader, test_loader = create_data_loaders(config, logger)
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logger.exception("Data loading failed")
        return
    
    from netv2 import LeNet
    model = LeNet().to(config.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    logger.info(f"Model architecture:\n{model}")
    check_vram_usage(config, logger)
    
    best_acc = 0.0
    acc_history = []
    loss_history = []
    
    logger.info("==== Starting Training ====")
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch+1}/{config.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config, logger, epoch)
        loss_history.append(train_loss)
        
        test_loss, accuracy = evaluate_model(model, test_loader, criterion, config)
        acc_history.append(accuracy)
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
        if accuracy > best_acc:
            best_acc = accuracy
            model_path = save_model(model, config, accuracy, epoch)
            logger.info(f"New best model saved: {model_path}")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    final_model_path = save_model(model, config, accuracy, config.epochs-1)
    logger.info(f"Training completed. Final model: {final_model_path}")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, 'b-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(acc_history, 'r-o', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./cifar10_training_results.png')
    logger.info("Training results plot saved")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        import traceback
        print(f"Unexpected error: {traceback.format_exc()}")