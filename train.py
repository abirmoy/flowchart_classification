import torch
import torch.nn as nn
from torchvision.models import VisionTransformer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import sys

# 1. Configuration
config = {
    "data_path": "dataset",  # Path to your dataset
    "batch_size": 32,
    "image_size": 224,
    "num_workers": 4,
    "lr": 1e-4,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "flowchart_vit.pth",
}

# 2. Dataset Class
class FlowchartDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = ['non_flowchart', 'flowchart']
        self.samples = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 3. Data Transforms
transform = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Create ViT Model
def create_vit_model(num_classes=2):
    # Using torchvision's built-in ViT
    model = VisionTransformer(
        image_size=config['image_size'],
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=num_classes
    )
    return model

# 5. Training Function
def train_model():
    # Create datasets
    train_dataset = FlowchartDataset(config['data_path'], 'train', transform)
    val_dataset = FlowchartDataset(config['data_path'], 'val', transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, num_workers=config['num_workers'])
    
    # Initialize model
    model = create_vit_model().to(config['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(config['device']), labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
    
    torch.save(model.state_dict(), config['save_path'])
    print(f'Model saved to {config["save_path"]}')

if __name__ == '__main__':
    train_model()