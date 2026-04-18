import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.img_paths = []
        self.labels = []

        for label, subfolder in enumerate(['NORMAL', 'PNEUMONIA']):
            subfolder_path = os.path.join(data_dir, subfolder)
            for img_name in os.listdir(subfolder_path):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(subfolder_path, img_name)
                    self.img_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
    
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_dataloaders(processed_dir,batch_size=32):
    train_dataset = ChestXrayDataset(os.path.join(processed_dir, 'train'), transform=train_transforms)
    val_dataset = ChestXrayDataset(os.path.join(processed_dir, 'val'), transform=val_test_transforms)
    test_dataset = ChestXrayDataset(os.path.join(processed_dir, 'test'), transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader