import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


class EmotionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data_frame = dataframe
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx, 0]
        image = Image.open(img_name).convert('L')
        label = int(self.data_frame.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.fc1 = nn.Linear(in_features=64*8*8, out_features=128)
        self.fc1 = nn.Linear(in_features=64*32*32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*32*32)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classifier:
    def __init__(self, csv_file="dataset.csv", batch_size=256, learning_rate=0.001, epochs=25):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        df = pd.read_csv(csv_file)
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['label'], random_state=42)
        
        self.train_loader = DataLoader(EmotionDataset(dataframe=train_df), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(EmotionDataset(dataframe=val_df), batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(EmotionDataset(dataframe=test_df), batch_size=self.batch_size, shuffle=False)

        self.model = CNN(num_classes=5).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.best_val_loss = np.inf
        self.best_model_path = "best_model.pth"
        print(self.best_model_path)


    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            self.model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = 100 * correct / total
            val_loss = self.validate()
            
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(self.train_loader)}, Train Accuracy: {train_accuracy}%')
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)


    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        validation_accuracy = 100 * correct / total
        print(f'Validation Accuracy: {validation_accuracy}%, Validation Loss: {total_loss / len(self.val_loader)}')
        return total_loss / len(self.val_loader)


    def test(self):
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        test_accuracy = 100 * correct / total
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        cm = confusion_matrix(all_labels, all_predictions)
        print(f'Test Accuracy: {test_accuracy}%, Test Loss: {total_loss/len(self.test_loader)}, Macro F1 Score: {f1_score}')
        print('Test Confusion Matrix:\n', cm)


model_trainer = Classifier(batch_size=256, learning_rate=0.001, epochs=25)
model_trainer.train()
model_trainer.test()
