import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPSILON = 0.8
LAMBDA = 0.01
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IntrusionDetectionDataset(Dataset):
    def __init__(self, features, labels):
        """
        Custom PyTorch Dataset for Intrusion Detection
        
        Args:
            features (numpy.ndarray): Input features
            labels (numpy.ndarray): Corresponding labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Deep Q-Network Architecture
        
        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output classes
        """
        super(DeepQNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input features
        
        Returns:
            torch.Tensor: Output Q-values
        """
        return self.network(x)

class DQNTrainer:
    def __init__(self, input_dim, output_dim):
        """
        DQN Trainer Class
        
        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output classes
        """
        self.model = DeepQNetwork(input_dim, output_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.CrossEntropyLoss()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def epsilon_greedy_action(self, state, epsilon):
        """
        Epsilon-Greedy Action Selection
        
        Args:
            state (torch.Tensor): Current state
            epsilon (float): Exploration probability
        
        Returns:
            int: Selected action
        """
        if random.random() < epsilon:
            # Explore: Random action
            return random.randint(0, self.output_dim - 1)
        else:
            # Exploit: Best action
            with torch.no_grad():
                q_values = self.model(state.unsqueeze(0).to(DEVICE))
                return q_values.argmax().item()
    
    def train(self, train_loader, test_loader, epochs=EPOCHS):
        """
        Train the DQN model
        
        Args:
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Testing data loader
            epochs (int): Number of training epochs
        
        Returns:
            tuple: Training losses and accuracies
        """
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                
                # Forward pass
                outputs = self.model(batch_features)
                loss = self.loss_fn(outputs, batch_labels)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                epoch_total += batch_labels.size(0)
                epoch_correct += (predicted == batch_labels).sum().item()
                
                epoch_loss += loss.item()
            
            # Calculate epoch metrics
            epoch_accuracy = epoch_correct / epoch_total
            avg_loss = epoch_loss / len(train_loader)
            
            losses.append(avg_loss)
            accuracies.append(epoch_accuracy)
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            # Optional: Validate on test set
            self.validate(test_loader)
        
        return losses, accuracies
    
    def validate(self, test_loader):
        """
        Validate model on test dataset
        
        Args:
            test_loader (DataLoader): Testing data loader
        
        Returns:
            dict: Performance metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = self.model(features)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print("\nValidation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def predict(self, features):
        """
        Make predictions for input features
        
        Args:
            features (numpy.ndarray): Input features
        
        Returns:
            numpy.ndarray: Predicted labels
        """
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(DEVICE)
            outputs = self.model(features_tensor)
            _, predictions = torch.max(outputs, 1)
            return predictions.cpu().numpy()

def main():
    # Load and preprocess data
    data = pd.read_csv('src/data/ids_data.csv')
    
    # Separate features and labels
    X = data.iloc[:, 1:-1].values  # Exclude first and last columns
    y = data.iloc[:, -1].values
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = IntrusionDetectionDataset(X_train, y_train)
    test_dataset = IntrusionDetectionDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize and train DQN
    dqn = DQNTrainer(input_dim=X_train.shape[1], output_dim=len(np.unique(y)))
    
    # Train the model
    losses, accuracies = dqn.train(train_loader, test_loader)
    
    # Final validation
    dqn.validate(test_loader)
    
    # Optional: Save the model
    torch.save(dqn.model.state_dict(), 'ids_dqn_model.pth')

if __name__ == '__main__':
    main()
