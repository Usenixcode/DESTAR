import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn.init as init
import pandas as pd
import numpy as np
from mlp import GAT_LSTM_Decoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

class VehicleDataDataset(Dataset):
    def __init__(self, csv_file, normalize=True, method="standard"):
        self.data = pd.read_csv(csv_file)
        
        num_samples = (len(self.data) // 11) * 11 
        self.data = self.data.iloc[:num_samples]  
        
        
        self.samples = self.data.values.reshape(-1, 11, self.data.shape[1])  
        
        self.normalize = normalize
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling method. Use 'standard' or 'minmax'.")
        
        self.fit_scaler()

    def fit_scaler(self):
        all_features = self.samples[:, :, 1:89].reshape(-1, 8)  
        self.scaler.fit(all_features)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]  
        
        num_frames = sample.shape[0]  
        
        node_features = sample[:, 1:89].reshape(num_frames, 11, 8)  
        node_features = self.scaler.transform(node_features.reshape(-1, 8)).reshape(num_frames, 11, 8)
        true_x = node_features[:, :, 0]
        true_y = node_features[:, :, 1]
        true_yaw = node_features[:, :, 2]
        true_vx = node_features[:, :, 3]
        true_vy = node_features[:, :, 4]
        true_yawrate = node_features[:, :, 5]
        true_ax = node_features[:, :, 6]
        true_ay = node_features[:, :, 7]
        edge_attributes = sample[:, 89:815].reshape(num_frames, 11, 11, 6)  
        adj_matrix = sample[:, 815:].reshape(num_frames, 11, 11)  

        return (
            torch.tensor(node_features, dtype=torch.float32),  
            torch.tensor(adj_matrix [0], dtype=torch.float32),  
            torch.tensor(edge_attributes, dtype=torch.float32),  
            true_x, true_y, true_yaw, true_vx, true_vy, true_yawrate, true_ax, true_ay
        )


class RLTrainer:
    def __init__(self, model, optimizer, batch_size, gamma, ppo_eps=0.2, entropy_coef=0.01, loss_weights=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.gamma = gamma
        self.ppo_eps = ppo_eps
        self.entropy_coef = entropy_coef
        self.loss_weights = loss_weights or {
            'x': 3.0,
            'y': 3.0,
            'yaw': 2.0,
            'vx': 2.0,
            'vy': 2.0,
            'yawrate': 1.0,
            'ax': 1.0,
            'ay': 1.0
        }
        self.best_loss = float('inf')

    def compute_loss(self, predictions, action, log_prob, reward, entropy):
        old_log_prob = log_prob.detach()
        ratio = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - self.ppo_eps, 1 + self.ppo_eps)
        policy_loss = -torch.min(ratio * reward, clipped_ratio * reward).mean()
        entropy_loss = -self.entropy_coef * entropy.mean()
        return policy_loss + entropy_loss



    def forward_pass(self, node_features, adj_matrix, edge_attributes_seq):
        node_features = node_features.permute(0, 2, 1, 3)
        batch_size, seq_len, num_nodes, in_features = node_features.shape
        adj_matrix = adj_matrix[0].unsqueeze(0).expand(batch_size, seq_len, 11, 11).contiguous()
        edge_attributes_seq = edge_attributes_seq.permute(0, 3, 1, 2, 4)
        predictions = self.model(node_features, adj_matrix, edge_attributes_seq)
        predictions = torch.softmax(predictions, dim=-1)
        action_dist = Categorical(predictions)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)  
        entropy = action_dist.entropy()  

        return predictions, action, log_prob, entropy  


    def get_old_log_prob(self, predictions, action):
        old_pred = predictions.detach()
        old_prob = Categorical(old_pred).log_prob(action)
        return old_prob

    def get_reward(self, node_features, adj_matrix, edge_attributes_seq, true_x, true_y, true_yaw, true_vx, true_vy, true_yawrate, true_ax, true_ay):
        batch_size, seq_len, num_nodes, _ = node_features.shape
        adj_matrix = adj_matrix.squeeze()
        if len(adj_matrix.shape) == 3:
            adj_matrix = adj_matrix.unsqueeze(1).expand(-1, node_features.shape[1], -1, -1)
        model_output = self.model(
            node_features, 
            adj_matrix,
            edge_attributes_seq
        )
        predicted_x = model_output[:, :, 0:1]  # [batch, nodes, 2]
        predicted_y = model_output[:, :, 1:2]
        predicted_yaw = model_output[:, :, 2:3] * (np.pi / 180)
        predicted_vx = model_output[:, :, 3:4]
        predicted_vy = model_output[:, :, 4:5]
        predicted_yawrate = model_output[:, :, 5:6]* (np.pi / 180)
        predicted_ax = model_output[:, :, 6:7]
        predicted_ay = model_output[:, :, 7:8]
        true_x = node_features[:, -1, :, 0:1]
        true_y = node_features[:, -1, :, 1:2]
        true_yaw = node_features[:, -1, :, 2:3]
        true_vx = node_features[:, -1, :, 3:4]
        true_vy = node_features[:, -1, :, 4:5]
        true_yawrate = node_features[:, -1, :, 5:6]
        true_ax = node_features[:, -1, :, 6:7]
        true_ay = node_features[:, -1, :, 7:8]

        true_x, predicted_x = self.normalize(true_x, predicted_x)
        true_y, predicted_y = self.normalize(true_y, predicted_y)
        true_vx, predicted_vx = self.normalize(true_vx, predicted_vx)
        true_vy, predicted_vy = self.normalize(true_vy, predicted_vy)
        true_yaw, predicted_yaw = self.normalize(true_yaw, predicted_yaw)
        true_yawrate, predicted_yawrate = self.normalize(true_yawrate, predicted_yawrate)
        true_ax, predicted_ax = self.normalize(true_ax, predicted_ax)
        true_ay, predicted_ay = self.normalize(true_ay, predicted_ay)

        loss_x = F.smooth_l1_loss(predicted_x, true_x) * self.loss_weights['x']
        loss_y = F.smooth_l1_loss(predicted_y, true_y) * self.loss_weights['y']
        loss_vx = F.smooth_l1_loss(predicted_vx, true_vx) * self.loss_weights['vx']
        loss_vy = F.smooth_l1_loss(predicted_vy, true_vy) * self.loss_weights['vy']
        loss_yaw = self.cosine_similarity_loss(predicted_yaw, true_yaw) * self.loss_weights['yaw']
        loss_yawrate = F.smooth_l1_loss(predicted_yawrate, true_yawrate) * self.loss_weights['yawrate']
        loss_ax = F.smooth_l1_loss(predicted_ax, true_ax) * self.loss_weights['ax']
        loss_ay = F.smooth_l1_loss(predicted_ay, true_ay) * self.loss_weights['ay']

        
        total_loss = (loss_x + loss_y + loss_vx + loss_vy + loss_yaw + loss_yawrate + loss_ax + loss_ay) / 8
        reward = -total_loss.mean()
        return reward
    def normalize(self, true_values, predicted_values):
        mean = true_values.mean(dim=(0, 1), keepdim=True)
        std = true_values.std(dim=(0, 1), keepdim=True)
        true_values = (true_values - mean) / std
        predicted_values = (predicted_values - mean) / std
        return true_values, predicted_values


    def forward(self, node_features, adj_matrix, edge_attributes):
        batch_size, seq_len, num_nodes, _ = node_features.shape
        

        node_features = node_features.view(batch_size * seq_len, num_nodes, -1)
        adj_matrix = adj_matrix.view(batch_size * seq_len, num_nodes, num_nodes)
        edge_attributes = edge_attributes.view(batch_size * seq_len, num_nodes, num_nodes, -1)

        gat_outputs = node_features
        for gat_layer in self.gat_layers:
            gat_outputs = gat_layer(gat_outputs, adj_matrix, edge_attributes)
        

        gat_outputs = gat_outputs.view(batch_size, seq_len, num_nodes, -1)
        

        lstm_input = gat_outputs.permute(0, 2, 1, 3)  # [batch, num_nodes, seq_len, features]
        

        h = torch.zeros(self.num_lstm_layers, batch_size * num_nodes, self.hidden_dim).to(node_features.device)
        c = torch.zeros(self.num_lstm_layers, batch_size * num_nodes, self.hidden_dim).to(node_features.device)
        

        lstm_output, _ = self.lstm(
            lstm_input.contiguous().view(batch_size * num_nodes, seq_len, -1),
            (h, c)
        )
        

        decoded = self.fc(lstm_output[:, -1, :])  
        

        decoded = decoded.view(batch_size, num_nodes, 8)
        
        return decoded

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

    def train(self, train_loader, val_loader, epochs, save_model_path="best_model.pth"):
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.8, min_lr=1e-7, verbose=True)
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            if epoch < epochs * 0.33:  
                self.loss_weights = {'x': 3.0, 'y': 3.0, 'yaw': 0, 'vx': 0, 'vy': 0, 'yawrate': 0, 'ax': 0, 'ay': 0}
                lr = 3e-4
            elif epoch < epochs * 0.66:
                self.loss_weights = {'x': 3.0, 'y': 3.0, 'yaw': 2.0, 'vx': 2.0, 'vy': 2.0, 'yawrate': 0, 'ax': 0, 'ay': 0}
                lr = 1e-4
            else:  
                self.loss_weights = {'x': 3.0, 'y': 3.0, 'yaw': 2.0, 'vx': 2.0, 'vy': 2.0, 'yawrate': 1.0, 'ax': 1.0, 'ay': 1.0}
                lr = 1e-5
            self.model.train()
            total_loss = 0
            for node_features, adj_matrix, edge_attributes_seq, true_x, true_y, true_yaw, true_vx, true_vy, true_yawrate, true_ax, true_ay in train_loader:
                self.optimizer.zero_grad()
                predictions, action, log_prob, entropy = self.forward_pass(node_features, adj_matrix, edge_attributes_seq)
                reward = self.get_reward(node_features, adj_matrix, edge_attributes_seq, true_x, true_y, true_yaw, true_vx, true_vy, true_yawrate, true_ax, true_ay)

                loss = self.compute_loss(predictions, action, log_prob, reward, entropy)
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss}')
            
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for node_features, adj_matrix, edge_attributes_seq, true_x, true_y, true_yaw, true_vx, true_vy, true_yawrate, true_ax, true_ay in val_loader:
                    predictions, action, log_prob, entropy = self.forward_pass(node_features, adj_matrix, edge_attributes_seq)
                    reward = self.get_reward(node_features, adj_matrix, edge_attributes_seq, true_x, true_y, true_yaw, true_vx, true_vy, true_yawrate, true_ax, true_ay)
                    val_loss = self.compute_loss(predictions, action, log_prob, reward, entropy)
                    total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                torch.save(self.model.state_dict(), save_model_path)
                print(f"Model saved at epoch {epoch} with validation loss {self.best_loss}")
            scheduler.step(avg_val_loss)
        return train_losses, val_losses

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0) 
        elif isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

def main():

    dataset = [VehicleDataDataset(csv_file=f'vehicle_data{i}.csv') for i in range(1, 5)]
    dataset.fit_scaler()
    total_size = len(dataset)


    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GAT_LSTM_Decoder(in_features, out_features, num_heads, num_gat_layers, 
                            hidden_dim, num_lstm_layers, decoder_out_dim)
    initialize_weights(model)
    checkpoint_path = "best_model.pth"  
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"✅ Successfully loaded existing model: {checkpoint_path}")
    except FileNotFoundError:
        print(f"⚠️ Model not found: {checkpoint_path}, starting training from scratch")
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    trainer = RLTrainer(model=model, optimizer=optimizer, batch_size=batch_size, 
                       gamma=0.99, ppo_eps=0.5, entropy_coef=0.001)


    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs)
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for node_features, adj_matrix, edge_attributes_seq, true_x, true_y, true_yaw, true_vx, true_vy, true_yawrate, true_ax, true_ay in test_loader:
            predictions, action, log_prob, entropy = trainer.forward_pass(
                node_features, adj_matrix, edge_attributes_seq)
            reward = trainer.get_reward(node_features, adj_matrix, edge_attributes_seq, true_x, true_y, true_yaw, true_vx, true_vy, true_yawrate, true_ax, true_ay)
            test_loss = trainer.compute_loss(predictions, action, log_prob, reward, entropy)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f'Final Test Loss: {avg_test_loss}')
    torch.save(model.state_dict(), 'final_model.pth')

if __name__ == '__main__':
    main()