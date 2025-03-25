
in_features = 8  
out_features = 64  
num_heads = 8  
num_gat_layers = 3  
hidden_dim = 1024  
num_lstm_layers = 3  
decoder_out_dim = 8  


gat_lstm_decoder_model = GAT_LSTM_Decoder(in_features, out_features, num_heads, num_gat_layers, hidden_dim, num_lstm_layers, decoder_out_dim)
            
class NodeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NodeMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EdgeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EdgeMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features

        self.Wq = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        self.Wk = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        self.Wv = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        self.bq = nn.Parameter(torch.FloatTensor(num_heads, out_features))
        self.bk = nn.Parameter(torch.FloatTensor(num_heads, out_features))
        self.bv = nn.Parameter(torch.FloatTensor(num_heads, out_features))
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.Wq)
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wv)
        nn.init.zeros_(self.bq)
        nn.init.zeros_(self.bk)
        nn.init.zeros_(self.bv)

    def forward(self, X, A, edge_attributes=None):
       
        batch_size, num_nodes, in_features = X.size()

       
        Q = torch.einsum('bni,hio->bhno', X, self.Wq) + self.bq.unsqueeze(1)  # (batch_size, num_heads, num_nodes, out_features)
        K = torch.einsum('bni,hio->bhno', X, self.Wk) + self.bk.unsqueeze(1)
        V = torch.einsum('bni,hio->bhno', X, self.Wv) + self.bv.unsqueeze(1)
        
        
        Aatt = torch.einsum('bhno,bhmo->bhnm', Q, K)  # (batch_size, num_heads, num_nodes, num_nodes)

       
        if edge_attributes is not None:
           
            batch_size, num_nodes, _, num_edge_features = edge_attributes.size()
            
            edge_attr_weighted = self.mlp_g(edge_attributes.view(-1, num_edge_features))  

           
            edge_attr_weighted = edge_attr_weighted.view(batch_size, num_nodes, num_nodes)
            Aatt += edge_attr_weighted.unsqueeze(1)  

       
        Aatt = torch.softmax(Aatt, dim=-1)

        
        output = torch.einsum('bhnm,bhmo->bhno', Aatt, V)  

        return output.mean(dim=1)  


class GAT(nn.Module):
    def __init__(self, in_features, out_features, num_heads, num_layers):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_feat = in_features if i == 0 else out_features
            self.layers.append(GATLayer(in_feat, out_features, num_heads))

    def forward(self, X, A, edge_attributes=None):
        
        for layer in self.layers:
            X = layer(X, A, edge_attributes=edge_attributes)
        return X


class GAT_LSTM_Decoder(nn.Module):
    def __init__(self, in_features, out_features, num_heads, num_gat_layers, hidden_dim, num_lstm_layers, decoder_out_dim):
        super(GAT_LSTM_Decoder, self).__init__()
        
        
        self.node_mlp = NodeMLP(input_dim=in_features, hidden_dim=64, output_dim=out_features)
        self.edge_mlp = EdgeMLP(input_dim=6, hidden_dim=32, output_dim=32) 
        
       
        self.gat = GAT(out_features, out_features, num_heads, num_gat_layers)
        
      
        self.lstm = nn.LSTM(input_size=out_features, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, decoder_out_dim)  
        )

    def forward(self, X_seq, A_seq, edge_attributes_seq):
        
        batch_size, seq_len, num_nodes, in_features = X_seq.size()

        
        encoded_node_seq = []
        encoded_edge_seq = []  
        for t in range(seq_len):
            X_t = X_seq[:, t, :, :]  
            edge_attr_t = edge_attributes_seq[:, t, :, :]  
            
            
            encoded_nodes_t = self.node_mlp(X_t)
            encoded_node_seq.append(encoded_nodes_t)

           
            encoded_edges_t = self.edge_mlp(edge_attr_t)
            encoded_edge_seq.append(encoded_edges_t)

        encoded_node_seq = torch.stack(encoded_node_seq, dim=1)  
        encoded_edge_seq = torch.stack(encoded_edge_seq, dim=1)  

        
        gat_outputs = []
        for t in range(seq_len):
            encoded_nodes_t = encoded_node_seq[:, t, :, :] 
            A_t = A_seq[:, t, :, :]  
            encoded_edges_t = encoded_edge_seq[:, t, :, :]  
            
           
            gat_output_t = self.gat(encoded_nodes_t, A_t, encoded_edges_t)  
            gat_outputs.append(gat_output_t)

        
        gat_outputs = torch.stack(gat_outputs, dim=1)  

        
        ego_node_features = gat_outputs[:, :, 0, :]  

        
        lstm_output, (hn, cn) = self.lstm(ego_node_features)  

       
        predictions = self.decoder(lstm_output)  

        return predictions


          
           
    