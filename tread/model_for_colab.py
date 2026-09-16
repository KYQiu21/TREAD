import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNoise(nn.Module):
    def __init__(self, std: float = 0.0):
        super().__init__()
        self.std = float(std)

    def forward(self, x):
        if not self.training or self.std <= 0.0:
            return x

        mask = (x.abs().sum(dim=-1, keepdim=True) > 0).float()
        noise = torch.randn_like(x) * self.std
        return x + noise * mask

    
class DMDModel(nn.Module):
    
    def __init__(
        self, 
        per_resi_emb_dim=1024, 
        hidden_dim=64, 
        out_channel=64, 
        num_block=2, 
        dropout=0.3, 
        kernel_size_conv1=7, 
        kernel_size_block=7, 
        bilstm=True, 
        device='cuda', 
        input_noise_std=0.01,
        multi=False,
        num_types=6
    ):
        
        super(DMDModel, self).__init__()

        self.per_resi_emb_dim = per_resi_emb_dim
        self.out_channel = out_channel
        self.hidden_dim = hidden_dim
        self.kernel_size_conv1 = kernel_size_conv1
        self.kernel_size_block = kernel_size_block
        self.input_noise = GaussianNoise(input_noise_std)
        self.num_block = num_block
        self.bilstm = bilstm
        self.W_size = 64
        self.device = device
        self.multi = multi
        self.num_types = num_types
        
        # stem blocks
        self.conv1 = nn.Conv1d(in_channels=self.per_resi_emb_dim, out_channels=self.out_channel, kernel_size=self.kernel_size_conv1, padding='same')
        self.bn1 = nn.BatchNorm1d(self.out_channel)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=self.kernel_size_block, padding='same')
        self.bn2 = nn.BatchNorm1d(self.out_channel)
        self.conv3 = nn.Conv1d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=self.kernel_size_block, padding='same')
        self.dropout2 = nn.Dropout(dropout)

        # residual Blocks with Loop
        self.blocks = nn.ModuleList()
        for _ in range(self.num_block):  
            block = nn.Sequential(
                nn.BatchNorm1d(self.out_channel),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(self.out_channel, self.out_channel, kernel_size=self.kernel_size_block, padding='same'),
                nn.BatchNorm1d(self.out_channel),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(self.out_channel, self.out_channel, kernel_size=self.kernel_size_block, padding='same')
            )
            self.blocks.append(block)

        # BiLSTM Layer
        self.bilstm = nn.LSTM(input_size=self.out_channel, hidden_size=self.out_channel, num_layers=1, batch_first=True, bidirectional=True)

        # Final Fully Connected Layers
        if self.bilstm:
            self.final_bn = nn.BatchNorm1d(self.out_channel*2)  # Adjusted to 64 for BiLSTM output
            self.fc1 = nn.Linear(self.out_channel*2, self.out_channel)
        else:
            self.final_bn = nn.BatchNorm1d(self.out_channel) 
            self.fc1 = nn.Linear(self.out_channel, self.out_channel) 
            
        self.fc2 = nn.Linear(self.out_channel, self.hidden_dim)
        self.seg_head = nn.Linear(self.hidden_dim, 1)
        self.dropout3 = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
        if multi:
            self.type_heads = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(self.num_types)])

    def forward(self, x, resi_emb=False):
        # Initial Conv Layer
        x = self.input_noise(x)
        x = x.permute(0, 2, 1)  # Change to (batch, channels, seq_len)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.dropout1(y)

        # First Residual Connection
        shortcut = y
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.dropout2(y)
        y = self.conv3(y)
        y = y + shortcut

        # Residual Blocks
        for block in self.blocks:
            shortcut = y
            y = block(y)
            y = y + shortcut

        if self.bilstm:
            # BiLSTM Layer
            y = y.permute(0, 2, 1) 
            y, _ = self.bilstm(y)
            y = y.permute(0, 2, 1) 

        # Flatten BiLSTM output and pass through final layers
        y = F.relu(self.final_bn(y))
        emb = y.permute(0, 2, 1)  # Reshape to (batch, seq_len, channels) for the linear layer
        y = F.relu(self.fc1(emb))
        y = self.dropout3(y)
        y = F.relu(self.fc2(y))
        seg_logit = self.seg_head(y).squeeze(2)
        # use y for training to improve numerical stability but output the normalized y as well
        seg_logit_normalized = self.sigmoid(seg_logit)
        
        
        if self.multi:
            type_logit_list = [head(y).squeeze(-1) for head in self.type_heads]  # K * (B, L)
            type_logit = torch.stack(type_logit_list, dim=-1)                    # (B, L, K)
            type_logit_normalized = self.sigmoid(type_logit)
            
            out = {
                "seg_logit": seg_logit,
                "seg_prob": seg_logit_normalized,
                "type_logit": type_logit,
                "type_prob": type_logit_normalized,
            }
            
            if resi_emb:
                out["emb"] = h
            
            return out
                
        else:
            
            if resi_emb:
                return seg_logit_normalized, seg_logit, emb
            else:
                return seg_logit_normalized, seg_logit
               

    def predict_single(self, emb, W_size=64):

        '''
        receive a single sequence embedding, split it like the Dataset object into sub-embeddings, and pass them through the model
        '''

        seq_len = emb.shape[0]

        subseq_emb_batch = self.split_emb(emb, W_size)
        
        if self.multi:
            out = self.forward(subseq_emb_batch)
            seg_prob = out['seg_prob'].view(-1).detach().cpu().numpy()[:seq_len]
            type_prob = out['type_prob']
            type_prob = [
                type_prob[..., i].contiguous().view(-1).detach().cpu().numpy()[:seq_len]
                for i in range(type_prob.shape[-1])
            ]

            return seg_prob, type_prob
        
        else:
            normalized_prediction, prediction = self.forward(subseq_emb_batch)
            
            return normalized_prediction.view(-1).detach().cpu().numpy()[:seq_len]

    
    def split_emb(self, emb, win_len):

        sequence_length = emb.shape[0]
        subseq_emb_list = []

        for n in range(0, sequence_length, win_len):
            beg = n
            end = min(beg + win_len, sequence_length)
            window_emb = emb[beg:end, :]
            pad_matrix = torch.zeros((win_len, self.per_resi_emb_dim)).to(self.device)
            pad_matrix[0:window_emb.shape[0], 0:window_emb.shape[1]] = window_emb
            subseq_emb_list.append(pad_matrix)

        subseq_emb_batch = torch.stack(subseq_emb_list, dim=0)

        return subseq_emb_batch