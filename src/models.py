import torch
import torch.nn as nn

class PrimeNet(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
    
    
class PrimeCNN(nn.Module):

    def __init__(self, input_length):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.MaxPool1d(2),  

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        reduced_length = input_length // 2  

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(128 * reduced_length, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = self.conv(x)        
        x = self.flatten(x)

        x = self.fc(x)

        return x.squeeze(1)
    
    
class PrimeTransformer(nn.Module):

    def __init__(self, seq_len, d_model=128, nhead=4, num_layers=3):
        super().__init__()

        self.embedding = nn.Embedding(2, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.norm = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = x.long()
        B, L = x.shape

        x = self.embedding(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:, :L+1]

        x = self.transformer(x)

        x = self.norm(x)

        # Use CLS token output
        x = x[:, 0]

        x = self.fc(x)

        return x.squeeze(1)