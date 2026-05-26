import torch 
import torch.nn as nn

class CLUBForCategorical(nn.Module): # Update 04/27/2022
    '''
    This class provide a CLUB estimator to calculate MI upper bound between vector-like embeddings and categorical labels.
    Estimate I(X,Y), where X is continuous vector and Y is discrete label.
    '''
    def __init__(self, input_dim, label_num, hidden_size=None):
        '''
        input_dim : the dimension of input embeddings
        label_num : the number of categorical labels 
        '''
        super().__init__()
        
        if hidden_size is None:
            self.variational_net = nn.Linear(input_dim, label_num)
        else:
            self.variational_net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, label_num)
            )
            
    def forward(self, inputs, labels):
        '''
        inputs : shape [batch_size, input_dim], a batch of embeddings
        labels : shape [batch_size], a batch of label index
        '''
        logits = self.variational_net(inputs)  #[sample_size, label_num]
        
        # log of conditional probability of positive sample pairs
        #positive = - nn.functional.cross_entropy(logits, labels, reduction='none')    
        sample_size, label_num = logits.shape
        
        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)  # shape [sample_size, sample_size, label_num]
        labels_extend = labels.unsqueeze(0).repeat(sample_size, 1)     # shape [sample_size, sample_size]

        # log of conditional probability of negative sample pairs
        log_mat = - nn.functional.cross_entropy(
            logits_extend.reshape(-1, label_num),
            labels_extend.reshape(-1, ),
            reduction='none'
        )
        
        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat).mean()
        
        # negative = log_mat.mean()
        n = log_mat.size(0)
        sum_all = log_mat.sum()
        sum_diag = log_mat.diag().sum()
        negative = (sum_all - sum_diag) / (n * (n - 1) + 1e-12)
        return positive - negative

    def loglikeli(self, inputs, labels):
        logits = self.variational_net(inputs)
        return - nn.functional.cross_entropy(logits, labels)
    
    def learning_loss(self, inputs, labels):
        return - self.loglikeli(inputs, labels)

class DACQKV(nn.Module):
    """
    Cross-layer QKV attention over tapped layer summaries, then a small MLP classifier.
    X: [B, L, D]  -> logits: [B, K]
    """
    def __init__(
        self,
        d_model: int,
        label_num: int,
        n_layers_tapped: int,
        n_heads: int = 4,
        mlp_ratio: float = 0.5,
        dropout: float = 0.1,
        use_layer_pos: bool = True,
        query_mode: str = "learned",            # "learned" | "conditioned" | "multi"
        n_queries: int = 4,
    ):
        super().__init__()
        self.use_layer_pos = use_layer_pos
        self.query_mode = query_mode
        self.n_queries = n_queries
        
        if use_layer_pos:
            self.layer_pos = nn.Parameter(torch.zeros(n_layers_tapped, d_model))
            nn.init.normal_(self.layer_pos, std=0.02)
            
        if query_mode == "learned":
            self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        elif query_mode == "conditioned":
            # make the query depend on the current input features
            # tiny network on pooled layer summaries
            self.q_net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        elif query_mode == "multi":
            assert n_queries >= 2, "Use n_queries>=2 for query_mode='multi'."
            self.query = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
            self.gate_ln = nn.LayerNorm(d_model)      
            self.gate = nn.Linear(d_model, 1)
            self.gate_tau = 0.5
        else:
            raise ValueError(f"Unknown query_mode: {query_mode}")

        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm  = nn.LayerNorm(d_model)

        hidden = max(1, int(d_model * mlp_ratio))
        self.cls = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, label_num),
        )

    def _pos(self, X):
        return X + self.layer_pos.unsqueeze(0) if self.use_layer_pos else X

    def _make_query(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, L, D] -> Q: [B, Q, D]
        """
        B, L, D = X.shape
        if self.query_mode == "learned":
            return self.query.expand(B, -1, -1)  # [B,1,D]

        elif self.query_mode == "conditioned":
            # condition query on current features (use mean over layers)
            # stop-grad is NOT used; this is part of the probe. (Later you can freeze it.)
            pooled = X.mean(dim=1)               # [B,D] # TODO we should check if we can change this.
            q = self.q_net(pooled).unsqueeze(1)  # [B,1,D]
            return q

        elif self.query_mode == "multi":
            return self.query.expand(B, -1, -1)  # [B,M,D]

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        p0 = next(self.parameters())
        tgt_dtype, tgt_device = p0.dtype, p0.device
        X = X.to(dtype=tgt_dtype, device=tgt_device)
        X = self._pos(X)                         # [B,L,D] - add layer positional embeddings
        
        # X_bottleneck = self.dac_qkv_dwn_prj(X)         # [B,L,D] -> [B,L,bottleneck_dim]
        Q = self._make_query(X)                  # [B,Q,D]

        out, _ = self.attn(Q, X, X, need_weights=False)  # [B,Q,D]

        # Project back up to original d_model space
        # out = self.dac_qkv_up_prj(out)                  # [B,Q,bottleneck_dim] -> [B,Q,D]
        
        if out.size(1) > 1 and self.query_mode == "multi":          
            g = self.gate(self.gate_ln(out))                        # [B,M,1]
            w = torch.softmax(g.squeeze(-1) / self.gate_tau, dim=1) # [B,M]
            
            # print(f"Gate weights w: {w[0]}")  # First sample
            # print(f"Gate max: {w.max(dim=1).values.mean():.4f}")
            # print(f"Gate entropy: {-(w * torch.log(w + 1e-10)).sum(-1).mean():.4f}")
    
            out = (w.unsqueeze(-1) * out).sum(dim=1, keepdim=True)  # [B,1,D]
        # print(f"Attention output shape: {out.shape}")
        # print(f"Attention weights shape: {attn_weights.shape}")
        # print(f"Attention weight entropy: {-(attn_weights * torch.log(attn_weights + 1e-10)).sum(-1).mean():.4f}")
        z = self.norm(out.squeeze(1))            # [B,D]
        return z

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z = self.embed(X)
        return self.cls(z)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

class DACMLP(nn.Module):
    """
    Simple MLP-based DAC classifier - easier to train than QKV attention.
    Supports same features as DACQKV: layer positional embeddings, freezing, etc.
    X: [B, L, D]  -> logits: [B, K]
    """
    def __init__(
        self,
        d_model: int,
        label_num: int,
        n_layers_tapped: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        use_layer_pos: bool = True,
        pooling_mode: str = "mean",  # "mean" | "max" | "weighted"
    ):
        super().__init__()
        self.use_layer_pos = use_layer_pos
        self.pooling_mode = pooling_mode
        
        # Layer positional embeddings (same as DACQKV)
        if use_layer_pos:
            self.layer_pos = nn.Parameter(torch.zeros(n_layers_tapped, d_model))
            nn.init.normal_(self.layer_pos, std=0.02)
        
        # Pooling network
        if pooling_mode == "weighted":
            # Learnable attention weights for pooling
            self.pool_attn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            )
        
        # Pre-pooling feature transformation
        self.pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Classifier MLP
        inner_dim = int(d_model * mlp_ratio)
        self.cls = nn.Sequential(
            nn.Linear(d_model, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, label_num),
        )

    def _pos(self, X):
        """Add layer positional embeddings"""
        return X + self.layer_pos.unsqueeze(0) if self.use_layer_pos else X

    def _pool(self, X: torch.Tensor) -> torch.Tensor:
        """
        Pool over layer dimension.
        X: [B, L, D] -> [B, D]
        """
        if self.pooling_mode == "mean":
            return X.mean(dim=1)
        elif self.pooling_mode == "max":
            return X.max(dim=1)[0]
        elif self.pooling_mode == "weighted":
            # Learnable weighted pooling
            attn_scores = self.pool_attn(X).squeeze(-1)  # [B, L]
            attn_weights = torch.softmax(attn_scores, dim=1)  # [B, L]
            return (attn_weights.unsqueeze(-1) * X).sum(dim=1)  # [B, D]
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        """Extract embedding before classifier (useful for probing)"""
        p0 = next(self.parameters())
        tgt_dtype, tgt_device = p0.dtype, p0.device
        X = X.to(dtype=tgt_dtype, device=tgt_device)
        
        X = self._pos(X)                    # [B, L, D] with positional embeddings
        X = self.pool(X)                    # [B, L, D] transformed
        z = self._pool(X)                   # [B, D] pooled
        return z

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, L, D] where L = n_layers_tapped
        Returns: logits [B, num_classes]
        """
        z = self.embed(X)
        return self.cls(z)

    def freeze(self):
        """Freeze all parameters and set to eval mode"""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()