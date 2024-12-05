import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        attention_maps = []
        for attn, ff in self.layers:
            x_attn, attn_map = attn(x) # Call attention layer and get its output and attention map
            x = x_attn + x  # Apply residual connection
            attention_maps.append(attn_map)  # Save attention map
            # x = self.norm(x) + x  # Apply residual connection and normalization
            x = ff(x) + x  # Call feedforward layer and apply residual connection

        return self.norm(x), attention_maps
    
# class Prior_Embedding(nn.Module):
#     def __init__(self, num_patches, embedding_dim, num_classes, alpha_weight):
#         super().__init__()
#         self.num_patches = num_patches
#         # Linear layer to transform the prior into desired shape
#         self.transform = nn.Linear(num_classes, num_patches * embedding_dim)
#         # Learnable weight parameter
#         self.alpha = nn.Parameter(torch.tensor(1.0))
#         self.alpha_weight = alpha_weight
        
#     def forward(self, patches, priors):
#         # patches shape: (B, num_patches, embedding_dim)
#         # priors shape: (B, embedding_dim)
        
#         # Transform each prior in the batch
#         transformed_priors = self.transform(priors)
        
#         # Reshape the transformed priors to match the patches shape
#         priors_embedding = transformed_priors.view(patches.shape[0], self.num_patches, -1)

#         alpha_sigmoid = self.alpha_weight * torch.sigmoid(self.alpha)
        
#         return patches + alpha_sigmoid * priors_embedding
        # return patches + priors_embedding
    
class PViT(nn.Module):
    def __init__(self, *, args, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if patch_dim == dim:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.LayerNorm(patch_dim),
            )
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))  
        # self.alpha_weight = args.alpha_weight
        # self.prior_embedding = Prior_Embedding(num_patches, dim, num_classes, self.alpha_weight)
        self.prior_projection = nn.Linear(num_classes, dim)  # Project prior to the embedding dimension   
        # Learnable scale factor     
        self.scale_factor = nn.Parameter(torch.full((1,), args.alpha_weight))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        # self.mlp_head = nn.Linear(dim, num_classes) #
        self.mlp_head = nn.Linear(dim, num_classes)
    
    def forward(self, img, prior):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add positional embeddings to patch embeddings
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # Class token + patch embeddings
        x += self.pos_embedding[:, :n+1]  # Positional embeddings for cls token and patches
        x = self.dropout(x)

        # Normalize and project the prior, then apply scaling
        prior_normalized = F.softmax(prior, dim=-1)
        prior_fes = self.prior_projection(prior_normalized)
        prior_token = prior_fes.unsqueeze(1).expand(b, 1, -1)
        prior_token = prior_token * self.scale_factor

        # Concatenate the prior token to the sequence without positional encoding
        x = torch.cat((x, prior_token), dim=1)

        # Pass through the transformer
        x, attention_maps = self.transformer(x)

        # Pooling and classifier head
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        # Return logits and optionally attention maps
        return prior_fes, x, self.mlp_head(x)
