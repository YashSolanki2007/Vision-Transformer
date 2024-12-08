'''
Basically the same architechture as a transformer
Splits an image into seperate parts similar to splitting words at some tokenization level
while feeding the fixed size patches also provide position embedding concat with that

Transformer inputs
1. 1D sequence of token embeddings
2. Reshape the image into (B, H * W * C)
3. Patch Shape: N x (P^2 * C) where N = HW/P^2 (N - Num of patches) thus each image is fed to the transformer by breaking into patches thus N = SEQ_LENGTH
4. The class embedding is passed along with the image into the transformer encoder

'''

import torch
from torch import nn
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

df = pd.read_csv("/content/sample_data/mnist_train_small.csv")[:2000]

df.head()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_images = torch.tensor(df.iloc[:, 1:].values, device=device)
train_labels = torch.tensor(df.iloc[:, 0].values, device=device)

train_images.shape, train_labels.shape

train_images = train_images / 255.0

train_images.shape[0]

# Splitting the image into patches
img_height = 28
img_width = 28
img_channels = 1
num_patches = 4

train_images_3d = train_images.reshape(train_images.shape[0], img_channels, img_height, img_width)

# Final shape after patching all images (B, N, P*P*C)
patch_size = int(math.sqrt((img_height * img_width) / num_patches))
stride_size = patch_size    # Probably does not have to be but just keeping for now to avoid overlap between images

patches = train_images_3d.unfold(2, patch_size, stride_size).unfold(3, patch_size, stride_size)
patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

patches = patches.view(*patches.size()[:3], -1)
train_patches = patches.view(patches.size(0), patches.size(1) * patches.size(2), patches.size(3)).to(device)

dataset = DataLoader(torch.utils.data.TensorDataset(train_patches, train_labels), batch_size=100)

sequence_length = train_patches.size(1)
pos_embed_seq_length = sequence_length + 1

# embed_dim = D as per the paper probably
embed_dim = 768
n_heads = 12

class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, embed_dim):
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        position_embed = torch.zeros(self.sequence_length, self.embed_dim, device=device)
        for pos in range(len(position_embed)):
            for i in range(len(position_embed[pos])):
                div_term = 10000 ** (2 * i / self.embed_dim)
                if i % 2 == 0:
                    position_embed[pos][i]  = math.sin(pos / div_term)
                else:
                    position_embed[pos][i]  = math.cos(pos / div_term)

        # Persists this in model training
        self.register_buffer('position_embed', position_embed)


    def forward(self, token_embed):
        return token_embed + self.position_embed

class PatchEmbedding(nn.Module):
  def __init__(self, patch_size, img_channels, num_embed, embed_dim, pos_emb_seq_length):
    super().__init__()
    self.patch_size = patch_size
    self.img_channels = img_channels
    self.num_embed = num_embed
    self.embed_dim = embed_dim
    self.pos_emb_seq_length = pos_emb_seq_length

    self.inp_embed = nn.Embedding(num_embeddings=self.num_embed, embedding_dim=self.embed_dim)
    self.pos_emb = PositionalEmbedding(self.pos_emb_seq_length, self.embed_dim)
    self.projection = nn.Linear(self.patch_size * self.patch_size * self.img_channels, self.embed_dim)
    self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_dim,  device=device))

  def forward(self, train_patches):
    o = self.projection(train_patches)
    class_token = self.class_token.expand(o.size(0), -1, -1)
    add_inp_embed = torch.cat((o, class_token), 1)
    pos_emb = self.pos_emb(add_inp_embed)

    '''
    Output shape: (B, N+1, D) or (B, pos_seq_length, embed_dim)
    '''
    return pos_emb

'''
For scaled dot product attention


FOR BASE IMPLEMENTATION AS PER PAPER
d_model = 768
head_size = 64
num_heads = 12
num_blocks = 12

d_k = d_v = d_q = d_model =

'''


num_heads = 12
head_size = embed_dim // num_heads

class ScaledDotProductAttention(nn.Module):
  def __init__(self, embed_dim, head_size):
    super().__init__()
    self.embed_dim = embed_dim
    self.head_size = head_size
    self.queries = nn.Linear(self.embed_dim, self.head_size, bias=False).to(device)
    self.keys = nn.Linear(self.embed_dim, self.head_size, bias=False).to(device)
    self.values = nn.Linear(self.embed_dim, self.head_size, bias=False).to(device)
    self.head_size = head_size

  def forward(self, x):
    B,T,C = x.shape
    q = self.queries(x)
    k = self.keys(x)
    v = self.values(x)

    weights = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)
    weights = torch.softmax(weights, dim=-1)
    out = weights @ v
    return out


class MultiHeadedAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, head_size):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_size = head_size

    self.heads = nn.ModuleList([ScaledDotProductAttention(self.embed_dim, self.head_size) for _ in range(self.num_heads)])
    self.projection = nn.Linear(self.num_heads * self.head_size, self.embed_dim).to(device)

  def forward(self, x):
    attn_out = torch.cat([head(x) for head in self.heads], dim=-1)
    out = self.projection(attn_out)
    return out

'''
Feed Forward information as per base model

MLP_SIZE = 3072
MLP_SIZE = 4 * embed_dim
'''

class FeedForward(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.embed_dim = embed_dim
    self.mlp_size = 4 * self.embed_dim
    self.net = nn.Sequential(
        nn.Linear(embed_dim, self.mlp_size),
        nn.GELU(),
        nn.Linear(self.mlp_size, embed_dim)
    ).to(device)

  def forward(self, x):
    return self.net(x)

class EncoderBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, head_size):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_size = head_size
    self.mha = MultiHeadedAttention(self.embed_dim, self.num_heads, self.head_size)
    self.ff = FeedForward(self.embed_dim)
    self.ln1 = nn.LayerNorm(self.embed_dim).to(device)
    self.ln2 = nn.LayerNorm(self.embed_dim).to(device)


  def forward(self, patch_embeds):
    patch_embeds = self.ln1(patch_embeds + self.mha(patch_embeds))
    patch_embeds = self.ln2(patch_embeds + self.ff(patch_embeds))
    return patch_embeds

n_blocks = 1     # Corresponds to base implementation

class TransformerEncoder(nn.Module):
  def __init__(self, embed_dim, num_heads, head_size, n_blocks, patch_size, img_channels, num_embed, pos_emb_seq_length):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_size = head_size
    self.n_blocks = n_blocks
    self.patch_size = patch_size
    self.img_channels = img_channels
    self.num_embed = num_embed
    self.pos_emb_seq_length = pos_emb_seq_length

    # Change this line to use self.pos_emb_seq_length instead of self.pos_emb_seq
    self.patch_embed = PatchEmbedding(self.patch_size, self.img_channels, self.num_embed, self.embed_dim, self.pos_emb_seq_length)
    self.blocks = nn.ModuleList([EncoderBlock(self.embed_dim, self.num_heads, self.head_size) for _ in range(self.n_blocks)])

  def forward(self, x):
    x = self.patch_embed(x)
    for block in self.blocks:
      x = block(x)
    return x

# Not very nicely written should probably inherit from the TransformerEncoder class
class VisionTransformer(nn.Module):
  def __init__(self, embed_dim, num_heads, head_size, n_blocks, patch_size, img_channels, num_embed, pos_emb_seq_length):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_size = head_size
    self.n_blocks = n_blocks
    self.patch_size = patch_size
    self.img_channels = img_channels
    self.num_embed = num_embed
    self.pos_emb_seq_length = pos_emb_seq_length

    self.transformer_encoder = TransformerEncoder(self.embed_dim, self.num_heads, self.head_size, self.n_blocks, self.patch_size, self.img_channels, self.num_embed, self.pos_emb_seq_length)
    # num_embed = class_size
    self.out_mlp = nn.Linear(self.embed_dim, self.num_embed).to(device)

  def forward(self, x):
    enc_out = self.transformer_encoder(x)
    class_token_out = enc_out[:, 0]  # Select the class token
    out = self.out_mlp(class_token_out)
    return out

class_size = 10
model = VisionTransformer(embed_dim, n_heads, head_size, n_blocks, patch_size, img_channels, class_size, pos_embed_seq_length)
optim = torch.optim.Adam(model.parameters())
loss = nn.CrossEntropyLoss()
epochs = 10

epochs_losses = []
for epoch in range(epochs):
  for i, (x, y) in enumerate(dataset):
    x, y = x.to(device), y.to(device)
    optim.zero_grad()
    out = model(x)
    l = loss(out, y)
    l.backward()
    optim.step()

  print(f"Epoch: {epoch+1} Loss: {l.item()}")
  epochs_losses.append(l.item())

plt.plot(epochs_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.show()

import numpy as np

test_dataset = DataLoader(torch.utils.data.TensorDataset(train_patches[:100], train_labels[:100]), batch_size=100)

# Switch the model to evaluation mode
model.eval()

all_predictions = []
all_ground_truths = []

with torch.no_grad():
    for x, y in test_dataset:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_ground_truths.extend(y.cpu().numpy())


all_predictions = np.array(all_predictions)
all_ground_truths = np.array(all_ground_truths)

accuracy = (all_predictions == all_ground_truths).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")

# Optionally: Display some predictions
for i in range(15):  # Display 5 predictions as examples
    print(f"Ground Truth: {all_ground_truths[i]}, Prediction: {all_predictions[i]}")

