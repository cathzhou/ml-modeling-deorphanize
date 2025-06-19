# %%
import torch
import math
import pandas as pd
import toolz
import seaborn as sns
import matplotlib.pyplot as plt
import datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
#import ggplot
import time
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# %%
def my_scaled_dot_product_attention(query, key=None, value=None):
    key = key if key is not None else query
    value = value if value is not None else query
    # query and key must have same embedding dimension
    assert query.size(-1) == key.size(-1)

    dk = key.size(-1) # embed dimension of key
    # query, key, value = (bs, seq_len, embed_dim)
    
    # compute dot-product to obtain pairwise "similarity" and scale it
    qk = query @ key.transpose(-1, -2) / dk**0.5
    
    # apply softmax
    # attn_weights = (bs, seq_len, seq_len)
    attn_weights = torch.softmax(qk, dim=-1)

    # compute weighted sum of value vectors
    # attn = (bs, seq_len, embed_dim)
    attn = attn_weights @ value
    return attn, attn_weights

# X = torch.normal(mean=0, std=1, size=(2, 3, 6))
# torch_attended = torch.nn.functional.scaled_dot_product_attention(X, X, X)
# attended, attn_weights = my_scaled_dot_product_attention(X, X, X)
# assert torch.allclose(torch_attended, attended) == True

# batch_size = 3
# A = matrix(batch_size, 10, 256)

# output = []
# for batch_idx in range(batch_size):
#     pairwise_dot_product = A[batch_idx] @ A[batch_idx].transpose(-1, -2)
#     output.append(pairwise_dot_product)

# # Output has shape (batch_size, 10, 10)
# return output

# batch_size = 3
# n_heads = 2
# A = matrix(batch_size, n_heads, 10, 256)
# output = []

# for batch_idx in range(batch_size):
#     output_per_head = []
#     for head_idx in range(n_heads):
#         pairwise_dot_product = A[batch_idx][head_idx] @ A[batch_idx][head_idx].transpose(-1, -2)
#         output_per_head.append(pairwise_dot_product)
#     output.append(output_per_head)

# # Output has shape (batch_size, n_heads, 10, 10)
# return output

class AttentionBlock(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias=False):
        super().__init__()
        # Linear layers to project Query, Key and Value 
        self.W_q = torch.nn.Linear(input_dim, output_dim, bias=bias)
        self.W_k = torch.nn.Linear(input_dim, output_dim, bias=bias)
        self.W_v = torch.nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, query, key, value):
        # project Q, K, V
        q_logits = self.W_q(query)
        k_logits = self.W_k(key)
        v_logits = self.W_v(value)

        # apply scaled dot product attention on projected values
        attn, weights = my_scaled_dot_product_attention(q_logits, k_logits, v_logits)
        return attn, weights

class MyMultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, projection_bias=False):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        head_embed_dim = self.embed_dim // n_heads
        # for each head, create an attention block
        self.head_blocks = torch.nn.ModuleList([AttentionBlock(input_dim=embed_dim, output_dim=head_embed_dim, bias=projection_bias) for i in range(self.n_heads)])
        # final projection of MHA
        self.projection = torch.nn.Linear(embed_dim, embed_dim, bias=projection_bias)


    def forward(self, query, key, value):
        # these lists are to store output of each head
        attns_list = []
        attn_weights_list = []

        # for every head pass the original query, key, value
        for head in self.head_blocks:
            attn, attn_weights = head(query, key, value)
            attns_list.append(attn)
            attn_weights_list.append(attn_weights)

        # concatenate attention outputs and take average of attention weights
        attns, attn_weights = torch.cat(attns_list, dim=2), torch.stack(attn_weights_list).mean(dim=0)
        # shape: (bs, seq_len, embed_dim), attn_weights: (bs, seq_len, seq_len)
        return self.projection(attns), attn_weights
    
original_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# %%

news_ds = datasets.load_dataset("SetFit/bbc-news", split="train")
# train a new tokenizer with limited vocab size for demo
tokenizer = original_tokenizer.train_new_from_iterator(news_ds['text'], vocab_size=1000)

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True)

ds = news_ds.map(tokenize, batched=True).select_columns(['label', 'input_ids', 'text']).train_test_split()

class_id_to_class = {
    0: "tech",
    1: "business",
    2: "sports",
    3: "entertainment",
    4: "politics",
}
num_classes = len(class_id_to_class)
# %%
class TextClassifier(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, mha: torch.nn.Module):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.mha = mha
        self.fc1 = torch.nn.Linear(in_features=embed_dim, out_features=128)
        self.relu = torch.nn.ReLU()
        self.final = torch.nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        # inputs: (bs, seq_len)
        # embeddings: (bs, seq_len, embed_dim)
        embeddings = self.get_embeddings(input_ids)
        attn, attn_weights = self.get_attention(embeddings, embeddings, embeddings)
        
        # take the first token's embeddings i.e. embeddings of CLS token
        # cls_token_embeddings: (bs, embed_dim)
        cls_token_embeddings = attn[:, 0, :] 
        return self.final(self.relu(self.fc1(cls_token_embeddings)))
    
    def get_embeddings(self, input_ids):
        return self.embedding(input_ids)
    
    def get_attention(self, query, key, value):
        attn, attn_weights = self.mha(query, key, value)
        return attn, attn_weights

n_heads = 8
embed_dim = 64
vocab_size = tokenizer.vocab_size
torch_mha = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
my_mha = MyMultiheadAttention(embed_dim=embed_dim, n_heads=n_heads, projection_bias=True)
torch_classifier = TextClassifier(vocab_size=tokenizer.vocab_size, embed_dim=embed_dim, num_classes=num_classes, mha=torch_mha)
my_classifier = TextClassifier(vocab_size=tokenizer.vocab_size, embed_dim=embed_dim, num_classes=num_classes, mha=my_mha)
# %%
def collate_fn(batch):
    labels = []
    input_ids = []
    for row in batch:
        labels.append(row['label'])
        input_ids.append(torch.LongTensor(row['input_ids']))

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    input_ids = torch.Tensor(input_ids)
    return {"labels": labels, "input_ids": input_ids}

train_dl = test_dl = DataLoader(ds['train'], shuffle=True, batch_size=32, collate_fn=collate_fn)
test_dl = DataLoader(ds['test'], shuffle=False, batch_size=32, collate_fn=collate_fn)
# %%
def train(model: torch.nn.Module, train_dl, val_dl, epochs=10) -> list[tuple[float, float]]:
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss = 0.0
        model.train()
        for batch in train_dl:
            optim.zero_grad()
            logits = model(**batch)
            loss = loss_fn(logits, batch['labels'])
            loss.backward()
            optim.step()
            train_loss += loss.item() * batch['labels'].size(0)

        train_loss /= len(train_dl.dataset)

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for batch in val_dl:
                logits = model(**batch)
                loss = loss_fn(logits, batch['labels'])
                val_loss += loss.item() * batch['labels'].size(0)
                val_accuracy += (logits.argmax(dim=1) == batch['labels']).sum()

        val_loss /= len(val_dl.dataset)
        val_accuracy /= len(val_dl.dataset)
        log_steps = max(1, int(0.2 * epochs))

        losses.append((train_loss, val_loss))
        if epoch % log_steps == 0 or epoch == epochs - 1:
            epoch_duartion = time.time() - epoch_start
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}. Epoch Duration: {epoch_duartion:.1f} seconds')

    train_duration = time.time() - train_start
    print(f"Training finished. Took {train_duration:.1f} seconds")

    return losses

def get_model_param_count(model):
    return sum(t.numel() for t in model.parameters())

print(f"My classifier params: {get_model_param_count(my_classifier):,}")
print(f"Torch classifier params: {get_model_param_count(torch_classifier):,}")

# My classifier params: 89,605
# Torch classifier params: 89,605

torch_losses = train(torch_classifier, train_dl, test_dl, epochs=10)
my_losses = train(my_classifier, train_dl, test_dl, epochs=10)
# %%
def predict(texts, model, bs=32):
    output_dfs = []
    for batch in toolz.partition_all(bs, texts):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            class_probs = torch.softmax(model(**inputs), dim=1).numpy()
            pred_classes = class_probs.argmax(axis=1)
            col_names = [f"class_{i}_prob" for i in range(class_probs.shape[-1])]
            df = pd.DataFrame(class_probs, columns=col_names)
            df['pred_class'] = pred_classes
            df['pred_class_name'] = df['pred_class'].map(class_id_to_class)
            output_dfs.append(df)

    return pd.concat(output_dfs)

my_preds_df = predict(ds['test']['text'], my_classifier)
my_preds_df['model'] = 'My Model'
my_preds_df['actual_class'] = ds['test']['label']
torch_preds_df = predict(ds['test']['text'], torch_classifier)
torch_preds_df['model'] = 'Torch Model'
torch_preds_df['actual_class'] = ds['test']['label']

print("My Classifier")
print(classification_report(my_preds_df['actual_class'], my_preds_df['pred_class']))

print("Torch Classifier")
print(classification_report(torch_preds_df['actual_class'], torch_preds_df['pred_class']))
# %%
def get_losses_as_df(losses_name_pairs: list[tuple[str, tuple[float, float]]]):
    dfs = []
    for model_name, losses in losses_name_pairs:
        df = pd.DataFrame(losses, columns=['train_loss', 'test_loss']).reset_index().rename(columns={"index": "epoch"})
        df['model'] = model_name
        dfs.append(df)
    return pd.concat(dfs)

def plot_losses(loss_df):
    df = loss_df.melt(id_vars=['model', 'epoch'], var_name='metric')
    return ggplot(df, aes('epoch', 'value', color='metric')) + geom_line() + geom_point(size=1.5) + facet_grid('model') + labs(title="Train and Validation loss")
# %%
def get_losses_as_df(losses_name_pairs: list[tuple[str, tuple[float, float]]]):
    dfs = []
    for model_name, losses in losses_name_pairs:
        df = pd.DataFrame(losses, columns=['train_loss', 'test_loss']).reset_index().rename(columns={"index": "epoch"})
        df['model'] = model_name
        dfs.append(df)
    return pd.concat(dfs)

def plot_losses(loss_df):
    # Get unique models
    models = loss_df['model'].unique()
    n_models = len(models)
    
    # Create subplots
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 5*n_models))
    if n_models == 1:
        axes = [axes]
    
    # Plot for each model
    for ax, model in zip(axes, models):
        model_data = loss_df[loss_df['model'] == model]
        
        # Plot train and test losses
        ax.plot(model_data['epoch'], model_data['train_loss'], 'b-', label='Train Loss', marker='o', markersize=3)
        ax.plot(model_data['epoch'], model_data['test_loss'], 'r-', label='Validation Loss', marker='o', markersize=3)
        
        # Customize subplot
        ax.set_title(f'Model: {model}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.suptitle('Train and Validation Losses', y=1.02, fontsize=14)
    
    return fig

plot_losses(get_losses_as_df([("My", my_losses), ("Torch", torch_losses)]))
# %%
class MyEfficientMultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, projection_bias=False):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_embed_dim = self.embed_dim // n_heads
        self.W_q = torch.nn.Linear(embed_dim, embed_dim, bias=projection_bias)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim, bias=projection_bias)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim, bias=projection_bias)
        self.projection = torch.nn.Linear(embed_dim, embed_dim, bias=projection_bias)

    def forward(self, query, key, value):
        # shape of query = (bs, seq_len, embed_dim)
        batch_size = query.size(0)

        # linear projection of query, key and value
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # reshape the projected query, key, value
        # to (bs, n_heads, seq_len, head_embed_dim)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # do scaled dot product attention
        # attn.shape = (bs, n_heads, seq_len, head_embed_dim)
        # attn_weights.shape (bs, n_heads, seq_len, seq_len)
        attn, attn_weights = my_scaled_dot_product_attention(q, k, v)
        # swap the n_heads and seq_len so that we have
        # (bs, seq_len, n_heads, head_embed_dim)
        # call .contiguous() so that view function will work later
        attn = attn.transpose(1, 2).contiguous()
        # "combine" (n_heads, head_embed_dim) matrix as a single "embed_dim" vector
        attn = attn.view(batch_size, -1, self.embed_dim)

        output = self.projection(attn)
        return output, attn_weights.mean(dim=1)

    def split_heads(self, x):
        # x.shape = (bs, seq_len, embed_dim)
        batch_size = x.size(0)
        # first split the embed_dim into (n_heads, head_embed_dim)
        temp =  x.view(batch_size, -1, self.n_heads, self.head_embed_dim)
        # now we swap seq_len and n_heads dimension
         # output shape = (bs, n_heads, seq_len, head_embed_dim)
        return temp.transpose(1, 2)

my_efficient_mha = MyEfficientMultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads, projection_bias=True)
my_efficient_classifier = TextClassifier(vocab_size=tokenizer.vocab_size, embed_dim=embed_dim, num_classes=num_classes, mha=my_efficient_mha)
my_efficient_losses = train(my_efficient_classifier, train_dl, test_dl, epochs=10)
# %%
from torch.nn import Module
# %%
class PositionalEncoding(torch.nn.Module):
    # source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html#Positional-encoding
    def __init__(self, embed_dim, max_len=256):
        super().__init__()
        # create a matrix of [seq_len, hidden_dim] representing positional encoding for each token in sequence
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
class TextClassifierWithPositionalEncoding(TextClassifier):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, mha: Module, max_len: int=256):
        super().__init__(vocab_size, embed_dim, num_classes, mha)
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

    def get_embeddings(self, input_ids):
        embeddings = super().get_embeddings(input_ids)
        return self.positional_encoding(embeddings)
    
my_efficient_mha2 = MyEfficientMultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads, projection_bias=True)
my_efficient_classifier_with_pe = TextClassifierWithPositionalEncoding(vocab_size=tokenizer.vocab_size, embed_dim=embed_dim, num_classes=num_classes, mha=my_efficient_mha2, max_len=tokenizer.model_max_length)
my_efficient_losses_with_pe = train(my_efficient_classifier_with_pe, train_dl, test_dl, epochs=10)

def visualize_attention_weights(model, text, tokenizer, ax):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        embeddings = model.get_embeddings(inputs['input_ids'])
        attn_weights = model.get_attention(embeddings, embeddings, embeddings)[1].squeeze()

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
    df = pd.DataFrame(attn_weights, columns=tokens, index=tokens)
    return sns.heatmap(df, annot=True, ax=ax)
    
# "Can you can that?" -> First can is a verb, second can is a verb: to preserve something in a Can
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)
for i, (model_name, model) in enumerate([("without PE: torch MHA", torch_classifier), ("without PE: My MHA", my_classifier), ("with PE: My MHA", my_efficient_classifier_with_pe)]):
    axes[i] = visualize_attention_weights(model, text="can you can that", tokenizer=tokenizer, ax=axes[i])
    axes[i].set_title(model_name)
    axes[i].tick_params(labeltop=True, bottom=False, left=False)
# %%
