from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from typing import List
import clip
import torch
from torch import nn

class TextClassifier(nn.Module):
    def __init__(self, class_names, hidden_dim, temperature=0.07):
        super().__init__()
        model, _ = clip.load("ViT-L/14", device="cpu", download_root='/home/phd_li/.cache/clip')
                
        with torch.no_grad():
            print(f"Preparing text tokens for class names: {class_names}")
            text = clip.tokenize(class_names).to("cpu")
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        del model
        self.text_class_tokens = nn.Parameter(text_features, requires_grad=False)
        self.no_object_token = nn.Parameter(torch.zeros(1, 768), requires_grad=True) ## Text embedding from CLIP-L/14 is [1, 768]
        self.class_embed_proj = nn.Linear(hidden_dim, 768) if hidden_dim != 768 else nn.Identity()

        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        x = self.class_embed_proj(x)
        # print(x.shape)
        x = x / x.norm(dim=-1, keepdim=True)
        no_object_token = self.no_object_token / (self.no_object_token.norm(dim=-1, keepdim=True) + 1e-8)
        logits = torch.matmul(x, torch.cat((self.text_class_tokens, no_object_token), dim=0).T)

        # print(self.temperature)
        logits = logits / self.temperature
        # print(f"no_object_token: {no_object_token}")
        # print(f"text_class_tokens: {self.text_class_tokens}")
        return logits

class OpenMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    def __init__(self, class_names: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mask_classification:
            self.class_embed = TextClassifier(class_names=class_names, hidden_dim=self.hidden_dim)

    