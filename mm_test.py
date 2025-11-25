import torch
from mmpretrain import get_model

model = get_model('convnext-v2-large_32xb32_in1k-384px', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))