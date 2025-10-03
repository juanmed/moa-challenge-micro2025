import sys

import matplotlib.pyplot as plt
import PIL
from PIL import Image
import json

import torch
from torch import nn
import torchvision.transforms as T

from timm import create_model
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

# print( transformers.utils.import_utils.get_torch_version() )
#     print( transformers.utils.import_utils.is_torch_greater_or_equal("2.6") )

# Define transforms for test
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
NORMALIZE_STD = IMAGENET_DEFAULT_STD
SIZE = 224
TEST_IMAGE = "/home/juanmedrano_eng/repos/micro2025_compiler/micro2025_compiler_DeepX_M1_v1.60.1/Calibration_Images_Classification/6.jpeg"
#MODEL_NAME = "convnext_xlarge_in22k"
MODEL_NAME = "efficientnet_b6"
EXPORT_PATH = f"/home/juanmedrano_eng/repos/micro2025_compiler/micro2025_compiler_DeepX_M1_v1.60.1/InputONNXModels/{MODEL_NAME}.onnx"

class NetTensorFlowWrapper(nn.Module):
    def __init__(self, main_module: nn.Module):
        super(NetTensorFlowWrapper, self).__init__()
        self.main_module = main_module
        
    def forward(self, x):
        x = torch.transpose(x, 1, 3)
        return self.main_module(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
# create a ConvNeXt model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py
# model = create_model(MODEL_NAME, pretrained=True).to(device)

preprocessor = EfficientNetImageProcessor.from_pretrained("google/efficientnet-b6")
model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b6")
model.to(device)
image = PIL.Image.open(TEST_IMAGE)
inputs = preprocessor(image, return_tensors="pt")
inputs.to(device)
model.eval()


with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label]),



# Here we resize smaller edge to 256, no center cropping
# transforms = [
#               T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
#               T.ToTensor(),
#               T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
#               ]
# transforms = T.Compose(transforms)

# img = PIL.Image.open(TEST_IMAGE)
# img_tensor = transforms(img).unsqueeze(0).to(device)    
# print(img_tensor.size())

# output = torch.softmax(model(img_tensor), dim=1)
# top5 = torch.topk(output, k=5)
# top5_prob = top5.values[0]
# top5_indices = top5.indices[0]

# imagenet_labels = json.load(open('label_to_words.json'))
# for i in range(5):
#     labels = imagenet_labels[str(int(top5_indices[i]))]
#     prob = "{:.2f}%".format(float(top5_prob[i])*100)
#     print(labels, prob)

# c = 3
# h = SIZE
# w = SIZE
# dummy_input = torch.rand(1, c, h, w, device=device)




# print("4. Exporting to onnx: ...")
# torch.onnx.export(
#     model,
#     dummy_input,
#     EXPORT_PATH,
#     export_params=True,
#     opset_version=12,
#     training=torch.onnx.TrainingMode.EVAL,
#     do_constant_folding=True,
#     dynamic_axes=None
# )
# print(f"ONNX export finished!: {EXPORT_PATH}")

#plt.imshow(img)
#plt.show()