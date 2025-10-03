import os
import json

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets.imagenet 
from PIL import Image
from torchvision import transforms

MODEL_ARCH = "resnet101"

{"resnet50" : "rn50_relabel_cutmix_IN21k_81.2.pth",
 "resnet101": "rn101_relabel_cutmix_81.6.pth",
 }
WEIGHT_FILE = "models/naver_relabel_imagenet/rn101_relabel_cutmix_81.6.pth"
EXPORT_PATH = "/home/juanmedrano_eng/repos/micro2025_compiler/micro2025_compiler_DeepX_M1_v1.60.1/InputDNNXModels/rn101_relabel_cutmix_81.6.onnx"
TEST_IMAGE = "/home/juanmedrano_eng/repos/micro2025_compiler/micro2025_compiler_DeepX_M1_v1.60.1/Calibration_Images_Classification/6.jpeg"

def remove_prefix_checkpoint(dictionary, prefix):
    keys = sorted(dictionary.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) + 1:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary

def load_checkpoint(weight_file, model):
    if os.path.isfile(weight_file):
        print(f"=> loading checkpoint '{weight_file}'")
        checkpoint = torch.load(weight_file)

        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        checkpoint = remove_prefix_checkpoint(checkpoint, 'module')
        model.load_state_dict(checkpoint)
        print(f"=> checkpoint loaded '{weight_file}'")
    else:
        raise Exception(f"=> no checkpoint found at '{weight_file}'")

def load_image_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Load image from disk and convert it into a 1x3x224x244 tensor."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 244)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
        ]
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image)

    # Batch 1
    return tensor.unsqueeze(0).to(device)

if __name__ == "__main__":
    print("Loading: ", WEIGHT_FILE)
    device = torch.device("cuda") # if torch.cuda.is_available() else "cpu")

    if os.path.isfile(WEIGHT_FILE):

        # Load initial network architecture
        if hasattr(torchvision.models, MODEL_ARCH):
            model = getattr(torchvision.models, MODEL_ARCH)()
        else:
            raise ValueError(
                f"Not supported model architecture {MODEL_ARCH}")

        load_checkpoint(WEIGHT_FILE, model)
        model.to(device)
        model.eval()

        if os.path.isfile(TEST_IMAGE):

            with open("imagenet_class_index.json") as f:
                idx_to_label = {int(k): v for k, v in json.load(f).items()}

            print(f"Testing on input image: {TEST_IMAGE}")
            image_tensor = load_image_tensor(TEST_IMAGE, device)
            with torch.no_grad():
                output = model(image_tensor)
            # print("First row of logits:", output)
            topk_indices = output.topk(5).indices.tolist()
            print("Topk indices: ", topk_indices)
            topk_names = [(idx_to_label[i][0], idx_to_label[i][1]) for i in topk_indices[0]]
            print("Maximum value pos:", output.argmax(1), " with class: ", topk_names)
        else:
            print("Test image not found. Skip inference.")
    else:
        print("Weight file not found.")
    
    c = 3
    h = 224
    w = 224
    dummy_input = torch.rand(1, c, h, w, device=device)

    print("4. Exporting to onnx: ...")
    torch.onnx.export(
        model,
        dummy_input,
        EXPORT_PATH,
        export_params=True,
        opset_version=12,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        dynamic_axes=None
    )
    print(" ONNX export finished!")
