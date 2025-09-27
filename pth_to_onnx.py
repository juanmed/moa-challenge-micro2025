import os
import json

import torch
import torchvision.models as models
import torchvision.datasets.imagenet 
from PIL import Image
from torchvision import transforms

{"resnet50" : "rn50_relabel_cutmix_IN21k_81.2.pth",
 "resnet101": "rn101_relabel_cutmix_81.6.pth",
 }
WEIGHT_FILE = "models/naver_relabel_imagenet/rn50_relabel_cutmix_IN21k_81.2.pth"
EXPORT_PATH = "/home/juanmedrano_eng/repos/micro2025_compiler/micro2025_compiler_DeepX_M1_v1.60.1/OutputDNNXModels/rn50_relabel_cutmix_IN21k_81.2.onnx"
TEST_IMAGE = "/home/juanmedrano_eng/repos/micro2025_compiler/micro2025_compiler_DeepX_M1_v1.60.1/Calibration_Images_Classification/4.jpeg"


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


        # Step 1: Initialize the model architecture
        print("1. Loading Resnet50...")
        model = models.resnet50(weights=None,)  # no pretrained weights
        print("  ...model loaded")

        print("2. Loading state dict if any:")
        # Step 2: Load the saved state dictionary
        state_dict = torch.load(WEIGHT_FILE, map_location=device)
        # Some checkpoints save a dict with key 'state_dict', so handle that case
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            print(" ... loaded state dict")
        print(" ... finished loading state dict")

        # Step 3: Load weights into the model
        print("3. Loading weights into model...")
        model.load_state_dict(state_dict, strict=True)  # strict=False if keys don’t match perfectly
        model.eval()
        model.to(device)
        print("Model loaded successfully!")

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
