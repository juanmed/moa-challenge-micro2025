import torch

device = torch.device("cuda")
model = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
model.to(device)
model.eval()
EXPORT_PATH = "/home/juanmedrano_eng/repos/micro2025_compiler/mae_in1k_ft_in1k.onnx"


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
    opset_version=14,
    training=torch.onnx.TrainingMode.EVAL,
    do_constant_folding=True,
    dynamic_axes=None
)
print(" ONNX export finished!")