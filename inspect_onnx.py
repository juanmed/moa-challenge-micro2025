import torch
import os
import onnx


INPUT_ONNX = "/home/juanmedrano_eng/repos/micro2025_compiler/micro2025_compiler_DeepX_M1_v1.60.1/InputONNXModels/efficientnet_b6.onnx"
model = onnx.load(INPUT_ONNX)
output =[node.name for node in model.graph.output]

input_all = [node.name for node in model.graph.input]
input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
print('Outputs: ', output)

print("Operations: ")
ops = set()
for node in model.graph.node:
  ops.add(node.op_type)
print(ops)