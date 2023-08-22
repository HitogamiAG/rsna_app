import torch

from model import CustomModel

model = CustomModel().cuda()
weights = torch.load('weights/best_model.ckpt')['state_dict']
model.load_state_dict({k.replace('model.', '') : v for k, v in weights.items()})

model = torch.nn.Sequential(
    model,
    torch.nn.Sigmoid()
)

input_sample = torch.zeros(1, 15, 3, 224, 224).cuda()

torch.onnx.export(
    model,
    input_sample,
    'weights/best_model.onnx',
    export_params=True,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input' : {0 : 'batch_size'},
                  'output' : {0 : 'batch_size'}}
)