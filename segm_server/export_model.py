import torch

model = torch.load('weights/best_model.pth')
input_sample = torch.randn((1, 3, 512 , 512)).to('cuda')
torch.onnx.export(
    model.module,
    input_sample,
    'weights/best_model.onnx',
    export_params=True,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input' : {0 : 'batch_size'},
                  'output' : {0 : 'batch_size'}}
)
