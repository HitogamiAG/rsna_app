import torch
import timm

class CustomModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s',
            in_chans = 3,
            num_classes = 1,
            features_only = False,
            drop_rate = 0,
            drop_path_rate = 0,
            pretrained = True
        )
        self.backbone.classifier = torch.nn.Identity()
        
        self.lstm = torch.nn.LSTM(self.backbone.conv_head.out_channels, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(256, 1),
        )
        
    def forward(self, x):
        batch_size, num_slices = x.shape[0:2]
        x = x.reshape(batch_size * num_slices, *x.shape[2:])
        out = self.backbone(x)
        out = out.reshape(batch_size, num_slices, -1)
        out, _ = self.lstm(out)
        out = out.reshape(batch_size * num_slices, -1)
        out = self.classifier(out)
        out = out.reshape(batch_size, -1)
        return out