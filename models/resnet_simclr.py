import torch.nn as nn
import torchvision.models as models

from models.vos_net import VOSNet


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}
        self.net = VOSNet('resnet50')

        # projection MLP
        self.l1 = nn.Linear(256 * 32 * 32, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.net(x)
        x = self.l1(h.view(-1, 256 * 32 * 32))
        return h, x
