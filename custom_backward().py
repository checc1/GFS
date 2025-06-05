from symtable import Function

import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=False,
                                                                      transform=torchvision.transforms.Compose([
                                                                          torchvision.transforms.ToTensor(),
                                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=32, shuffle=True)



class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def _get_flattened_size(self):
        single_input = torch.randn(1, 1, 28, 28)
        out = self.extractor(single_input)
        return out.view(1, -1).size(1)

    def forward(self, x) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        features = self.feature_layer(x)
        x = self.predictor(features)
        return x


class CustomizedModel(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        model = CnnModel()
        output = model.block2(input)  ### I wanna save only the output of the last block
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        calculated_grad = torch.autograd.grad(input, inputs=input, grad_outputs=grad_input, retain_graph=True)  ## :TODO qui devo modificare

        #grad_input.requires_grad = True

        return calculated_grad[0]

    def apply(cls, input: torch.Tensor):
        return cls.forward(input)



model = CnnModel()
input_img = next(iter(train_loader))[0]
model = CustomizedModel.apply(input_img)
output = model.forward(input_img)
output.backward()
print("Output shape:", output.shape)
