import torch
from lora_linear import LoRALinear

x=torch.randn(4,128)

layer=LoRALinear(
    in_features=128,
    out_features=64
)
output=layer(x)
print(output.shape)
print((layer.B == 0).all())
base = x @ layer.weight.T
output = layer(x)

print(torch.allclose(
    base,
    output
))
for name, param in layer.named_parameters():
    print(
        name,
        param.shape,
        param.requires_grad
    )