from lora_linear import LoRALinear

layer=LoRALinear(
    in_features=128,
    out_features=64,
    r=16
)
total=sum(
    p.numel()
    for p in layer.parameters()
)

trainable=sum(
    p.numel()
    for p in layer.parameters()
    if p.requires_grad
)
print(f"Total Parameters: {total}")
print(f"Trainable Parameters: {trainable}")
print(
    f"Trainable %: {100 * trainable / total:.2f}%"
)