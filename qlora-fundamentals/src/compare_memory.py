import torch

print(
    f"Allocated Memory: "
    f"{torch.cuda.memory_allocated()/1024**2:.2f} MB"
)

print(
    f"Reserved Memory: "
    f"{torch.cuda.memory_reserved()/1024**2:.2f} MB"
)