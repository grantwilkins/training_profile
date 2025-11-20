import torch

print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    major_cc, minor_cc = torch.cuda.get_device_capability(0)
    print(f"compute_capability sm_{major_cc}{minor_cc}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"total_memory {total_mem:.2f} GB")
