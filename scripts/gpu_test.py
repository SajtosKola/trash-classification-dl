import torch

print("PyTorch verzió:", torch.__version__)
print("CUDA elérhető:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU neve:", torch.cuda.get_device_name(0))

    x = torch.rand(3, 3, device="cuda")
    print("\nGPU-n létrehozott tensor:")
    print(x)

    print("\nTensor eszköze:", x.device)
else:
    print("A PyTorch jelenleg nem tudja használni a GPU-t.")