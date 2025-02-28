import torch
print(f'Torch version: {torch.__version__}')
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
    print('Supports apple silicon')
else:
    print ("MPS device not found.")

