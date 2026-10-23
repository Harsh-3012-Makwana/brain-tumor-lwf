import sys
import time

print("Step 1: Importing torch...", flush=True)
import torch
print(f"Torch imported: version={torch.__version__}", flush=True)

print("Step 2: Checking cuda available...", flush=True)
cuda_avail = torch.cuda.is_available()
print(f"CUDA available: {cuda_avail}", flush=True)

if cuda_avail:
    print(f"Device name: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}", flush=True)
    print(f"CUDA arch list: {torch.cuda.get_arch_list()}", flush=True)
    
    print("Step 3: Allocating tensor on CUDA...", flush=True)
    t0 = time.time()
    x = torch.zeros(1, device="cuda")
    print(f"Tensor allocated in {time.time() - t0:.2f}s", flush=True)
    
    print("Step 4: Running Conv2d on CUDA...", flush=True)
    import torch.nn as nn
    conv = nn.Conv2d(3, 16, 3, padding=1).cuda()
    inp = torch.randn(2, 3, 150, 150, device="cuda", requires_grad=True)
    out = conv(inp)
    loss = out.sum()
    loss.backward()
    print(f"Conv forward+backward SUCCESS! Loss = {loss.item():.4f}", flush=True)
else:
    print("CUDA not available.")
