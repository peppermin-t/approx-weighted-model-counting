import torch

def check_cuda():
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # set device
        print("CUDA device name:", torch.cuda.get_device_name(device))
        print("Memory allocated:", torch.cuda.memory_allocated(device) / 1024**2, "MB")
        print("Max memory allocated:", torch.cuda.max_memory_allocated(device) / 1024**2, "MB")
        print("Memory Cached:", torch.cuda.memory_reserved(device) / 1024**2, "MB")
        torch.cuda.empty_cache()
        print("Cache cleared")
        
        # force initialising CUDA context
        _ = torch.tensor([1.0]).cuda()
        print("device: ", _.device)
        print("CUDA context initialized.")

        x = torch.rand(10, 10, device=device)  # move tensor
        print("Reserved memory after allocation:", torch.cuda.memory_reserved(device) / 1024**2, "MB")
        print("Allocated memory after allocation:", torch.cuda.memory_allocated(device) / 1024**2, "MB")
        
        # move more tensor
        y = torch.rand(1000, 1000, device=device)
        print("Reserved memory after large allocation:", torch.cuda.memory_reserved(device) / 1024**2, "MB")
        print("Allocated memory after large allocation:", torch.cuda.memory_allocated(device) / 1024**2, "MB")

if __name__ == "__main__":
    print("Start testing...")
    check_cuda()
