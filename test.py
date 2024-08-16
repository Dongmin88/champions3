import torch

def check_cuda_availability():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of available GPU(s): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")
        print(f"PyTorch version: {torch.__version__}")

if __name__ == "__main__":
    check_cuda_availability()