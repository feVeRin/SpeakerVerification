import pynvml

def check_vram(device=0) -> None:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f">> GPU:{device} 총 VRAM: {meminfo.total / 1024**2:.2f} MB")
        print(f">> GPU:{device} 사용 중인 VRAM: {meminfo.used / 1024**2:.2f} MB")
        print(f">> GPU:{device} 남은 VRAM: {meminfo.free / 1024**2:.2f} MB")
    except pynvml.NVMLError as e:
        print(f"NVML 에러: {e}")
    finally:
        pynvml.nvmlShutdown()