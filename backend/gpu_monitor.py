"""
GPU monitoring - simple functions, no classes.
"""
import pynvml
import time
import threading
from typing import Dict, Callable, Optional

def get_gpu_stats() -> Dict:
    """Get GPU stats as simple dict."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Memory info
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        # Temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        # Power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
        except:
            power = 0
        
        return {
            'available': True,
            'memory_used': mem.used // (1024**2),  # MB
            'memory_total': mem.total // (1024**2),
            'memory_percent': (mem.used / mem.total) * 100,
            'utilization': util.gpu,
            'temperature': temp,
            'power': power
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e),
            'memory_used': 0,
            'memory_total': 0,
            'memory_percent': 0,
            'utilization': 0,
            'temperature': 0,
            'power': 0
        }

def monitor_gpu(callback: Callable, interval: float = 2.0, storage_list: Optional[list] = None):
    """Monitor GPU in background thread."""
    def loop():
        while True:
            try:
                stats = get_gpu_stats()
                if storage_list is not None:
                    storage_list.append(stats)
                    # Keep only last 100 readings
                    if len(storage_list) > 100:
                        storage_list.pop(0)
                callback(stats)
            except Exception as e:
                print(f"GPU monitoring error: {e}")
            time.sleep(interval)
    
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread
