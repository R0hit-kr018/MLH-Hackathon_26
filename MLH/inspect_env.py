import sys
import shutil
import urllib.request
import json
import ctypes
import os

print(f"PYTHON_VERSION: {sys.version}")

# Check Disk space
total, used, free = shutil.disk_usage("C:\\")
print(f"DISK_TOTAL_GB: {total / (1024**3):.2f}")
print(f"DISK_FREE_GB: {free / (1024**3):.2f}")

# Check RAM on Windows
try:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    total_ram = stat.ullTotalPhys / (1024**3)
    avail_ram = stat.ullAvailPhys / (1024**3)
    print(f"RAM_TOTAL_GB: {total_ram:.2f}")
    print(f"RAM_AVAIL_GB: {avail_ram:.2f}")
except Exception as e:
    print(f"RAM_CHECK_ERROR: {e}")

# Check Ollama executable
ollama_path = shutil.which("ollama")
print(f"OLLAMA_EXE: {ollama_path}")

# Check Ollama Server
try:
    req = urllib.request.Request("http://127.0.0.1:11434/api/tags", headers={"User-Agent": "OpenResearchAI"})
    with urllib.request.urlopen(req, timeout=2) as response:
        data = json.loads(response.read().decode())
        models = [m.get("name") for m in data.get("models", [])]
        print("OLLAMA_RUNNING: True")
        print(f"OLLAMA_MODELS: {models}")
except Exception as e:
    print(f"OLLAMA_RUNNING: False ({e})")
