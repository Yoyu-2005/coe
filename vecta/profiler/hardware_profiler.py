"""
VECTA Hardware & Environment Profiler.
Monitors CPU threads, system memory headroom, and threadpool states.
"""

import os
import psutil
from typing import Dict, Any


class HardwareProfiler:
    """Detects host computational capabilities and available resources."""
    
    @staticmethod
    def get_hardware_profile() -> Dict[str, Any]:
        """Capture host CPU, memory, and runtime thread allocations."""
        logical_cpus = os.cpu_count() or 4
        physical_cpus = psutil.cpu_count(logical=False) or (logical_cpus // 2)
        
        vm = psutil.virtual_memory()
        total_ram_gb = round(vm.total / (1024 ** 3), 2)
        available_ram_gb = round(vm.available / (1024 ** 3), 2)
        ram_percent_used = vm.percent
        
        return {
            "logical_cpus": logical_cpus,
            "physical_cpus": physical_cpus,
            "total_ram_gb": total_ram_gb,
            "available_ram_gb": available_ram_gb,
            "ram_percent_used": ram_percent_used,
            "platform": os.name,
        }

    @staticmethod
    def current_rss_memory_mb() -> float:
        """Get current process Resident Set Size (RSS) in megabytes."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
