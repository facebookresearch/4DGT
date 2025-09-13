# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

#!/usr/bin/env python3
"""
Simple GPU monitoring utility for profiling inference.
"""

import torch
import time
import threading
from typing import List, Tuple
from tlod.easyvolcap.utils.console_utils import logger


class GPUMonitor:
    """Monitor GPU utilization and memory during inference."""

    def __init__(self, interval: float = 0.5):
        """
        Initialize GPU monitor.

        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.thread = None
        self.gpu_utils = []
        self.memory_utils = []

        # Check if CUDA is available
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.pynvml_available = True
            except:
                self.pynvml_available = False
                logger.warning("pynvml not available. Install with: pip install pynvml")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            if self.pynvml_available:
                try:
                    import pynvml
                    # Get GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    self.gpu_utils.append(util.gpu)

                    # Get memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    mem_percent = (mem_info.used / mem_info.total) * 100
                    self.memory_utils.append(mem_percent)
                except:
                    pass
            elif self.cuda_available:
                # Fallback to PyTorch memory stats
                try:
                    mem_allocated = torch.cuda.memory_allocated(0)
                    mem_reserved = torch.cuda.memory_reserved(0)
                    max_mem = torch.cuda.max_memory_allocated(0)

                    # Estimate utilization (not as accurate as pynvml)
                    self.memory_utils.append((mem_allocated / (1024**3)))  # GB
                except:
                    pass

            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.gpu_utils = []
            self.memory_utils = []
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            logger.info("GPU monitoring started")

    def stop(self) -> Tuple[List[float], List[float]]:
        """
        Stop monitoring and return statistics.

        Returns:
            Tuple of (gpu_utilization_list, memory_utilization_list)
        """
        if self.monitoring:
            self.monitoring = False
            if self.thread:
                self.thread.join(timeout=1)
            logger.info("GPU monitoring stopped")
        return self.gpu_utils, self.memory_utils

    def print_summary(self):
        """Print monitoring summary."""
        if not self.gpu_utils and not self.memory_utils:
            logger.info("No GPU monitoring data available")
            return

        logger.info("\n" + "=" * 60)
        logger.info("GPU Monitoring Summary")
        logger.info("=" * 60)

        if self.gpu_utils:
            avg_gpu = sum(self.gpu_utils) / len(self.gpu_utils)
            max_gpu = max(self.gpu_utils)
            min_gpu = min(self.gpu_utils)
            logger.info(f"GPU Utilization:")
            logger.info(f"  Average: {avg_gpu:.1f}%")
            logger.info(f"  Max: {max_gpu:.1f}%")
            logger.info(f"  Min: {min_gpu:.1f}%")

        if self.memory_utils:
            if self.pynvml_available:
                avg_mem = sum(self.memory_utils) / len(self.memory_utils)
                max_mem = max(self.memory_utils)
                min_mem = min(self.memory_utils)
                logger.info(f"Memory Utilization:")
                logger.info(f"  Average: {avg_mem:.1f}%")
                logger.info(f"  Max: {max_mem:.1f}%")
                logger.info(f"  Min: {min_mem:.1f}%")
            else:
                avg_mem = sum(self.memory_utils) / len(self.memory_utils)
                max_mem = max(self.memory_utils)
                logger.info(f"Memory Usage (GB):")
                logger.info(f"  Average: {avg_mem:.2f} GB")
                logger.info(f"  Max: {max_mem:.2f} GB")

        logger.info("=" * 60 + "\n")


def profile_inference(demo, batch, batch_idx=0):
    """
    Profile a single inference call with detailed timing.

    Args:
        demo: FourDGTDemo instance
        batch: Input batch
        batch_idx: Batch index
    """
    import torch.profiler

    # Warmup
    logger.info("Running warmup...")
    _ = demo.process_batch(batch, batch_idx, save_outputs=False)
    torch.cuda.synchronize()

    logger.info("Starting profiling...")

    # Profile with PyTorch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.profiler.record_function("model_inference"):
            outputs = demo.process_batch(batch, batch_idx, save_outputs=False)

    # Print profiler results
    logger.info("\n" + "=" * 60)
    logger.info("PyTorch Profiler Results")
    logger.info("=" * 60)

    # Print top operations by CUDA time
    logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Export to Chrome tracing format
    prof.export_chrome_trace("inference_trace.json")
    logger.info("Trace exported to inference_trace.json (view in chrome://tracing)")

    return outputs
