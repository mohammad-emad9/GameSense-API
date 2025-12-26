"""Performance monitoring service for GameSense API.

This module provides comprehensive system performance monitoring including
CPU, GPU, memory, disk I/O, and network statistics. It supports NVIDIA GPUs
via pynvml and falls back to WMI for AMD/Intel GPUs on Windows.

Example:
    >>> import asyncio
    >>> from core.performance_monitor import PerformanceMonitor
    >>>
    >>> async def main():
    ...     monitor = PerformanceMonitor()
    ...     await monitor.initialize()
    ...     stats = await monitor.get_system_stats()
    ...     print(f"CPU: {stats.cpu.usage_percent}%")
    ...     await monitor.cleanup()
    >>>
    >>> asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import psutil

from models.performance import (
    BottleneckAnalysis,
    BottleneckType,
    CPUStats,
    DiskStats,
    GPUStats,
    MemoryStats,
    NetworkStats,
    SystemStats,
)

# Configure structured JSON logging
logger = logging.getLogger(__name__)


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": getattr(record, "component", "performance"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


# Apply JSON formatter if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ThermalSample:
    """A single thermal/utilization sample for throttling detection.

    Attributes:
        timestamp: When the sample was taken.
        gpu_temp: GPU temperature in Celsius.
        cpu_temp: CPU temperature in Celsius (if available).
        gpu_utilization: GPU utilization percentage.
        cpu_utilization: CPU utilization percentage.
    """

    timestamp: float
    gpu_temp: Optional[float] = None
    cpu_temp: Optional[float] = None
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0


@dataclass
class PerformanceCache:
    """Thread-safe cache for performance metrics.

    Attributes:
        stats: Cached SystemStats.
        last_disk_counters: Previous disk I/O counters for speed calculation.
        last_disk_time: Timestamp of last disk reading.
        thermal_samples: Rolling window of thermal samples for throttling detection.
        latency_samples: Rolling window of network latency samples for jitter.
    """

    stats: Optional[SystemStats] = None
    last_disk_counters: Optional[Any] = None
    last_disk_time: float = 0.0
    thermal_samples: deque = field(default_factory=lambda: deque(maxlen=10))
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=20))


class PerformanceMonitor:
    """High-performance system monitoring service.

    Provides real-time performance telemetry for CPU, GPU, memory, disk, and
    network. Supports background monitoring with thread-safe caching.

    Attributes:
        _nvml_available: Whether NVIDIA NVML is available.
        _nvml_handle: NVML device handle (if available).
        _cache: Cached performance data.
        _cache_lock: Asyncio lock for thread-safe cache access.
        _executor: Thread pool for blocking I/O operations.
        _background_task: Background monitoring asyncio task.

    Example:
        >>> monitor = PerformanceMonitor()
        >>> await monitor.initialize()
        >>> stats = await monitor.get_system_stats()
        >>> print(f"GPU Temp: {stats.gpu.temperature}°C")
    """

    _PING_HOST: str = "8.8.8.8"
    _THROTTLE_TEMP_THRESHOLD: float = 90.0
    _THROTTLE_UTILIZATION_DROP: float = 20.0

    def __init__(self, executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Initialize the performance monitor.

        Args:
            executor: Optional thread pool executor for blocking operations.
                     If not provided, a default executor with 4 workers is created.
        """
        self._nvml_available: bool = False
        self._nvml_handle: Optional[Any] = None
        self._gpu_name: Optional[str] = None
        self._driver_version: Optional[str] = None
        self._wmi_available: bool = False
        self._wmi_gpu: Optional[Any] = None
        self._cache: PerformanceCache = PerformanceCache()
        self._cache_lock: asyncio.Lock = asyncio.Lock()
        self._executor: ThreadPoolExecutor = executor or ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="perf_monitor"
        )
        self._background_task: Optional[asyncio.Task] = None
        self._shutdown: bool = False

        logger.info(
            "PerformanceMonitor instance created",
            extra={"component": "performance"},
        )

    async def initialize(self) -> None:
        """Initialize GPU monitoring backends.

        Attempts to initialize NVIDIA NVML first, then falls back to WMI
        for AMD/Intel GPU support on Windows.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._init_gpu_backends)
        logger.info(
            f"GPU backends initialized: NVML={self._nvml_available}, WMI={self._wmi_available}",
            extra={"component": "performance"},
        )

    def _init_gpu_backends(self) -> None:
        """Initialize GPU monitoring backends (blocking).

        First attempts NVIDIA NVML, then falls back to WMI for Windows.
        """
        # Try NVIDIA NVML first
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                # Use first GPU by default
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_name = pynvml.nvmlDeviceGetName(self._nvml_handle)
                if isinstance(self._gpu_name, bytes):
                    self._gpu_name = self._gpu_name.decode("utf-8")
                self._driver_version = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(self._driver_version, bytes):
                    self._driver_version = self._driver_version.decode("utf-8")
                self._nvml_available = True
                logger.info(
                    f"NVML initialized: {self._gpu_name}",
                    extra={"component": "performance"},
                )
                return
        except ImportError:
            logger.warning(
                "pynvml not installed - NVIDIA GPU monitoring unavailable",
                extra={"component": "performance"},
            )
        except Exception as e:
            logger.warning(
                f"NVML initialization failed: {e}",
                extra={"component": "performance"},
            )

        # Fallback to WMI for AMD/Intel
        try:
            import wmi

            c = wmi.WMI()
            gpus = c.Win32_VideoController()
            if gpus:
                self._wmi_gpu = gpus[0]
                self._gpu_name = self._wmi_gpu.Name
                self._driver_version = self._wmi_gpu.DriverVersion
                self._wmi_available = True
                logger.info(
                    f"WMI GPU initialized: {self._gpu_name}",
                    extra={"component": "performance"},
                )
            else:
                logger.warning(
                    "No GPU found via WMI",
                    extra={"component": "performance"},
                )
        except ImportError:
            logger.warning(
                "wmi module not installed - WMI GPU monitoring unavailable",
                extra={"component": "performance"},
            )
        except Exception as e:
            logger.warning(
                f"WMI GPU initialization failed: {e}",
                extra={"component": "performance"},
            )

    async def cleanup(self) -> None:
        """Cleanup resources and shutdown monitoring.

        Cancels background task and shuts down NVML if initialized.
        """
        self._shutdown = True

        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            logger.info(
                "Background monitoring task cancelled",
                extra={"component": "performance"},
            )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._cleanup_nvml)

        self._executor.shutdown(wait=False)
        logger.info(
            "PerformanceMonitor cleanup complete",
            extra={"component": "performance"},
        )

    def _cleanup_nvml(self) -> None:
        """Shutdown NVML (blocking)."""
        if self._nvml_available:
            try:
                import pynvml

                pynvml.nvmlShutdown()
                logger.info(
                    "NVML shutdown complete",
                    extra={"component": "performance"},
                )
            except Exception as e:
                logger.warning(
                    f"NVML shutdown error: {e}",
                    extra={"component": "performance"},
                )

    async def get_system_stats(self) -> SystemStats:
        """Get current system performance statistics.

        Collects CPU, GPU, memory, disk, and network metrics. All operations
        that may block are run in a thread pool executor to avoid blocking
        the async event loop.

        Returns:
            SystemStats containing all current performance metrics.

        Example:
            >>> stats = await monitor.get_system_stats()
            >>> print(f"CPU: {stats.cpu.usage_percent}%")
        """
        loop = asyncio.get_event_loop()

        # Run all blocking operations concurrently in executor
        cpu_future = loop.run_in_executor(self._executor, self._get_cpu_stats)
        gpu_future = loop.run_in_executor(self._executor, self._get_gpu_stats)
        memory_future = loop.run_in_executor(self._executor, self._get_memory_stats)
        disk_future = loop.run_in_executor(self._executor, self._get_disk_stats)
        network_future = loop.run_in_executor(self._executor, self._get_network_stats)

        # Await all results
        cpu, gpu, memory, disk, network = await asyncio.gather(
            cpu_future, gpu_future, memory_future, disk_future, network_future
        )

        stats = SystemStats(
            cpu=cpu,
            gpu=gpu,
            memory=memory,
            disk=disk,
            network=network,
            timestamp=datetime.utcnow(),
        )

        # Update thermal samples for throttling detection
        async with self._cache_lock:
            self._cache.stats = stats
            self._cache.thermal_samples.append(
                ThermalSample(
                    timestamp=time.time(),
                    gpu_temp=gpu.temperature,
                    gpu_utilization=gpu.utilization,
                    cpu_utilization=cpu.usage_percent,
                )
            )

        return stats

    def _get_cpu_stats(self) -> CPUStats:
        """Get CPU statistics (blocking).

        Returns:
            CPUStats with current CPU metrics.
        """
        try:
            usage_percent = psutil.cpu_percent(interval=None)
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            core_count = psutil.cpu_count(logical=True) or 1

            freq = psutil.cpu_freq()
            frequency_mhz = freq.current if freq else None

            return CPUStats(
                usage_percent=usage_percent,
                per_core_usage=per_core,
                core_count=core_count,
                frequency_mhz=frequency_mhz,
            )
        except Exception as e:
            logger.warning(
                f"Failed to get CPU stats: {e}",
                extra={"component": "performance"},
            )
            return CPUStats()

    def _get_gpu_stats(self) -> GPUStats:
        """Get GPU statistics (blocking).

        Uses NVML for NVIDIA GPUs, WMI fallback for AMD/Intel.

        Returns:
            GPUStats with current GPU metrics, or defaults if unavailable.
        """
        if self._nvml_available:
            return self._get_nvidia_gpu_stats()
        elif self._wmi_available:
            return self._get_wmi_gpu_stats()
        else:
            return GPUStats(
                name=self._gpu_name,
                driver_version=self._driver_version,
            )

    def _get_nvidia_gpu_stats(self) -> GPUStats:
        """Get NVIDIA GPU statistics via NVML (blocking).

        Returns:
            GPUStats with NVIDIA GPU metrics.
        """
        try:
            import pynvml

            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            temperature = pynvml.nvmlDeviceGetTemperature(
                self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
            )

            return GPUStats(
                utilization=float(utilization.gpu),
                temperature=float(temperature),
                vram_used=memory.used,
                vram_total=memory.total,
                name=self._gpu_name,
                driver_version=self._driver_version,
            )
        except Exception as e:
            logger.warning(
                f"Failed to get NVIDIA GPU stats: {e}",
                extra={"component": "performance"},
            )
            return GPUStats(
                name=self._gpu_name,
                driver_version=self._driver_version,
            )

    def _get_wmi_gpu_stats(self) -> GPUStats:
        """Get GPU statistics via WMI for AMD/Intel (blocking).

        Note: WMI provides limited GPU metrics compared to NVML.
        Temperature is attempted via MSAcpi_ThermalZoneTemperature.

        Returns:
            GPUStats with available WMI GPU metrics.
        """
        try:
            import wmi

            # WMI doesn't provide real-time utilization like NVML
            # Try to get temperature from thermal zone
            temperature = None
            try:
                # Requires admin privileges
                w = wmi.WMI(namespace="root\\wmi")
                temp_info = w.MSAcpi_ThermalZoneTemperature()
                if temp_info:
                    # Convert from tenths of Kelvin to Celsius
                    temperature = (temp_info[0].CurrentTemperature / 10.0) - 273.15
            except Exception:
                # Expected to fail without admin privileges
                pass

            # Get VRAM info from Win32_VideoController
            vram_total = 0
            if self._wmi_gpu and hasattr(self._wmi_gpu, "AdapterRAM"):
                vram_total = self._wmi_gpu.AdapterRAM or 0
                # Handle negative values (32-bit overflow for large VRAM)
                if vram_total < 0:
                    vram_total = 0

            return GPUStats(
                utilization=0.0,  # Not available via WMI
                temperature=temperature,
                vram_used=0,  # Not available via WMI
                vram_total=vram_total,
                name=self._gpu_name,
                driver_version=self._driver_version,
            )
        except Exception as e:
            logger.warning(
                f"Failed to get WMI GPU stats: {e}",
                extra={"component": "performance"},
            )
            return GPUStats(
                name=self._gpu_name,
                driver_version=self._driver_version,
            )

    def _get_memory_stats(self) -> MemoryStats:
        """Get system memory statistics (blocking).

        Returns:
            MemoryStats with current RAM metrics.
        """
        try:
            mem = psutil.virtual_memory()
            return MemoryStats(
                used=mem.used,
                total=mem.total,
                percent=mem.percent,
                available=mem.available,
            )
        except Exception as e:
            logger.warning(
                f"Failed to get memory stats: {e}",
                extra={"component": "performance"},
            )
            return MemoryStats()

    def _get_disk_stats(self) -> DiskStats:
        """Get disk I/O statistics (blocking).

        Calculates read/write speeds based on delta from last reading.

        Returns:
            DiskStats with current disk I/O metrics.
        """
        try:
            counters = psutil.disk_io_counters()
            if counters is None:
                return DiskStats()

            current_time = time.time()
            read_speed = 0.0
            write_speed = 0.0

            if self._cache.last_disk_counters is not None:
                time_delta = current_time - self._cache.last_disk_time
                if time_delta > 0:
                    read_delta = counters.read_bytes - self._cache.last_disk_counters.read_bytes
                    write_delta = counters.write_bytes - self._cache.last_disk_counters.write_bytes
                    # Convert to MB/s
                    read_speed = (read_delta / time_delta) / (1024 * 1024)
                    write_speed = (write_delta / time_delta) / (1024 * 1024)

            # Update cache (non-async, but data is simple and thread-safe enough)
            self._cache.last_disk_counters = counters
            self._cache.last_disk_time = current_time

            return DiskStats(
                read_speed_mbps=round(read_speed, 2),
                write_speed_mbps=round(write_speed, 2),
                read_bytes=counters.read_bytes,
                write_bytes=counters.write_bytes,
            )
        except Exception as e:
            logger.warning(
                f"Failed to get disk stats: {e}",
                extra={"component": "performance"},
            )
            return DiskStats()

    def _get_network_stats(self) -> NetworkStats:
        """Get network statistics (blocking).

        Measures latency via ICMP ping and calculates jitter from samples.

        Returns:
            NetworkStats with current network metrics.
        """
        try:
            net_io = psutil.net_io_counters()
            bytes_sent = net_io.bytes_sent if net_io else 0
            bytes_recv = net_io.bytes_recv if net_io else 0

            # Measure latency using ping
            latency = self._measure_latency()

            # Calculate jitter from latency samples
            jitter = None
            if latency is not None:
                self._cache.latency_samples.append(latency)
                if len(self._cache.latency_samples) >= 2:
                    samples = list(self._cache.latency_samples)
                    differences = [
                        abs(samples[i] - samples[i - 1])
                        for i in range(1, len(samples))
                    ]
                    jitter = round(sum(differences) / len(differences), 2)

            return NetworkStats(
                latency_ms=latency,
                jitter_ms=jitter,
                bytes_sent=bytes_sent,
                bytes_recv=bytes_recv,
            )
        except Exception as e:
            logger.warning(
                f"Failed to get network stats: {e}",
                extra={"component": "performance"},
            )
            return NetworkStats()

    def _measure_latency(self) -> Optional[float]:
        """Measure network latency via ICMP ping (blocking).

        Returns:
            Latency in milliseconds, or None if ping fails.
        """
        try:
            # Use Windows ping command with single packet
            result = subprocess.run(
                ["ping", "-n", "1", "-w", "1000", self._PING_HOST],
                capture_output=True,
                text=True,
                timeout=2,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            if result.returncode == 0:
                # Parse latency from output (e.g., "time=25ms" or "time<1ms")
                output = result.stdout
                if "time=" in output:
                    time_part = output.split("time=")[1].split("ms")[0]
                    return float(time_part)
                elif "time<" in output:
                    time_part = output.split("time<")[1].split("ms")[0]
                    return float(time_part)
        except (subprocess.TimeoutExpired, ValueError, IndexError, Exception):
            pass
        return None

    async def is_thermal_throttling(self) -> bool:
        """Check if the system is experiencing thermal throttling.

        Uses heuristic: temperature > 90°C AND utilization drop > 20% in last 2s.

        Returns:
            True if thermal throttling is detected, False otherwise.

        Example:
            >>> if await monitor.is_thermal_throttling():
            ...     print("Warning: System is thermal throttling!")
        """
        async with self._cache_lock:
            samples = list(self._cache.thermal_samples)

        if len(samples) < 2:
            return False

        # Get samples from last 2 seconds
        current_time = time.time()
        recent_samples = [s for s in samples if current_time - s.timestamp <= 2.0]

        if len(recent_samples) < 2:
            return False

        # Check for high temperature
        latest = recent_samples[-1]
        highest_temp = max(
            s.gpu_temp or 0 for s in recent_samples
        )

        if highest_temp < self._THROTTLE_TEMP_THRESHOLD:
            return False

        # Check for utilization drop
        earliest = recent_samples[0]
        gpu_drop = earliest.gpu_utilization - latest.gpu_utilization
        cpu_drop = earliest.cpu_utilization - latest.cpu_utilization

        return gpu_drop >= self._THROTTLE_UTILIZATION_DROP or cpu_drop >= self._THROTTLE_UTILIZATION_DROP

    async def detect_bottleneck(self) -> BottleneckAnalysis:
        """Analyze current system state to detect performance bottlenecks.

        Examines CPU, GPU, memory, and network metrics to identify limiting factors.

        Returns:
            BottleneckAnalysis indicating the primary bottleneck (if any).

        Example:
            >>> analysis = await monitor.detect_bottleneck()
            >>> print(f"Bottleneck: {analysis.bottleneck_type.value}")
        """
        async with self._cache_lock:
            stats = self._cache.stats

        if stats is None:
            stats = await self.get_system_stats()

        # Check for thermal throttling first (highest priority)
        if await self.is_thermal_throttling():
            return BottleneckAnalysis(
                bottleneck_type=BottleneckType.THERMAL_THROTTLING,
                severity=0.9,
                description="System is thermal throttling - high temperature with utilization drop detected",
                recommendations=[
                    "Improve cooling (clean fans, improve airflow)",
                    "Reduce graphics settings",
                    "Check thermal paste application",
                    "Consider undervolting GPU/CPU",
                ],
            )

        # Check GPU limitation (high GPU usage, low CPU)
        if stats.gpu.utilization > 95 and stats.cpu.usage_percent < 70:
            return BottleneckAnalysis(
                bottleneck_type=BottleneckType.GPU_LIMITED,
                severity=0.8,
                description=f"GPU is at {stats.gpu.utilization}% utilization while CPU is at {stats.cpu.usage_percent}%",
                recommendations=[
                    "Lower resolution or graphics quality",
                    "Disable ray tracing if enabled",
                    "Reduce shadow quality and draw distance",
                    "Enable DLSS/FSR if supported",
                ],
            )

        # Check CPU bottleneck (high CPU, low GPU)
        if stats.cpu.usage_percent > 90 and stats.gpu.utilization < 70:
            return BottleneckAnalysis(
                bottleneck_type=BottleneckType.CPU_BOUND,
                severity=0.8,
                description=f"CPU is at {stats.cpu.usage_percent}% while GPU is only at {stats.gpu.utilization}%",
                recommendations=[
                    "Close background applications",
                    "Reduce game's CPU-intensive settings (physics, AI, draw calls)",
                    "Increase resolution to shift load to GPU",
                    "Check for CPU thermal throttling",
                ],
            )

        # Check memory pressure
        if stats.memory.percent > 90:
            return BottleneckAnalysis(
                bottleneck_type=BottleneckType.MEMORY_PRESSURE,
                severity=0.7,
                description=f"RAM usage is at {stats.memory.percent}% ({stats.memory.used_gb:.1f}GB / {stats.memory.total_gb:.1f}GB)",
                recommendations=[
                    "Close memory-heavy background applications",
                    "Reduce texture quality in game settings",
                    "Check for memory leaks in running applications",
                    "Consider adding more RAM",
                ],
            )

        # Check VRAM pressure
        if stats.gpu.vram_total > 0:
            vram_percent = (stats.gpu.vram_used / stats.gpu.vram_total) * 100
            if vram_percent > 95:
                return BottleneckAnalysis(
                    bottleneck_type=BottleneckType.GPU_LIMITED,
                    severity=0.75,
                    description=f"VRAM usage is critical at {vram_percent:.1f}%",
                    recommendations=[
                        "Lower texture quality",
                        "Reduce render resolution",
                        "Disable texture streaming if available",
                    ],
                )

        # Check network issues
        if stats.network.jitter_ms is not None and stats.network.jitter_ms > 50:
            return BottleneckAnalysis(
                bottleneck_type=BottleneckType.NETWORK_SPIKE,
                severity=0.6,
                description=f"High network jitter detected: {stats.network.jitter_ms}ms",
                recommendations=[
                    "Check for network congestion",
                    "Use wired connection instead of WiFi",
                    "Close bandwidth-heavy applications",
                    "Contact ISP if issue persists",
                ],
            )

        if stats.network.latency_ms is not None and stats.network.latency_ms > 100:
            return BottleneckAnalysis(
                bottleneck_type=BottleneckType.NETWORK_SPIKE,
                severity=0.5,
                description=f"High network latency: {stats.network.latency_ms}ms",
                recommendations=[
                    "Use servers closer to your region",
                    "Check for VPN overhead",
                    "Restart router/modem",
                ],
            )

        # No bottleneck detected
        return BottleneckAnalysis(
            bottleneck_type=BottleneckType.NONE,
            severity=0.0,
            description="System is running optimally with no detected bottlenecks",
            recommendations=None,
        )

    async def start_background_monitoring(
        self, interval: float = 1.0
    ) -> asyncio.Task:
        """Start continuous background performance monitoring.

        Samples system stats at the specified interval and maintains a
        thread-safe cache of the latest readings.

        Args:
            interval: Sampling interval in seconds (default: 1.0).

        Returns:
            The asyncio Task running the background monitor.

        Example:
            >>> task = await monitor.start_background_monitoring(interval=0.5)
            >>> # ... later ...
            >>> task.cancel()
        """
        if self._background_task and not self._background_task.done():
            logger.warning(
                "Background monitoring already running",
                extra={"component": "performance"},
            )
            return self._background_task

        self._shutdown = False
        self._background_task = asyncio.create_task(
            self._background_monitor_loop(interval)
        )

        logger.info(
            f"Background monitoring started with {interval}s interval",
            extra={"component": "performance"},
        )

        return self._background_task

    async def _background_monitor_loop(self, interval: float) -> None:
        """Background monitoring loop (internal).

        Args:
            interval: Sampling interval in seconds.
        """
        while not self._shutdown:
            try:
                await self.get_system_stats()
            except Exception as e:
                logger.error(
                    f"Background monitoring error: {e}",
                    extra={"component": "performance"},
                )
            await asyncio.sleep(interval)

    async def get_cached_stats(self) -> Optional[SystemStats]:
        """Get the most recently cached system stats.

        Returns cached data without triggering new measurements.
        Useful for high-frequency access patterns.

        Returns:
            Cached SystemStats, or None if no data has been collected yet.
        """
        async with self._cache_lock:
            return self._cache.stats


if __name__ == "__main__":
    # Example usage and basic testing
    import sys

    async def main() -> None:
        """Run performance monitor demonstration."""
        print("GameSense API - Performance Monitor Demo")
        print("=" * 50)

        monitor = PerformanceMonitor()

        try:
            print("\nInitializing GPU backends...")
            await monitor.initialize()

            print("\nCollecting system stats...")
            stats = await monitor.get_system_stats()

            print("\n--- CPU Stats ---")
            print(f"Usage: {stats.cpu.usage_percent}%")
            print(f"Cores: {stats.cpu.core_count}")
            if stats.cpu.frequency_mhz:
                print(f"Frequency: {stats.cpu.frequency_mhz:.0f} MHz")
            if stats.cpu.per_core_usage:
                print(f"Per-core: {[f'{u:.1f}%' for u in stats.cpu.per_core_usage[:4]]}...")

            print("\n--- GPU Stats ---")
            print(f"Name: {stats.gpu.name or 'Unknown'}")
            print(f"Utilization: {stats.gpu.utilization}%")
            if stats.gpu.temperature is not None:
                print(f"Temperature: {stats.gpu.temperature}°C")
            print(f"VRAM: {stats.gpu.vram_used_gb:.2f} / {stats.gpu.vram_total_gb:.2f} GB")

            print("\n--- Memory Stats ---")
            print(f"RAM: {stats.memory.used_gb:.2f} / {stats.memory.total_gb:.2f} GB ({stats.memory.percent}%)")

            print("\n--- Disk Stats ---")
            print(f"Read: {stats.disk.read_speed_mbps:.2f} MB/s")
            print(f"Write: {stats.disk.write_speed_mbps:.2f} MB/s")

            print("\n--- Network Stats ---")
            if stats.network.latency_ms is not None:
                print(f"Latency: {stats.network.latency_ms}ms")
            if stats.network.jitter_ms is not None:
                print(f"Jitter: {stats.network.jitter_ms}ms")

            print("\n--- Bottleneck Analysis ---")
            analysis = await monitor.detect_bottleneck()
            print(f"Type: {analysis.bottleneck_type.value}")
            print(f"Severity: {analysis.severity}")
            print(f"Description: {analysis.description}")

            print("\n--- Thermal Throttling ---")
            throttling = await monitor.is_thermal_throttling()
            print(f"Throttling detected: {throttling}")

            print("\nDemo complete!")

        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            raise
        finally:
            print("\nCleaning up...")
            await monitor.cleanup()

    asyncio.run(main())
