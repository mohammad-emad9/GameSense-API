"""Performance telemetry Pydantic models for GameSense API.

This module defines type-safe, validated data models for system performance
metrics including CPU, GPU, memory, disk I/O, and network statistics.

Example:
    >>> from models.performance import SystemStats, CPUStats
    >>> cpu = CPUStats(usage_percent=45.2, per_core_usage=[40.0, 50.0], core_count=2)
    >>> print(cpu.usage_percent)
    45.2
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BottleneckType(str, Enum):
    """Enumeration of possible system bottleneck types.

    Attributes:
        NONE: No bottleneck detected.
        CPU_BOUND: System is limited by CPU performance.
        GPU_LIMITED: System is limited by GPU performance.
        MEMORY_PRESSURE: System is experiencing memory pressure.
        THERMAL_THROTTLING: System is throttling due to high temperatures.
        DISK_BOTTLENECK: System is limited by disk I/O.
        NETWORK_SPIKE: Network latency or jitter is abnormally high.
    """

    NONE = "none"
    CPU_BOUND = "cpu_bound"
    GPU_LIMITED = "gpu_limited"
    MEMORY_PRESSURE = "memory_pressure"
    THERMAL_THROTTLING = "thermal_throttling"
    DISK_BOTTLENECK = "disk_bottleneck"
    NETWORK_SPIKE = "network_spike"


class CPUStats(BaseModel):
    """CPU performance statistics.

    Attributes:
        usage_percent: Overall CPU usage as a percentage (0-100).
        per_core_usage: Optional list of per-core usage percentages.
        core_count: Number of logical CPU cores.
        frequency_mhz: Current CPU frequency in MHz (optional).

    Example:
        >>> cpu = CPUStats(usage_percent=65.5, core_count=8)
        >>> cpu.usage_percent
        65.5
    """

    usage_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall CPU usage percentage (0-100)",
    )
    per_core_usage: Optional[list[float]] = Field(
        default=None,
        description="Per-core CPU usage percentages",
    )
    core_count: int = Field(
        default=1,
        ge=1,
        description="Number of logical CPU cores",
    )
    frequency_mhz: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Current CPU frequency in MHz",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "usage_percent": 45.2,
                "per_core_usage": [40.0, 50.0, 42.0, 48.0],
                "core_count": 4,
                "frequency_mhz": 3600.0,
            }
        }


class GPUStats(BaseModel):
    """GPU performance statistics.

    Supports NVIDIA GPUs via NVML and provides fallback defaults for
    AMD/Intel GPUs when detailed metrics are unavailable.

    Attributes:
        utilization: GPU utilization percentage (0-100).
        temperature: GPU temperature in Celsius (None if unavailable).
        vram_used: VRAM currently in use in bytes.
        vram_total: Total available VRAM in bytes.
        name: GPU device name (optional).
        driver_version: GPU driver version string (optional).

    Example:
        >>> gpu = GPUStats(utilization=80.0, temperature=72.0, vram_used=4*1024**3, vram_total=8*1024**3)
        >>> gpu.vram_used_gb
        4.0
    """

    utilization: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="GPU utilization percentage (0-100)",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="GPU temperature in Celsius",
    )
    vram_used: int = Field(
        default=0,
        ge=0,
        description="VRAM used in bytes",
    )
    vram_total: int = Field(
        default=0,
        ge=0,
        description="Total VRAM in bytes",
    )
    name: Optional[str] = Field(
        default=None,
        description="GPU device name",
    )
    driver_version: Optional[str] = Field(
        default=None,
        description="GPU driver version",
    )

    @property
    def vram_used_gb(self) -> float:
        """Get VRAM used in gigabytes.

        Returns:
            VRAM used in GB, rounded to 2 decimal places.
        """
        return round(self.vram_used / (1024**3), 2) if self.vram_used > 0 else 0.0

    @property
    def vram_total_gb(self) -> float:
        """Get total VRAM in gigabytes.

        Returns:
            Total VRAM in GB, rounded to 2 decimal places.
        """
        return round(self.vram_total / (1024**3), 2) if self.vram_total > 0 else 0.0

    @property
    def vram_usage_percent(self) -> float:
        """Calculate VRAM usage as a percentage.

        Returns:
            VRAM usage percentage (0-100), or 0.0 if total is zero.
        """
        if self.vram_total == 0:
            return 0.0
        return round((self.vram_used / self.vram_total) * 100, 2)

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "utilization": 75.0,
                "temperature": 68.5,
                "vram_used": 4294967296,
                "vram_total": 8589934592,
                "name": "NVIDIA GeForce RTX 3080",
                "driver_version": "535.154.05",
            }
        }


class MemoryStats(BaseModel):
    """System RAM statistics.

    Attributes:
        used: RAM currently in use in bytes.
        total: Total available RAM in bytes.
        percent: RAM usage as a percentage (0-100).
        available: Available RAM in bytes.

    Example:
        >>> mem = MemoryStats(used=8*1024**3, total=16*1024**3, percent=50.0)
        >>> mem.used_gb
        8.0
    """

    used: int = Field(
        default=0,
        ge=0,
        description="RAM used in bytes",
    )
    total: int = Field(
        default=0,
        ge=0,
        description="Total RAM in bytes",
    )
    percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="RAM usage percentage (0-100)",
    )
    available: int = Field(
        default=0,
        ge=0,
        description="Available RAM in bytes",
    )

    @property
    def used_gb(self) -> float:
        """Get RAM used in gigabytes.

        Returns:
            RAM used in GB, rounded to 2 decimal places.
        """
        return round(self.used / (1024**3), 2) if self.used > 0 else 0.0

    @property
    def total_gb(self) -> float:
        """Get total RAM in gigabytes.

        Returns:
            Total RAM in GB, rounded to 2 decimal places.
        """
        return round(self.total / (1024**3), 2) if self.total > 0 else 0.0

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "used": 8589934592,
                "total": 17179869184,
                "percent": 50.0,
                "available": 8589934592,
            }
        }


class DiskStats(BaseModel):
    """Disk I/O performance statistics.

    Attributes:
        read_speed_mbps: Disk read speed in MB/s.
        write_speed_mbps: Disk write speed in MB/s.
        read_bytes: Total bytes read since boot.
        write_bytes: Total bytes written since boot.

    Example:
        >>> disk = DiskStats(read_speed_mbps=150.5, write_speed_mbps=120.3)
        >>> disk.read_speed_mbps
        150.5
    """

    read_speed_mbps: float = Field(
        default=0.0,
        ge=0.0,
        description="Disk read speed in MB/s",
    )
    write_speed_mbps: float = Field(
        default=0.0,
        ge=0.0,
        description="Disk write speed in MB/s",
    )
    read_bytes: int = Field(
        default=0,
        ge=0,
        description="Total bytes read since boot",
    )
    write_bytes: int = Field(
        default=0,
        ge=0,
        description="Total bytes written since boot",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "read_speed_mbps": 250.5,
                "write_speed_mbps": 180.3,
                "read_bytes": 104857600000,
                "write_bytes": 52428800000,
            }
        }


class NetworkStats(BaseModel):
    """Network performance statistics.

    Attributes:
        latency_ms: Network latency in milliseconds.
        jitter_ms: Network latency jitter (variation) in milliseconds.
        bytes_sent: Total bytes sent since boot.
        bytes_recv: Total bytes received since boot.

    Example:
        >>> net = NetworkStats(latency_ms=25.0, jitter_ms=5.0)
        >>> net.latency_ms
        25.0
    """

    latency_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Network latency in milliseconds",
    )
    jitter_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Network latency jitter in milliseconds",
    )
    bytes_sent: int = Field(
        default=0,
        ge=0,
        description="Total bytes sent since boot",
    )
    bytes_recv: int = Field(
        default=0,
        ge=0,
        description="Total bytes received since boot",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "latency_ms": 25.5,
                "jitter_ms": 3.2,
                "bytes_sent": 1073741824,
                "bytes_recv": 5368709120,
            }
        }


class SystemStats(BaseModel):
    """Aggregated system performance statistics.

    This is the primary response model for the performance endpoint,
    containing all subsystem metrics in a single response.

    Attributes:
        cpu: CPU performance statistics.
        gpu: GPU performance statistics.
        memory: RAM statistics.
        disk: Disk I/O statistics.
        network: Network statistics.
        timestamp: ISO 8601 timestamp of when stats were collected.

    Example:
        >>> stats = SystemStats(
        ...     cpu=CPUStats(usage_percent=45.0, core_count=8),
        ...     gpu=GPUStats(utilization=70.0, temperature=65.0),
        ...     memory=MemoryStats(used=8*1024**3, total=16*1024**3, percent=50.0),
        ... )
    """

    cpu: CPUStats = Field(
        default_factory=CPUStats,
        description="CPU performance statistics",
    )
    gpu: GPUStats = Field(
        default_factory=GPUStats,
        description="GPU performance statistics",
    )
    memory: MemoryStats = Field(
        default_factory=MemoryStats,
        description="RAM statistics",
    )
    disk: DiskStats = Field(
        default_factory=DiskStats,
        description="Disk I/O statistics",
    )
    network: NetworkStats = Field(
        default_factory=NetworkStats,
        description="Network statistics",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when stats were collected (UTC)",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "cpu": {
                    "usage_percent": 45.2,
                    "per_core_usage": [40.0, 50.0, 42.0, 48.0],
                    "core_count": 4,
                    "frequency_mhz": 3600.0,
                },
                "gpu": {
                    "utilization": 75.0,
                    "temperature": 68.5,
                    "vram_used": 4294967296,
                    "vram_total": 8589934592,
                    "name": "NVIDIA GeForce RTX 3080",
                },
                "memory": {
                    "used": 8589934592,
                    "total": 17179869184,
                    "percent": 50.0,
                    "available": 8589934592,
                },
                "disk": {
                    "read_speed_mbps": 250.5,
                    "write_speed_mbps": 180.3,
                },
                "network": {
                    "latency_ms": 25.5,
                    "jitter_ms": 3.2,
                },
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class BottleneckAnalysis(BaseModel):
    """System bottleneck analysis result.

    Provides insight into what component may be limiting system performance.

    Attributes:
        bottleneck_type: The type of bottleneck detected.
        severity: Severity level from 0.0 (none) to 1.0 (critical).
        description: Human-readable description of the bottleneck.
        recommendations: Optional list of recommendations to address the issue.

    Example:
        >>> analysis = BottleneckAnalysis(
        ...     bottleneck_type=BottleneckType.GPU_LIMITED,
        ...     severity=0.7,
        ...     description="GPU utilization is at 95%, likely limiting frame rate"
        ... )
    """

    bottleneck_type: BottleneckType = Field(
        default=BottleneckType.NONE,
        description="Type of bottleneck detected",
    )
    severity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Severity level (0.0=none, 1.0=critical)",
    )
    description: str = Field(
        default="No bottleneck detected",
        description="Human-readable description of the bottleneck",
    )
    recommendations: Optional[list[str]] = Field(
        default=None,
        description="Optional list of recommendations",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "bottleneck_type": "gpu_limited",
                "severity": 0.7,
                "description": "GPU utilization is at 95%, likely limiting frame rate",
                "recommendations": [
                    "Lower graphics settings",
                    "Reduce render resolution",
                    "Disable ray tracing",
                ],
            }
        }
