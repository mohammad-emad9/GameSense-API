"""Log file reader telemetry strategy.

This strategy reads and parses game log files to extract telemetry data.
Supports various log formats including JSON, CSV, and custom patterns.

Example:
    >>> strategy = LogReaderStrategy(config={
    ...     "log_path": "C:/Games/MyGame/logs/game.log",
    ...     "format": "json"
    ... })
    >>> if strategy.is_available():
    ...     data = await strategy.read()
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from adapters.base import BaseTelemetryStrategy
from models.game import TelemetrySource

# Configure logging
logger = logging.getLogger(__name__)


class LogReaderStrategy(BaseTelemetryStrategy):
    """Telemetry strategy that reads game log files.

    Supports multiple log formats and custom parsing patterns.
    This is the highest priority strategy as it requires no admin
    privileges and provides rich data.

    Attributes:
        source_type: TelemetrySource.LOG
        priority: 1 (highest)
        requires_admin: False

    Example:
        >>> strategy = LogReaderStrategy(config={
        ...     "log_path": "C:/Games/CS2/csgo/logs/gsi.log",
        ...     "format": "json",
        ...     "tail_lines": 100
        ... })
    """

    source_type = TelemetrySource.LOG
    priority = 1
    requires_admin = False

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the log reader strategy.

        Args:
            config: Configuration dict with keys:
                - log_path: Path to the log file (required)
                - format: Log format ('json', 'csv', 'text', 'custom')
                - tail_lines: Number of lines to read from end (default: 50)
                - patterns: Dict of regex patterns for 'custom' format
                - encoding: File encoding (default: 'utf-8')
        """
        super().__init__(config)

        self.log_path: Optional[Path] = None
        if config and "log_path" in config:
            self.log_path = Path(config["log_path"])

        self.format = config.get("format", "text") if config else "text"
        self.tail_lines = config.get("tail_lines", 50) if config else 50
        self.patterns = config.get("patterns", {}) if config else {}
        self.encoding = config.get("encoding", "utf-8") if config else "utf-8"
        self._last_position: int = 0
        self._last_modify_time: float = 0

    def is_available(self) -> bool:
        """Check if the log file exists and is readable.

        Returns:
            True if log file is accessible, False otherwise.
        """
        if self.log_path is None:
            return False

        try:
            return self.log_path.exists() and self.log_path.is_file()
        except (PermissionError, OSError):
            return False

    async def read(self) -> Dict[str, Any]:
        """Read telemetry data from the log file.

        Returns:
            Dictionary with 'state', 'stats', and/or 'metadata' keys.

        Raises:
            FileNotFoundError: If log file doesn't exist.
            PermissionError: If log file isn't readable.
        """
        if not self.log_path or not self.is_available():
            return {}

        # Read in executor to avoid blocking
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, self._read_file)

        if not content:
            return {}

        # Parse based on format
        if self.format == "json":
            return self._parse_json(content)
        elif self.format == "csv":
            return self._parse_csv(content)
        elif self.format == "custom":
            return self._parse_custom(content)
        else:
            return self._parse_text(content)

    def _read_file(self) -> str:
        """Read the log file content (blocking).

        Returns:
            File content as string.
        """
        try:
            if not self.log_path:
                return ""

            # Check if file has been modified
            stat = self.log_path.stat()
            if stat.st_mtime <= self._last_modify_time:
                # No changes, return empty
                return ""

            self._last_modify_time = stat.st_mtime

            # Read last N lines
            with open(self.log_path, "r", encoding=self.encoding, errors="replace") as f:
                lines = f.readlines()
                return "".join(lines[-self.tail_lines:])

        except Exception as e:
            logger.warning(
                f"Failed to read log file {self.log_path}: {e}",
                extra={"component": "strategy"},
            )
            return ""

    def _parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON format log content.

        Args:
            content: Raw log content.

        Returns:
            Parsed telemetry data.
        """
        result: Dict[str, Any] = {"state": {}, "stats": {}, "metadata": {}}

        # Try to parse as JSON lines (one JSON object per line)
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # Extract known fields
                if isinstance(data, dict):
                    for key in ["state", "stats", "metadata"]:
                        if key in data and isinstance(data[key], dict):
                            result[key].update(data[key])

                    # Also extract top-level fields
                    for key, value in data.items():
                        if key not in ["state", "stats", "metadata"]:
                            # Heuristic: numeric values go to stats
                            if isinstance(value, (int, float)):
                                result["stats"][key] = value
                            else:
                                result["state"][key] = value

            except json.JSONDecodeError:
                continue

        return result

    def _parse_csv(self, content: str) -> Dict[str, Any]:
        """Parse CSV format log content.

        Args:
            content: Raw log content.

        Returns:
            Parsed telemetry data.
        """
        result: Dict[str, Any] = {"state": {}, "stats": {}}
        lines = content.strip().split("\n")

        if len(lines) < 2:
            return result

        # First line is header
        headers = [h.strip() for h in lines[0].split(",")]

        # Last line is most recent data
        values = [v.strip() for v in lines[-1].split(",")]

        for i, header in enumerate(headers):
            if i < len(values):
                value = values[i]
                # Try to convert to number
                try:
                    if "." in value:
                        result["stats"][header] = float(value)
                    else:
                        result["stats"][header] = int(value)
                except ValueError:
                    result["state"][header] = value

        return result

    def _parse_custom(self, content: str) -> Dict[str, Any]:
        """Parse log content using custom regex patterns.

        Args:
            content: Raw log content.

        Returns:
            Parsed telemetry data.
        """
        result: Dict[str, Any] = {"state": {}, "stats": {}}

        for key, pattern in self.patterns.items():
            try:
                match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
                if match:
                    value = match.group(1) if match.groups() else match.group(0)
                    # Try to convert to number
                    try:
                        if "." in value:
                            result["stats"][key] = float(value)
                        else:
                            result["stats"][key] = int(value)
                    except ValueError:
                        result["state"][key] = value
            except re.error:
                continue

        return result

    def _parse_text(self, content: str) -> Dict[str, Any]:
        """Parse plain text log content with common patterns.

        Args:
            content: Raw log content.

        Returns:
            Parsed telemetry data.
        """
        result: Dict[str, Any] = {"state": {}, "stats": {}}

        # Common patterns for game logs
        patterns = {
            "map": r"map[:\s]+([^\s\n]+)",
            "level": r"level[:\s]+(\d+)",
            "score": r"score[:\s]+(\d+)",
            "health": r"health[:\s]+(\d+)",
            "kills": r"kills[:\s]+(\d+)",
            "deaths": r"deaths[:\s]+(\d+)",
        }

        for key, pattern in patterns.items():
            try:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    try:
                        result["stats"][key] = int(value)
                    except ValueError:
                        result["state"][key] = value
            except re.error:
                continue

        return result
