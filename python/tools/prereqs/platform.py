# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Platform detection utilities for nvFuser build system.

Detects OS, architecture, and Linux distribution to provide platform-specific
error messages and installation guidance.
"""

import platform
from typing import Dict, Optional


def detect_platform() -> Dict[str, Optional[str]]:
    """
    Detect the current platform and return structured information.
    
    Returns:
        dict: Platform information with keys:
            - 'os': Operating system (Linux, Darwin, Windows, etc.)
            - 'arch': Architecture (x86_64, aarch64, arm64, etc.)
            - 'distro': Linux distribution ID (ubuntu, debian, rhel, etc.) or None
            - 'distro_version': Distribution version (22.04, 20.04, etc.) or None
            - 'distro_name': Human-readable distribution name or None
            - 'ubuntu_based': Boolean indicating if this is Ubuntu-based distro
    
    Example:
        >>> info = detect_platform()
        >>> print(info['os'])
        'Linux'
        >>> print(info['distro'])
        'ubuntu'
    """
    system = platform.system()
    machine = platform.machine()
    
    # Initialize distro information
    distro_info = {}
    distro_id = None
    distro_version = None
    distro_name = None
    ubuntu_based = False
    
    # Detect Linux distribution from /etc/os-release
    if system == "Linux":
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line:
                        key, value = line.split("=", 1)
                        # Remove quotes from value
                        distro_info[key] = value.strip('"').strip("'")
            
            distro_id = distro_info.get("ID", "unknown")
            distro_version = distro_info.get("VERSION_ID", "unknown")
            distro_name = distro_info.get("NAME", "unknown")
            
            # Check if Ubuntu-based (useful for PPA availability)
            ubuntu_based = distro_id in ("ubuntu", "debian", "linuxmint", "pop", "zorin")
            
        except FileNotFoundError:
            # /etc/os-release doesn't exist (not a standard Linux or very old system)
            distro_id = "unknown"
            distro_version = "unknown"
            distro_name = "unknown"
        except Exception as e:
            # Other errors reading/parsing the file
            distro_id = f"error: {e}"
            distro_version = "unknown"
            distro_name = "unknown"
    
    return {
        "os": system,
        "arch": machine,
        "distro": distro_id,
        "distro_version": distro_version,
        "distro_name": distro_name,
        "ubuntu_based": ubuntu_based,
    }


def format_platform_info(platform_info: Optional[Dict[str, Optional[str]]] = None) -> str:
    """
    Format platform information as a human-readable string.
    
    Args:
        platform_info: Platform information dict from detect_platform().
                      If None, will call detect_platform() automatically.
    
    Returns:
        str: Formatted platform string like "Linux x86_64 (Ubuntu 22.04)"
    
    Example:
        >>> print(format_platform_info())
        'Linux x86_64 (Ubuntu 22.04)'
    """
    if platform_info is None:
        platform_info = detect_platform()
    
    os_name = platform_info["os"]
    arch = platform_info["arch"]
    
    # Build distro info if available
    distro_parts = []
    if platform_info.get("distro") and platform_info["distro"] not in ("unknown", "error"):
        distro_parts.append(platform_info["distro"].capitalize())
    if platform_info.get("distro_version") and platform_info["distro_version"] != "unknown":
        distro_parts.append(platform_info["distro_version"])
    
    if distro_parts:
        return f"{os_name} {arch} ({' '.join(distro_parts)})"
    else:
        return f"{os_name} {arch}"

