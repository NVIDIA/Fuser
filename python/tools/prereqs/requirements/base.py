# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Base classes for requirement types.

This module provides the foundational classes for all dependency requirements.
Each requirement knows how to format its status and determine if it represents a failure.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class RequirementStatus:
    """Validation status constants."""
    SUCCESS = "SUCCESS"
    NOT_FOUND = "NOT_FOUND"
    INCOMPATIBLE = "INCOMPATIBLE"


class Requirement(ABC):
    """
    Base class for all requirement types.

    All requirements must implement:
    - format_status_line(): Format output for terminal
    - is_failure(): Determine if this represents a build failure
    """

    def __init__(
        self,
        name: str,
        cmake_vars: Dict,
        found_var: str,
        status_var: str,
        optional_var: str,
        location_var: Optional[str] = None,
    ):
        """
        Initialize from CMake variable names.

        Args:
            name: Dependency name
            cmake_vars: Dictionary of all CMake variables
            found_var: Name of CMake variable for found status (e.g., "Python_FOUND")
            status_var: Name of CMake variable for validation status (e.g., "Python_STATUS")
            optional_var: Name of CMake variable for optional flag (e.g., "NVFUSER_REQUIREMENT_Python_OPTIONAL")
            location_var: Optional name of CMake variable for location (e.g., "Python_EXECUTABLE")
        """
        self.name = name
        self.found = self._to_bool(cmake_vars.get(found_var, "FALSE"))
        self.status = cmake_vars.get(status_var, "UNKNOWN")
        self.optional = self._to_bool(cmake_vars.get(optional_var, "FALSE"))

        # Look up location if location_var is provided
        self.location = cmake_vars.get(location_var) if location_var else None

    @staticmethod
    def _to_bool(value) -> bool:
        """Convert CMake boolean string to Python bool."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.upper() in ("TRUE", "ON", "YES", "1")
        return bool(value)

    @abstractmethod
    def format_status_line(self, colors) -> str:
        """Format terminal status line for this requirement."""
        pass

    def is_failure(self) -> bool:
        """Check if this requirement represents a failure."""
        return not self.optional and self.status != RequirementStatus.SUCCESS

    def get_failure_data(self):
        """Get data for help text generation."""
        # Return a dict compatible with help system
        return {
            "name": self.name,
            "status": self.status,
            "found": self.found,
            "optional": self.optional,
            "location": self.location,
        }

    def generate_help(self, platform_info):
        """Generate help text for this requirement when it fails.

        Subclasses should override this to provide specific installation instructions.
        """
        pass  # Default: no help text


class VersionRequirement(Requirement):
    """
    Base class for requirements with version checking.

    Provides standard version display formatting.
    Subclasses inherit all version comparison logic.
    """

    def __init__(
        self,
        name: str,
        cmake_vars: Dict,
        found_var: str,
        status_var: str,
        optional_var: str,
        version_found_var: Optional[str] = None,
        version_required_var: Optional[str] = None,
        location_var: Optional[str] = None,
    ):
        """
        Initialize version requirement.

        Args:
            name: Dependency name
            cmake_vars: Dictionary of all CMake variables
            found_var: Name of CMake variable for found status
            status_var: Name of CMake variable for validation status
            optional_var: Name of CMake variable for optional flag
            version_found_var: Name of CMake variable for detected version (e.g., "Python_VERSION")
            version_required_var: Name of CMake variable for minimum required version (e.g., "NVFUSER_REQUIREMENT_Python_VERSION_MIN")
            location_var: Optional name of CMake variable for location
        """
        super().__init__(name, cmake_vars, found_var, status_var, optional_var, location_var)
        self.version_found = cmake_vars.get(version_found_var) if version_found_var else None
        self.version_required = cmake_vars.get(version_required_var) if version_required_var else None

    def format_status_line(self, colors) -> str:
        """Format status line with version information."""
        if self.status == RequirementStatus.SUCCESS:
            return self._format_success(colors)
        elif self.status == RequirementStatus.NOT_FOUND:
            return self._format_not_found(colors)
        elif self.status == RequirementStatus.INCOMPATIBLE:
            return self._format_incompatible(colors)
        else:
            return f"{colors.BOLD_RED}[nvFuser] ✗ {self.name} unknown status{colors.RESET}"

    def _format_success(self, colors) -> str:
        """Format success: [nvFuser] ✓ Python 3.12.3 >= 3.8"""
        if self.version_found and self.version_required:
            return f"{colors.GREEN}[nvFuser] ✓ {self.name} {self.version_found} >= {self.version_required}{colors.RESET}"
        elif self.version_found:
            return f"{colors.GREEN}[nvFuser] ✓ {self.name} {self.version_found}{colors.RESET}"
        else:
            return f"{colors.GREEN}[nvFuser] ✓ {self.name}{colors.RESET}"

    def _format_not_found(self, colors) -> str:
        """Format not found line."""
        if self.optional:
            if self.version_required:
                return f"{colors.YELLOW}[nvFuser] ○ {self.name} NOT found (optional, v{self.version_required}+ recommended){colors.RESET}"
            else:
                return f"{colors.YELLOW}[nvFuser] ○ {self.name} NOT found (optional){colors.RESET}"
        else:
            if self.version_required:
                return f"{colors.BOLD_RED}[nvFuser] ✗ {self.name} NOT found (requires {self.version_required}+){colors.RESET}"
            else:
                return f"{colors.BOLD_RED}[nvFuser] ✗ {self.name} NOT found{colors.RESET}"

    def _format_incompatible(self, colors) -> str:
        """Format incompatible: [nvFuser] ✗ Python 3.7.0 < 3.8"""
        if self.version_found and self.version_required:
            return f"{colors.BOLD_RED}[nvFuser] ✗ {self.name} {self.version_found} < {self.version_required}{colors.RESET}"
        else:
            return f"{colors.BOLD_RED}[nvFuser] ✗ {self.name} incompatible{colors.RESET}"


class BooleanRequirement(Requirement):
    """
    Base class for requirements without version checking.

    Simple pass/fail validation (Git submodules, Ninja).
    """

    def format_status_line(self, colors) -> str:
        """Format status line without version information."""
        if self.status == RequirementStatus.SUCCESS:
            return f"{colors.GREEN}[nvFuser] ✓ {self.name}{colors.RESET}"
        elif self.status == RequirementStatus.NOT_FOUND:
            if self.optional:
                return f"{colors.YELLOW}[nvFuser] ○ {self.name} NOT found (optional){colors.RESET}"
            else:
                return f"{colors.BOLD_RED}[nvFuser] ✗ {self.name} NOT found{colors.RESET}"
        else:
            return f"{colors.BOLD_RED}[nvFuser] ✗ {self.name} validation failed{colors.RESET}"
