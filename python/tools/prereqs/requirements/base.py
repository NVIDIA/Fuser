# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Base classes for requirement types.

This module provides the foundational classes for all dependency requirements.
Each requirement knows how to format its status and determine if it represents a failure.
"""

from abc import ABC, abstractmethod
from typing import Dict
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
    - get_failure_data(): Provide data for help text generation
    """

    def __init__(self, data: Dict):
        """
        Initialize from CMake JSON data.

        Args:
            data: Dictionary from nvfuser_dependencies.json with keys:
                - name: Dependency name
                - type: Dependency type (find_package, compiler, etc.)
                - cmake_vars: Dict of all CMake variables for this dependency
                - metadata: Dict of requirement metadata (VERSION_MIN, etc.)
        """
        self.name = data["name"]
        self.type = data["type"]
        cmake_vars = data.get("cmake_vars", {})
        metadata = data.get("metadata", {})

        # Extract common fields from cmake_vars
        self.found = cmake_vars.get(f"{self.name}_FOUND", False)
        self.status = cmake_vars.get(f"{self.name}_STATUS", "UNKNOWN")

        # Extract metadata
        self.optional = metadata.get(f"NVFUSER_REQUIREMENT_{self.name}_OPTIONAL", False)

        # Get location from location_var metadata
        location_var = metadata.get(f"NVFUSER_REQUIREMENT_{self.name}_LOCATION_VAR")
        if location_var:
            self.location = cmake_vars.get(location_var)
        else:
            self.location = None

        self._data = data  # Store for subclass access
        self._cmake_vars = cmake_vars
        self._metadata = metadata

    @abstractmethod
    def format_status_line(self, colors) -> str:
        """Format terminal status line for this requirement."""
        pass

    def is_failure(self) -> bool:
        """Check if this requirement represents a failure."""
        return not self.optional and self.status != RequirementStatus.SUCCESS

    def get_failure_data(self) -> Dict:
        """Get data for help text generation."""
        return self._data.copy()


class VersionRequirement(Requirement):
    """
    Base class for requirements with version checking.

    Provides standard version display formatting.
    Subclasses inherit all version comparison logic.
    """

    def __init__(self, data: Dict):
        super().__init__(data)

        # Extract version information
        self.version_found = self._cmake_vars.get(f"{self.name}_VERSION")
        self.version_required = self._metadata.get(f"NVFUSER_REQUIREMENT_{self.name}_VERSION_MIN")

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
