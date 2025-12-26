# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Requirement class hierarchy for dependency reporting.

This module provides OOP abstractions for different types of requirements:
- VersionRequirement: Dependencies with version checks (Python, LLVM, etc.)
- BooleanRequirement: Dependencies without versions (Git submodules, Ninja)
- ConstraintRequirement: Pseudo-requirements that check constraints (Torch_CUDA)

Each requirement knows how to:
1. Format its status line for display
2. Determine if it represents a failure
3. Provide data for help text generation
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class RequirementStatus:
    """Represents the validation status of a requirement."""
    SUCCESS = "SUCCESS"
    NOT_FOUND = "NOT_FOUND"
    INCOMPATIBLE = "INCOMPATIBLE"


class Requirement(ABC):
    """Base class for all requirement types."""

    def __init__(self, data: Dict):
        """
        Initialize requirement from JSON data.

        Args:
            data: Dictionary from nvfuser_dependencies.json containing:
                - name: Dependency name
                - type: Dependency type (find_package, compiler, etc.)
                - found: Whether dependency was found
                - status: Validation status (SUCCESS, NOT_FOUND, INCOMPATIBLE)
                - optional: Whether dependency is optional
                - location: Where dependency was found (if applicable)
        """
        self.name = data["name"]
        self.type = data["type"]
        self.found = data["found"]
        self.status = data["status"]
        self.optional = data.get("optional", False)
        self.location = data.get("location")
        self._data = data  # Store for subclass access

    @abstractmethod
    def format_status_line(self, colors) -> str:
        """
        Format the status line for this requirement.

        Args:
            colors: Colors instance for terminal formatting

        Returns:
            Formatted status line string
        """
        pass

    def is_failure(self) -> bool:
        """
        Check if this requirement represents a failure.

        Returns:
            True if this is a required dependency that failed
        """
        return not self.optional and self.status != RequirementStatus.SUCCESS

    def get_failure_data(self) -> Dict:
        """
        Get data for help text generation.

        Returns:
            Dictionary with failure information for help providers
        """
        return self._data.copy()


class VersionRequirement(Requirement):
    """
    Requirement with version checking (Python, LLVM, pybind11, etc.).

    Displays version information in status line.
    """

    def __init__(self, data: Dict):
        super().__init__(data)
        self.version_found = data.get("version_found")
        self.version_required = data.get("version_required")

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
        """Format success line: [nvFuser] ✓ Python 3.12.3 >= 3.8"""
        if self.version_found and self.version_required:
            return f"{colors.GREEN}[nvFuser] ✓ {self.name} {self.version_found} >= {self.version_required}{colors.RESET}"
        elif self.version_found:
            return f"{colors.GREEN}[nvFuser] ✓ {self.name} {self.version_found}{colors.RESET}"
        else:
            return f"{colors.GREEN}[nvFuser] ✓ {self.name}{colors.RESET}"

    def _format_not_found(self, colors) -> str:
        """Format not found line."""
        if self.optional:
            # Yellow ○ for optional
            if self.version_required:
                return f"{colors.YELLOW}[nvFuser] ○ {self.name} NOT found (optional, v{self.version_required}+ recommended){colors.RESET}"
            else:
                return f"{colors.YELLOW}[nvFuser] ○ {self.name} NOT found (optional){colors.RESET}"
        else:
            # Red ✗ for required
            if self.version_required:
                return f"{colors.BOLD_RED}[nvFuser] ✗ {self.name} NOT found (requires {self.version_required}+){colors.RESET}"
            else:
                return f"{colors.BOLD_RED}[nvFuser] ✗ {self.name} NOT found{colors.RESET}"

    def _format_incompatible(self, colors) -> str:
        """Format incompatible line: [nvFuser] ✗ Python 3.7.0 < 3.8"""
        if self.version_found and self.version_required:
            return f"{colors.BOLD_RED}[nvFuser] ✗ {self.name} {self.version_found} < {self.version_required}{colors.RESET}"
        else:
            return f"{colors.BOLD_RED}[nvFuser] ✗ {self.name} incompatible{colors.RESET}"


class BooleanRequirement(Requirement):
    """
    Requirement without version checking (Git submodules, Ninja).

    Simple pass/fail validation with no version info.
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


def create_requirement(data: Dict) -> Requirement:
    """
    Factory function to create appropriate Requirement subclass from JSON data.

    Uses the 'class_type' field from JSON (defined in CMake) to determine
    which class to instantiate.

    Args:
        data: Dictionary from nvfuser_dependencies.json

    Returns:
        Appropriate Requirement subclass instance
    """
    class_type = data.get("class_type", "version")

    if class_type == "torch":
        # Torch with CUDA constraint checking
        try:
            from prereqs.help.torch import TorchRequirement
            return TorchRequirement(data)
        except ImportError:
            # Fallback to VersionRequirement if torch module not available
            return VersionRequirement(data)

    elif class_type == "boolean":
        # Boolean requirements (no version checking)
        return BooleanRequirement(data)

    elif class_type == "version":
        # Standard version requirements (Python, LLVM, pybind11, etc.)
        return VersionRequirement(data)

    else:
        # Unknown class type - default to version requirement
        return VersionRequirement(data)
