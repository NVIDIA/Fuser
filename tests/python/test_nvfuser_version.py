# SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import io
import sys
from unittest.mock import patch
from nvfuser.nvfuser_version import NvfuserVersion


class TestNvfuserVersionNewMethods:
    """Test cases for the newly added methods in NvfuserVersion class."""

    def test_say_something_to_version_output(self):
        """Test that _say_something_to_version prints the correct message."""
        version = NvfuserVersion("1.0.0")
        
        # Capture stdout to test the print output
        captured_output = io.StringIO()
        with patch('sys.stdout', captured_output):
            version._say_something_to_version()
        
        assert captured_output.getvalue().strip() == "Something to version!"

    def test_say_something_to_version_return_value(self):
        """Test that _say_something_to_version returns the correct value."""
        version = NvfuserVersion("1.0.0")
        result = version._say_something_to_version()
        assert result == "Something"

    def test_say_goodbye_to_version_output(self):
        """Test that _say_goodbye_to_version prints the correct message."""
        version = NvfuserVersion("1.0.0")
        
        # Capture stdout to test the print output
        captured_output = io.StringIO()
        with patch('sys.stdout', captured_output):
            version._say_goodbye_to_version()
        
        assert captured_output.getvalue().strip() == "Goodbye, version!"

    def test_say_goodbye_to_version_return_value(self):
        """Test that _say_goodbye_to_version returns None (no explicit return)."""
        version = NvfuserVersion("1.0.0")
        result = version._say_goodbye_to_version()
        assert result is None

    def test_all_new_methods_with_different_versions(self):
        """Test that all new methods work with different version strings."""
        test_versions = ["1.0.0", "2.1.3", "0.0.1", "10.20.30"]
        
        for version_str in test_versions:
            version = NvfuserVersion(version_str)
            
            # Test _say_something_to_version
            captured_output = io.StringIO()
            with patch('sys.stdout', captured_output):
                result = version._say_something_to_version()
            assert result == "Something"
            assert captured_output.getvalue().strip() == "Something to version!"
            
            # Test _say_goodbye_to_version
            captured_output = io.StringIO()
            with patch('sys.stdout', captured_output):
                result = version._say_goodbye_to_version()
            assert result is None
            assert captured_output.getvalue().strip() == "Goodbye, version!"

    def test_new_methods_inheritance_behavior(self):
        """Test that new methods work correctly with string inheritance."""
        version = NvfuserVersion("1.2.3")
        
        # Verify it's still a string
        assert isinstance(version, str)
        assert version == "1.2.3"
        
        # Test new methods still work
        result = version._say_something_to_version()
        assert result == "Something"
        
        captured_output = io.StringIO()
        with patch('sys.stdout', captured_output):
            version._say_goodbye_to_version()
        assert captured_output.getvalue().strip() == "Goodbye, version!"

    def test_new_methods_with_existing_functionality(self):
        """Test that new methods don't interfere with existing version comparison functionality."""
        version1 = NvfuserVersion("1.0.0")
        version2 = NvfuserVersion("2.0.0")
        
        # Test existing functionality still works
        assert version1 < version2
        assert version2 > version1
        assert version1 == NvfuserVersion("1.0.0")
        
        # Test new methods still work
        result = version1._say_something_to_version()
        assert result == "Something"
        
        captured_output = io.StringIO()
        with patch('sys.stdout', captured_output):
            version2._say_goodbye_to_version()
        assert captured_output.getvalue().strip() == "Goodbye, version!"

    def test_method_chaining_with_new_methods(self):
        """Test that new methods can be called in sequence."""
        version = NvfuserVersion("1.0.0")
        
        captured_output = io.StringIO()
        with patch('sys.stdout', captured_output):
            # Call both methods in sequence
            result1 = version._say_something_to_version()
            version._say_goodbye_to_version()
        
        output_lines = captured_output.getvalue().strip().split('\n')
        assert len(output_lines) == 2
        assert output_lines[0] == "Something to version!"
        assert output_lines[1] == "Goodbye, version!"
        assert result1 == "Something"

    def test_new_methods_with_edge_case_versions(self):
        """Test new methods with edge case version strings."""
        edge_cases = [
            "0.0.0",
            "999.999.999",
            "1.0.0+dev",
            "2.0.0-beta.1",
            "3.0.0-alpha.1+build.123"
        ]
        
        for version_str in edge_cases:
            version = NvfuserVersion(version_str)
            
            # Test both new methods work
            result = version._say_something_to_version()
            assert result == "Something"
            
            captured_output = io.StringIO()
            with patch('sys.stdout', captured_output):
                version._say_goodbye_to_version()
            assert captured_output.getvalue().strip() == "Goodbye, version!"


class TestNvfuserVersionIntegration:
    """Integration tests to ensure new methods work well with existing functionality."""

    def test_version_comparison_with_new_methods(self):
        """Test that version comparisons work correctly after calling new methods."""
        version1 = NvfuserVersion("1.0.0")
        version2 = NvfuserVersion("2.0.0")
        
        # Call new methods
        version1._say_something_to_version()
        version2._say_goodbye_to_version()
        
        # Verify comparisons still work
        assert version1 < version2
        assert version2 > version1
        assert version1 != version2

    def test_version_string_operations_with_new_methods(self):
        """Test that string operations work correctly after calling new methods."""
        version = NvfuserVersion("1.2.3")
        
        # Call new methods
        version._say_something_to_version()
        version._say_goodbye_to_version()
        
        # Verify string operations still work
        assert len(version) == 5  # "1.2.3"
        assert version.startswith("1")
        assert version.endswith("3")
        assert "2" in version


if __name__ == "__main__":
    pytest.main([__file__])
