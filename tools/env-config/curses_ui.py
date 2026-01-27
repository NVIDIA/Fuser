#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Curses-based TUI for nvFuser environment configuration.
Provides a ccmake-like interface for navigating and configuring options.
"""

import curses


class CursesUI:
    """Curses-based UI for environment configuration."""

    def __init__(self, stdscr, config):
        self.stdscr = stdscr
        self.config = config
        self.current_row = 0
        self.top_row = 0
        self.modified = False
        self.should_exit = False

        # Build flat list of all options with category headers
        self.display_items = []
        category_names = {
            "build": "Build Configuration (NVFUSER_BUILD_*)",
            "build_advanced": "Advanced Build Options (NVFUSER_BUILD_*)",
            "environment": "Environment & Compiler Settings (CC, CXX, CUDA_HOME, etc.)",
            "dump": "Debug/Diagnostic Options (NVFUSER_DUMP)",
            "enable": "Feature Enable Options (NVFUSER_ENABLE)",
            "disable": "Feature Disable Options (NVFUSER_DISABLE)",
            "profiler": "Profiler Options (NVFUSER_PROF)",
            "compilation": "Runtime Compilation Control",
        }

        for category, opts in config.categories.items():
            # Add category header
            self.display_items.append(
                {
                    "type": "header",
                    "text": category_names.get(category, category.upper()),
                }
            )
            # Add options
            for opt in opts:
                self.display_items.append(
                    {
                        "type": "option",
                        "option": opt,
                    }
                )

        # Initialize curses
        curses.curs_set(0)  # Hide cursor
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Highlight
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Header
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Enabled/Set
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Modified

    def draw_header(self):
        """Draw the top header."""
        height, width = self.stdscr.getmaxyx()
        self.stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        header = "nvFuser Environment Configuration"
        self.stdscr.addstr(0, (width - len(header)) // 2, header)
        self.stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

    def draw_footer(self):
        """Draw the bottom help text."""
        height, width = self.stdscr.getmaxyx()
        help_text = (
            "[↑↓] Navigate  [Enter] Toggle  [e] Edit  [a] Apply  [g] Generate  [q] Quit"
        )

        if self.modified:
            help_text = "[MODIFIED] " + help_text

        self.stdscr.attron(curses.color_pair(2))
        self.stdscr.addstr(height - 1, 0, help_text.ljust(width - 1))
        self.stdscr.attroff(curses.color_pair(2))

    def draw_option(self, y: int, item: dict, is_selected: bool):
        """Draw a single option or header."""
        height, width = self.stdscr.getmaxyx()

        if y < 2 or y >= height - 1:
            return  # Skip if outside visible area

        if item["type"] == "header":
            # Draw category header
            attr = curses.color_pair(2) | curses.A_BOLD
            if is_selected:
                attr |= curses.A_REVERSE
            self.stdscr.attron(attr)
            self.stdscr.addstr(y, 0, ("  " + item["text"]).ljust(width - 1))
            self.stdscr.attroff(attr)

        elif item["type"] == "option":
            opt = item["option"]
            base_attr = 0

            if is_selected:
                base_attr |= curses.A_REVERSE

            # Format the option display
            prefix = "  [" if opt.var_type == "bool" else "    "
            status = ""

            if opt.var_type == "bool":
                # Boolean option with checkbox
                if opt.current_value == "1":
                    status = "X"
                    attr = base_attr | curses.color_pair(3)  # Green for enabled
                else:
                    status = " "
                    attr = base_attr
                checkbox = f"{prefix}{status}] "
                name_part = opt.name

                line = f"{checkbox}{name_part}"
                max_len = width - 1
                if len(line) > max_len:
                    line = line[: max_len - 3] + "..."

                self.stdscr.attron(attr)
                self.stdscr.addstr(y, 0, line.ljust(width - 1))
                self.stdscr.attroff(attr)

            else:
                # String/int/multi option with value display
                name_part = f"{prefix}{opt.name}"

                # Draw name part (no color)
                self.stdscr.attron(base_attr)
                self.stdscr.addstr(y, 0, name_part)

                # Draw value part with color if set
                if opt.current_value:
                    value_part = f" = {opt.current_value}"
                    value_attr = base_attr | curses.color_pair(
                        3
                    )  # Green for set values
                else:
                    value_part = " = (not set)"
                    value_attr = base_attr

                # Calculate remaining space
                remaining_space = width - 1 - len(name_part)
                if len(value_part) > remaining_space:
                    value_part = value_part[: remaining_space - 3] + "..."

                self.stdscr.attroff(base_attr)
                self.stdscr.attron(value_attr)
                self.stdscr.addstr(y, len(name_part), value_part)

                # Pad the rest of the line
                total_len = len(name_part) + len(value_part)
                if total_len < width - 1:
                    self.stdscr.addstr(y, total_len, " " * (width - 1 - total_len))
                self.stdscr.attroff(value_attr)

    def draw_description(self):
        """Draw description of currently selected item."""
        height, width = self.stdscr.getmaxyx()

        if self.current_row >= len(self.display_items):
            return

        item = self.display_items[self.current_row]

        # Draw description area separator
        desc_start = height - 5
        self.stdscr.hline(desc_start, 0, curses.ACS_HLINE, width)

        if item["type"] == "option":
            opt = item["option"]
            desc_y = desc_start + 1

            # Wrap description text
            desc_text = f"Description: {opt.description}"
            words = desc_text.split()
            lines = []
            current_line = ""

            for word in words:
                if len(current_line) + len(word) + 1 <= width - 2:
                    current_line += word + " "
                else:
                    lines.append(current_line.strip())
                    current_line = word + " "
            if current_line:
                lines.append(current_line.strip())

            # Draw description lines
            for i, line in enumerate(lines[:3]):  # Max 3 lines
                if desc_y + i < height - 1:
                    self.stdscr.addstr(desc_y + i, 1, line)

    def draw(self):
        """Draw the entire UI."""
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()

        self.draw_header()

        # Draw options list
        visible_rows = height - 7  # Leave space for header, description, footer
        for i, item in enumerate(
            self.display_items[self.top_row : self.top_row + visible_rows]
        ):
            self.draw_option(i + 2, item, i + self.top_row == self.current_row)

        self.draw_description()
        self.draw_footer()

        self.stdscr.refresh()

    def handle_toggle(self):
        """Handle toggling current item."""
        if self.current_row >= len(self.display_items):
            return

        item = self.display_items[self.current_row]

        if item["type"] == "option":
            opt = item["option"]

            if opt.var_type == "bool":
                # Toggle boolean
                opt.current_value = "1" if opt.current_value != "1" else None
                self.modified = True

            elif opt.var_type == "multi":
                # Cycle through choices
                if not opt.current_value or opt.current_value not in opt.choices:
                    opt.current_value = opt.choices[0] if opt.choices else ""
                else:
                    idx = opt.choices.index(opt.current_value)
                    opt.current_value = opt.choices[(idx + 1) % len(opt.choices)]
                self.modified = True

    def handle_edit(self):
        """Handle editing current item value."""
        if self.current_row >= len(self.display_items):
            return

        item = self.display_items[self.current_row]

        if item["type"] == "option":
            opt = item["option"]

            if opt.var_type in ["int", "string", "multi"]:
                # Show input dialog
                height, width = self.stdscr.getmaxyx()
                input_y = height // 2

                # Draw input box
                self.stdscr.attron(curses.color_pair(1))
                prompt = f"Enter value for {opt.name}: "
                self.stdscr.addstr(input_y, 2, " " * (width - 4))
                self.stdscr.addstr(input_y, 2, prompt)
                self.stdscr.attroff(curses.color_pair(1))

                # Get input
                curses.echo()
                curses.curs_set(1)
                try:
                    input_str = self.stdscr.getstr(
                        input_y, 2 + len(prompt), width - 4 - len(prompt)
                    )
                    value = input_str.decode("utf-8").strip()
                    if value:
                        opt.current_value = value
                        self.modified = True
                except InputError:
                    pass
                finally:
                    curses.noecho()
                    curses.curs_set(0)

    def show_confirmation(self, message: str, prompt: str) -> str:
        """Show a simple y/n confirmation dialog

        Returns the character pressed by the user (y/n/etc).
        """
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()

        # Center the message
        y = height // 2
        x = max(0, (width - len(message)) // 2)
        self.stdscr.addstr(y, x, message, curses.A_BOLD)

        # Show prompt below
        y += 2
        x = max(0, (width - len(prompt)) // 2)
        self.stdscr.addstr(y, x, prompt)

        self.stdscr.refresh()

        # Get single character response
        response = self.stdscr.getch()
        if response == 27:  # Escape
            return "n"
        return chr(response) if response < 256 else "n"

    def show_message(
        self, message: str, submessage: str = "", wait_for_key: bool = True
    ):
        """Show a simple message dialog"""
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()

        # Center the message
        y = height // 2
        x = max(0, (width - len(message)) // 2)
        self.stdscr.addstr(y, x, message, curses.A_BOLD)

        # Show submessage below
        if submessage:
            y += 2
            x = max(0, (width - len(submessage)) // 2)
            self.stdscr.addstr(y, x, submessage)

        if wait_for_key:
            y += 2
            prompt = "Press any key to continue..."
            x = max(0, (width - len(prompt)) // 2)
            self.stdscr.addstr(y, x, prompt)

        self.stdscr.refresh()

        if wait_for_key:
            self.stdscr.getch()

    def handle_apply_now(self):
        """Show confirmation, generate apply script, and signal to exit"""
        # Import save_config here to avoid circular import
        from configure_env import save_config

        # Show confirmation dialog
        response = self.show_confirmation("Apply configuration and exit?", "[y/N]")

        if response not in ["y", "Y"]:
            return  # User cancelled

        # Get exports and unsets
        exports = self.config.get_env_exports()
        unsets = self.config.get_unset_vars()

        # Generate apply script
        apply_script = "/tmp/nvfuser_apply_now.sh"
        save_config(exports, unsets, apply_script)

        # Set flag to exit
        self.should_exit = True
        self.modified = False  # Prevent quit prompt

    def handle_generate(self):
        """Prompt for filename and generate script"""
        # Import save_config here to avoid circular import
        from configure_env import save_config

        # Simple text input prompt
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()

        # Show prompt
        prompt = "Enter filename [nvfuser_env.sh]: "
        self.stdscr.addstr(height // 2, 2, prompt, curses.A_BOLD)
        self.stdscr.refresh()

        # Enable echo for input
        curses.echo()
        curses.curs_set(1)

        # Get input (basic - no arrow key editing, but backspace works naturally)
        try:
            filename_input = (
                self.stdscr.getstr(height // 2, 2 + len(prompt), 50)
                .decode("utf-8")
                .strip()
            )
        finally:
            curses.noecho()
            curses.curs_set(0)

        # Use default if empty
        if not filename_input:
            filename = "nvfuser_env.sh"
        else:
            filename = filename_input

        # Get exports and unsets
        exports = self.config.get_env_exports()
        unsets = self.config.get_unset_vars()

        # Generate script
        save_config(exports, unsets, filename)

        # Show confirmation dialog
        self.show_message(
            f"Configuration saved to {filename}",
            f"To apply: source {filename}",
            wait_for_key=True,
        )

        # Exit
        self.should_exit = True
        self.modified = False

    def run(self):
        """Main event loop."""
        while True:
            self.draw()

            key = self.stdscr.getch()

            if key == ord("q") or key == ord("Q"):
                # Quit
                if self.modified:
                    # Ask what to do with changes
                    height, width = self.stdscr.getmaxyx()
                    msg_y = height // 2 - 3

                    self.stdscr.clear()
                    self.stdscr.attron(curses.color_pair(1))

                    title = "You have unsaved changes!"
                    self.stdscr.addstr(msg_y, (width - len(title)) // 2, title)

                    self.stdscr.attroff(curses.color_pair(1))

                    # Show options
                    options = [
                        "",
                        "What would you like to do?",
                        "",
                        "  [a] Apply Now    - Generate /tmp/nvfuser_apply_now.sh",
                        "                     (then source it after exit)",
                        "",
                        "  [g] Generate     - Create nvfuser_env.sh",
                        "                     (then source it after exit)",
                        "",
                        "  [q] Quit         - Exit without saving",
                        "",
                        "  [Esc] Cancel     - Return to configuration",
                    ]

                    for i, line in enumerate(options):
                        if line.startswith("  ["):
                            # Highlight the option keys
                            self.stdscr.attron(curses.A_BOLD)
                        self.stdscr.addstr(
                            msg_y + 2 + i, (width - len(line)) // 2, line
                        )
                        if line.startswith("  ["):
                            self.stdscr.attroff(curses.A_BOLD)

                    self.stdscr.refresh()

                    # Get response
                    response = self.stdscr.getch()

                    if response == ord("a") or response == ord("A"):
                        self.handle_apply_now()
                        break
                    elif response == ord("g") or response == ord("G"):
                        self.handle_generate()
                        break
                    elif response == ord("q") or response == ord("Q"):
                        break
                    elif response == 27:  # Escape
                        continue  # Return to main loop
                    else:
                        # Invalid key, show again
                        continue
                else:
                    # No changes, just quit
                    break

            elif key == curses.KEY_UP:
                if self.current_row > 0:
                    self.current_row -= 1
                    # Adjust scroll if needed
                    if self.current_row < self.top_row:
                        self.top_row = self.current_row

            elif key == curses.KEY_DOWN:
                if self.current_row < len(self.display_items) - 1:
                    self.current_row += 1
                    # Adjust scroll if needed
                    height, width = self.stdscr.getmaxyx()
                    visible_rows = height - 7
                    if self.current_row >= self.top_row + visible_rows:
                        self.top_row = self.current_row - visible_rows + 1

            elif key == ord(" ") or key == ord("\n") or key == curses.KEY_ENTER:
                self.handle_toggle()

            elif key == ord("e") or key == ord("E"):
                self.handle_edit()

            elif key == ord("a") or key == ord("A"):
                self.handle_apply_now()
                if self.should_exit:
                    break  # Exit immediately after apply

            elif key == ord("g") or key == ord("G"):
                self.handle_generate()
                if self.should_exit:
                    break  # Exit immediately after generate


def run_curses_ui(stdscr, config):
    """Entry point for curses UI."""
    ui = CursesUI(stdscr, config)
    ui.run()
