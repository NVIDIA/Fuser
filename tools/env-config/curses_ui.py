#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Curses-based TUI for nvFuser environment configuration.
Provides a ccmake-like interface for navigating and configuring options.
"""

import curses
from configure_env import CATEGORY_NAMES


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

        # Display categories in the order defined in CATEGORY_NAMES
        for category in CATEGORY_NAMES.keys():
            if category not in config.categories:
                continue
            opts = config.categories[category]
            # Add category header
            self.display_items.append(
                {
                    "type": "header",
                    "text": CATEGORY_NAMES[category],
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
        curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)  # Multi tag

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

        # Show context-sensitive help based on current selection
        item = (
            self.display_items[self.current_row]
            if self.current_row < len(self.display_items)
            else None
        )

        if item and item["type"] == "option":
            opt = item["option"]
            if opt.var_type == "bool":
                help_text = "[↑↓/jk] Nav  [{}] Sect  [PgUp/PgDn] Top/Bot  [Enter] Toggle  [a] Apply  [g] Gen  [q] Quit"
            elif opt.var_type == "multi":
                help_text = "[↑↓/jk] Nav  [{}] Sect  [PgUp/PgDn] Top/Bot  [Enter] Cycle  [a] Apply  [g] Gen  [q] Quit"
            else:
                # String/Int - Enter to edit
                help_text = "[↑↓/jk] Nav  [{}] Sect  [PgUp/PgDn] Top/Bot  [Enter] Edit  [a] Apply  [g] Gen  [q] Quit"
        else:
            help_text = "[↑↓/jk] Nav  [{}] Sect  [PgUp/PgDn] Top/Bot  [Enter] Toggle  [a] Apply  [g] Gen  [q] Quit"

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
            # All options get a checkbox at the beginning showing if they're set
            prefix = "   "  # 3 spaces for better visual separation

            if opt.var_type == "bool":
                # Boolean option - checkbox shows enabled/disabled status
                name_part = opt.get_display_name()

                if opt.current_value == "1":
                    checkbox = "[X] "
                    attr = base_attr | curses.color_pair(3)  # Green for enabled
                else:
                    checkbox = "[ ] "
                    attr = base_attr

                line = f"{prefix}{checkbox}{name_part}"
                max_len = width - 1
                if len(line) > max_len:
                    line = line[: max_len - 3] + "..."

                self.stdscr.attron(attr)
                self.stdscr.addstr(y, 0, line.ljust(width - 1))
                self.stdscr.attroff(attr)

            else:
                # String/int/multi option with value display and checkbox
                display_name = opt.get_display_name()

                # Determine if this option has a value (affects checkbox and color)
                has_value = opt.current_value is not None
                line_attr = base_attr | curses.color_pair(3) if has_value else base_attr

                # Add checkbox at the beginning - checked if value is set
                if has_value:
                    checkbox = "[X] "
                else:
                    checkbox = "[ ] "

                name_part = f"{prefix}{checkbox}{display_name}"

                # Draw name part with checkbox (green if value is set)
                self.stdscr.attron(line_attr)
                self.stdscr.addstr(y, 0, name_part)
                self.stdscr.attroff(line_attr)

                # Add [multi] tag in yellow for multi-choice options
                multi_tag_len = 0
                if opt.var_type == "multi":
                    multi_tag = " [multi]"
                    multi_tag_len = len(multi_tag)
                    multi_attr = base_attr | curses.color_pair(4)  # Yellow
                    self.stdscr.attron(multi_attr)
                    self.stdscr.addstr(y, len(name_part), multi_tag)
                    self.stdscr.attroff(multi_attr)

                # Draw value part (green if set)
                if opt.current_value:
                    value_part = f" = {opt.current_value}"
                else:
                    # Show = "" for empty string/int/multi to indicate they expect values
                    value_part = ' = ""'

                # Calculate remaining space
                total_prefix = len(name_part) + multi_tag_len
                remaining_space = width - 1 - total_prefix
                if len(value_part) > remaining_space:
                    value_part = value_part[: remaining_space - 3] + "..."

                self.stdscr.attron(line_attr)
                self.stdscr.addstr(y, total_prefix, value_part)

                # Pad the rest of the line
                total_len = total_prefix + len(value_part)
                if total_len < width - 1:
                    self.stdscr.addstr(y, total_len, " " * (width - 1 - total_len))
                self.stdscr.attroff(line_attr)

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

            # Add choices for multi options
            if opt.var_type == "multi" and opt.choices:
                desc_text += f" | Choices: {', '.join(repr(c) for c in opt.choices)}"

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
        """Handle toggling/editing current item."""
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
                # Cycle through choices, including unset at the end
                # Note: Some choices may include "" (empty string) as a valid choice
                if opt.current_value is None:
                    # Currently unset, go to first choice
                    opt.current_value = opt.choices[0] if opt.choices else ""
                elif opt.current_value not in opt.choices:
                    # Invalid value, go to first choice
                    opt.current_value = opt.choices[0] if opt.choices else ""
                else:
                    # Valid value, cycle to next (or back to None after last)
                    idx = opt.choices.index(opt.current_value)
                    if idx == len(opt.choices) - 1:
                        # Last choice, cycle back to unset (None)
                        opt.current_value = None
                    else:
                        # Go to next choice
                        opt.current_value = opt.choices[idx + 1]
                self.modified = True

            elif opt.var_type in ["int", "string"]:
                # For string/int, Enter opens the editor
                self.handle_edit()

    def handle_edit(self):
        """Handle editing current item value."""
        if self.current_row >= len(self.display_items):
            return

        item = self.display_items[self.current_row]

        if item["type"] == "option":
            opt = item["option"]

            if opt.var_type in ["int", "string", "multi"]:
                height, width = self.stdscr.getmaxyx()

                # Calculate the Y position of the current row on screen
                # Rows start at y=2 (after header), and we need to account for scroll
                screen_y = 2 + (self.current_row - self.top_row)

                # Make sure we're within visible area
                if screen_y < 2 or screen_y >= height - 5:
                    # Fall back to center if somehow out of range
                    screen_y = height // 2

                # Build the display name with [multi] tag if applicable
                display_name = opt.get_display_name()
                if opt.var_type == "multi":
                    display_name += " [multi]"

                # Get current value to pre-fill
                current_val = opt.current_value if opt.current_value else ""

                # Create prompt: "name = |cursor here|"
                prefix = "    "
                prompt_text = f"{prefix}{display_name} = "

                # Clear the line and show the prompt
                self.stdscr.move(screen_y, 0)
                self.stdscr.clrtoeol()
                self.stdscr.addstr(screen_y, 0, prompt_text, curses.A_BOLD)

                # Pre-fill with current value
                if current_val:
                    self.stdscr.addstr(screen_y, len(prompt_text), current_val)

                self.stdscr.refresh()

                # Position cursor at the end of the current value
                cursor_pos = len(prompt_text) + len(current_val)
                self.stdscr.move(screen_y, cursor_pos)

                # Enable cursor and echo
                curses.curs_set(1)
                curses.echo()

                try:
                    # Manual text editing with basic line editing support
                    buffer = list(current_val)  # Start with current value
                    cursor_offset = len(buffer)  # Cursor at end
                    max_len = width - len(prompt_text) - 2

                    while True:
                        # Display current buffer
                        self.stdscr.move(screen_y, len(prompt_text))
                        self.stdscr.clrtoeol()
                        display_text = "".join(buffer)
                        if len(display_text) > max_len:
                            display_text = display_text[:max_len]
                        self.stdscr.addstr(screen_y, len(prompt_text), display_text)
                        self.stdscr.move(screen_y, len(prompt_text) + cursor_offset)
                        self.stdscr.refresh()

                        # Get key
                        key = self.stdscr.getch()

                        if key == ord("\n") or key == curses.KEY_ENTER:
                            # Accept input
                            value = "".join(buffer).strip()
                            if value or opt.current_value:  # Allow clearing
                                opt.current_value = value if value else None
                                self.modified = True
                            break
                        elif key == 27:  # Escape
                            # Cancel editing
                            break
                        elif key == curses.KEY_BACKSPACE or key == 127 or key == 8:
                            # Backspace
                            if cursor_offset > 0:
                                buffer.pop(cursor_offset - 1)
                                cursor_offset -= 1
                        elif key == curses.KEY_DC:
                            # Delete key
                            if cursor_offset < len(buffer):
                                buffer.pop(cursor_offset)
                        elif key == curses.KEY_LEFT:
                            # Move cursor left
                            if cursor_offset > 0:
                                cursor_offset -= 1
                        elif key == curses.KEY_RIGHT:
                            # Move cursor right
                            if cursor_offset < len(buffer):
                                cursor_offset += 1
                        elif key == curses.KEY_HOME or key == 1:  # Ctrl-A
                            # Move to start
                            cursor_offset = 0
                        elif key == curses.KEY_END or key == 5:  # Ctrl-E
                            # Move to end
                            cursor_offset = len(buffer)
                        elif key == 21:  # Ctrl-U
                            # Clear line
                            buffer = []
                            cursor_offset = 0
                        elif 32 <= key <= 126:
                            # Printable character
                            if len(buffer) < max_len:
                                buffer.insert(cursor_offset, chr(key))
                                cursor_offset += 1

                except Exception:
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

    def jump_to_next_section(self):
        """Jump to the next section header."""
        for i in range(self.current_row + 1, len(self.display_items)):
            if self.display_items[i]["type"] == "header":
                self.current_row = i
                # Adjust scroll if needed
                height, _ = self.stdscr.getmaxyx()
                visible_rows = height - 7
                if self.current_row >= self.top_row + visible_rows:
                    self.top_row = self.current_row - visible_rows + 1
                break

    def jump_to_prev_section(self):
        """Jump to the previous section header."""
        for i in range(self.current_row - 1, -1, -1):
            if self.display_items[i]["type"] == "header":
                self.current_row = i
                # Adjust scroll if needed
                if self.current_row < self.top_row:
                    self.top_row = self.current_row
                break

    def jump_to_top(self):
        """Jump to the first item."""
        self.current_row = 0
        self.top_row = 0

    def jump_to_bottom(self):
        """Jump to the last item."""
        self.current_row = len(self.display_items) - 1
        # Adjust scroll to show bottom
        height, _ = self.stdscr.getmaxyx()
        visible_rows = height - 7
        self.top_row = max(0, self.current_row - visible_rows + 1)

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

            # Vim-style navigation
            elif key == ord("k") or key == ord("K"):
                # Up (same as arrow up)
                if self.current_row > 0:
                    self.current_row -= 1
                    if self.current_row < self.top_row:
                        self.top_row = self.current_row

            elif key == ord("j") or key == ord("J"):
                # Down (same as arrow down)
                if self.current_row < len(self.display_items) - 1:
                    self.current_row += 1
                    height, width = self.stdscr.getmaxyx()
                    visible_rows = height - 7
                    if self.current_row >= self.top_row + visible_rows:
                        self.top_row = self.current_row - visible_rows + 1

            elif key == ord("}"):
                # Jump to next section
                self.jump_to_next_section()

            elif key == ord("{"):
                # Jump to previous section
                self.jump_to_prev_section()

            elif key == curses.KEY_PPAGE:
                # Page Up - jump to top
                self.jump_to_top()

            elif key == curses.KEY_NPAGE:
                # Page Down - jump to bottom
                self.jump_to_bottom()

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
