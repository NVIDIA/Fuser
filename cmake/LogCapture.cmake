# ==============================================================================
# nvFuser Log Capture Utilities
# ==============================================================================

# Global switch to control logging behavior
set(LOG_CAPTURE_MODE FALSE CACHE INTERNAL "")
set(GLOBAL_LOG_BUFFER "" CACHE INTERNAL "")

# 1. Override message() ONCE for the whole project.
#    This acts as a "Gatekeeper" for BOTH capture and suppression.
function(message)
    # Handle empty lines
    if(NOT ARGV)
        if(LOG_CAPTURE_MODE)
            # Store empty line marker when capturing
            set_property(GLOBAL APPEND PROPERTY GLOBAL_LOG_BUFFER "EMPTY_LINE")
        elseif(NOT SUPPRESS_MESSAGE_OUTPUT)
            # Print empty line when not suppressing
            _message("")
        endif()
        return()
    endif()

    # Get the message type (STATUS, WARNING, FATAL_ERROR, etc.)
    list(GET ARGV 0 type)

    # Pass through FATAL_ERROR and SEND_ERROR immediately (Fail Fast)
    if(type STREQUAL "FATAL_ERROR" OR type STREQUAL "SEND_ERROR")
        _message(${ARGV})
        return()
    endif()

    # Logic: Capture, Suppress, or Print?
    if(LOG_CAPTURE_MODE)
        # CAPTURE MODE: Store type and content separately
        # Remove the type from ARGV to get just the content
        set(_argv_copy ${ARGV})
        list(REMOVE_AT _argv_copy 0)
        string(JOIN " " msg_content ${_argv_copy})
        # Use a delimiter (|||) to separate type from content
        set_property(GLOBAL APPEND PROPERTY GLOBAL_LOG_BUFFER "${type}|||${msg_content}")
    elseif(SUPPRESS_MESSAGE_OUTPUT)
        # SUPPRESS MODE: Block all non-critical messages (already handled errors above)
        # Do nothing
    else()
        # NORMAL MODE: Pass through to internal CMake message
        _message(${ARGV})
    endif()
endfunction()

# 2. Macros to control the switch
macro(start_capture)
    set(LOG_CAPTURE_MODE TRUE)
    set_property(GLOBAL PROPERTY GLOBAL_LOG_BUFFER "") # Clear buffer
endmacro()

macro(stop_capture target_var)
    set(LOG_CAPTURE_MODE FALSE)
    # Move global buffer to user variable
    get_property(_logs GLOBAL PROPERTY GLOBAL_LOG_BUFFER)
    set(${target_var} "${_logs}")
endmacro()

# 3. Helper to print the logs later
function(dump_captured_logs log_list)
    foreach(entry ${log_list})
        if("${entry}" STREQUAL "EMPTY_LINE")
            _message("")
        else()
            # Split "TYPE|||CONTENT"
            string(FIND "${entry}" "|||" pos)
            string(SUBSTRING "${entry}" 0 ${pos} type)
            math(EXPR content_start "${pos} + 3")
            string(SUBSTRING "${entry}" ${content_start} -1 content)

            # Print using the original type (STATUS, WARNING, etc.)
            # This preserves color and formatting!
            _message(${type} "${content}")
        endif()
    endforeach()
endfunction()

# Macro to run a command and capture its output to a variable
macro(execute_with_log target_log_var)
  # Define a unique global storage key
  set(_storage_key "_LOG_STORE_${target_log_var}")
  set_property(GLOBAL PROPERTY "${_storage_key}" "")

  # Override 'message' to capture instead of print
  function(message)
    if(NOT ARGV)
      set_property(GLOBAL APPEND PROPERTY "${_storage_key}" "EMPTY_LINE")
      return()
    endif()

    # Check for message type (STATUS, WARNING, etc.)
    list(GET ARGV 0 type)
    set(valid_types STATUS WARNING AUTHOR_WARNING SEND_ERROR FATAL_ERROR DEPRECATION NOTICE)

    if(type IN_LIST valid_types)
      list(REMOVE_AT ARGV 0)
      set(msg_type "${type}")
    else()
      set(msg_type "NOTICE")
    endif()

    # Join arguments and store
    string(JOIN " " msg_content ${ARGV})
    set_property(GLOBAL APPEND PROPERTY "${_storage_key}" "${msg_type}|||${msg_content}")

    # Pass hard errors through immediately
    if(msg_type STREQUAL "FATAL_ERROR")
      _message(FATAL_ERROR "${msg_content}")
    endif()
  endfunction()

  # Execute the command passed in ${ARGN}
  cmake_language(EVAL CODE "${ARGN}")

  # Restore original message function
  function(message)
    _message(${ARGV})
  endfunction()

  # Retrieve logs and clean up
  get_property(_captured GLOBAL PROPERTY "${_storage_key}")
  set_property(GLOBAL PROPERTY "${_storage_key}" "")
  set(${target_log_var} "${_captured}")
endmacro()

# Helper function to print the captured logs prettily
function(print_captured_logs log_list prefix)
  message(STATUS "${prefix} Detailed Logs:")
  foreach(entry ${log_list})
    if("${entry}" STREQUAL "EMPTY_LINE")
      message("")
    else()
      # Parse "TYPE|||CONTENT"
      string(FIND "${entry}" "|||" pos)
      string(SUBSTRING "${entry}" 0 ${pos} type)
      math(EXPR content_start "${pos} + 3")
      string(SUBSTRING "${entry}" ${content_start} -1 content)

      # Print with indentation
      message(${type} "    | ${content}")
    endif()
  endforeach()
  message(STATUS "${prefix} End Logs.")
endfunction()

# --- MESSAGE SUPPRESSION MACROS ---
# Note: The message() override above already handles both capture and suppression
# These macros just control the SUPPRESS_MESSAGE_OUTPUT flag

# 2. Macros to toggle suppression
macro(mute_messages)
  set(SUPPRESS_MESSAGE_OUTPUT TRUE)
endmacro()

macro(unmute_messages)
  unset(SUPPRESS_MESSAGE_OUTPUT)
endmacro()

# --- END MESSAGE SUPPRESSION MACROS ---


