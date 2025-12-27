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
