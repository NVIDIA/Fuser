#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "flatbuffers::flatc" for configuration "Release"
set_property(TARGET flatbuffers::flatc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(flatbuffers::flatc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/flatc"
  )

list(APPEND _cmake_import_check_targets flatbuffers::flatc )
list(APPEND _cmake_import_check_files_for_flatbuffers::flatc "${_IMPORT_PREFIX}/bin/flatc" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
