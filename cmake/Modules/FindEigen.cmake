SET(EIGEN_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/eigen3
  /usr/local/include
  /usr/local/include/eigen3
  $ENV{EIGEN_HOME}
)

FIND_PATH(EIGEN_INCLUDE_DIR NAMES Eigen/Dense PATHS ${EIGEN_INCLUDE_SEARCH_PATHS})

SET(EIGEN_FOUND ON)

#    Check include files
IF(NOT EIGEN_INCLUDE_DIR)
    SET(EIGEN_FOUND OFF)
    MESSAGE(STATUS "Could not find EIGEN include. Turning EIGEN_FOUND off")
ENDIF()

IF (EIGEN_FOUND)
  IF (NOT EIGEN_FIND_QUIETLY)
    MESSAGE(STATUS "Found EIGEN include: ${EIGEN_INCLUDE_DIR}")
  ENDIF (NOT EIGEN_FIND_QUIETLY)
ELSE (EIGEN_FOUND)
  IF (EIGEN_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find EIGEN")
  ENDIF (EIGEN_FIND_REQUIRED)
ENDIF (EIGEN_FOUND)

MARK_AS_ADVANCED(
    EIGEN_INCLUDE_DIR
    EIGEN
)
