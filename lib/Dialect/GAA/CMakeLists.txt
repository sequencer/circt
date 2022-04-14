add_circt_dialect_library(CIRCTGAA
  GAADialect.cpp
  GAAOps.cpp
  InstanceGraph.cpp
  GAAOpInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/GAA

  DEPENDS
  CIRCTHW
  MLIRGAAIncGen
  CIRCTGAAEnumsIncGen
  CIRCTGAAAttrsIncGen
  CIRCTGAAOpInterfacesIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTSupport
  CIRCTHW
  MLIRIR
  )
add_subdirectory(Transforms)
