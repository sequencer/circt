set(GAA_SRCS
  GAADialect.cpp
  GAAOps.cpp
  GAATypes.cpp
  )

add_circt_dialect_library(GAA
  ${GAA_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR

  DEPENDS
  CIRCTGAAEnumsIncGen
  CIRCTGAAAttrsIncGen
  )
