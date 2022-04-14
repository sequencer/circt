add_circt_dialect_library(CIRCTGAATransforms
  GenerateConflictMatrix.cpp

  DEPENDS
  CIRCTGAAEnumsIncGen
  CIRCTGAAAttrsIncGen
  CIRCTGAATransformsIncGen
  CIRCTGAAOpInterfacesIncGen
  CIRCTGAA

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  CIRCTGAA
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
