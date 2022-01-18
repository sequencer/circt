add_circt_dialect_library(CIRCTGAATransforms
  Analysis.cpp

  DEPENDS
  CIRCTGAATransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  CIRCTGAA
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
