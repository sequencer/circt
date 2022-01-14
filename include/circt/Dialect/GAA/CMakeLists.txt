# GAA
add_circt_dialect(GAA gaa)
add_circt_doc(GAA -gen-dialect-doc GAA Dialects/)

set(LLVM_TARGET_DEFINITIONS GAA.td)

mlir_tablegen(GAAEnums.h.inc -gen-enum-decls)
mlir_tablegen(GAAEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(CIRCTGAAEnumsIncGen)
add_dependencies(circt-headers CIRCTGAAEnumsIncGen)

mlir_tablegen(GAAAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(GAAAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTGAAAttrsIncGen)
add_dependencies(circt-headers CIRCTGAAAttrsIncGen)