// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @affine_apply_loops_shorthand(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: index,
// CHECK-SAME:                                                 %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_3:.*]]:3 = fork [3] %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]]#1 {value = 0 : index} : index
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_3]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_6:.*]] = br %[[VAL_2]] : index
// CHECK:           %[[VAL_7:.*]] = br %[[VAL_3]]#2 : none
// CHECK:           %[[VAL_8:.*]] = br %[[VAL_4]] : index
// CHECK:           %[[VAL_9:.*]] = br %[[VAL_5]] : index
// CHECK:           %[[VAL_10:.*]] = mux %[[VAL_11:.*]]#2 {{\[}}%[[VAL_12:.*]], %[[VAL_6]]] : index, index
// CHECK:           %[[VAL_13:.*]]:2 = fork [2] %[[VAL_10]] : index
// CHECK:           %[[VAL_14:.*]] = mux %[[VAL_11]]#1 {{\[}}%[[VAL_15:.*]], %[[VAL_9]]] : index, index
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = control_merge %[[VAL_18:.*]], %[[VAL_7]] : none
// CHECK:           %[[VAL_11]]:3 = fork [3] %[[VAL_17]] : index
// CHECK:           %[[VAL_19:.*]] = mux %[[VAL_11]]#0 {{\[}}%[[VAL_20:.*]], %[[VAL_8]]] : index, index
// CHECK:           %[[VAL_21:.*]]:2 = fork [2] %[[VAL_19]] : index
// CHECK:           %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_21]]#1, %[[VAL_13]]#1 : index
// CHECK:           %[[VAL_23:.*]]:4 = fork [4] %[[VAL_22]] : i1
// CHECK:           %[[VAL_24:.*]], %[[VAL_25:.*]] = cond_br %[[VAL_23]]#3, %[[VAL_13]]#0 : index
// CHECK:           sink %[[VAL_25]] : index
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = cond_br %[[VAL_23]]#2, %[[VAL_14]] : index
// CHECK:           sink %[[VAL_27]] : index
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]] = cond_br %[[VAL_23]]#1, %[[VAL_16]] : none
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = cond_br %[[VAL_23]]#0, %[[VAL_21]]#0 : index
// CHECK:           sink %[[VAL_31]] : index
// CHECK:           %[[VAL_32:.*]] = merge %[[VAL_30]] : index
// CHECK:           %[[VAL_33:.*]]:2 = fork [2] %[[VAL_32]] : index
// CHECK:           %[[VAL_34:.*]] = merge %[[VAL_26]] : index
// CHECK:           %[[VAL_35:.*]] = merge %[[VAL_24]] : index
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]] = control_merge %[[VAL_28]] : none
// CHECK:           %[[VAL_38:.*]]:3 = fork [3] %[[VAL_36]] : none
// CHECK:           sink %[[VAL_37]] : index
// CHECK:           %[[VAL_39:.*]] = constant %[[VAL_38]]#1 {value = 42 : index} : index
// CHECK:           %[[VAL_40:.*]] = constant %[[VAL_38]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_41:.*]] = br %[[VAL_33]]#1 : index
// CHECK:           %[[VAL_42:.*]] = br %[[VAL_33]]#0 : index
// CHECK:           %[[VAL_43:.*]] = br %[[VAL_34]] : index
// CHECK:           %[[VAL_44:.*]] = br %[[VAL_35]] : index
// CHECK:           %[[VAL_45:.*]] = br %[[VAL_38]]#2 : none
// CHECK:           %[[VAL_46:.*]] = br %[[VAL_39]] : index
// CHECK:           %[[VAL_47:.*]] = br %[[VAL_40]] : index
// CHECK:           %[[VAL_48:.*]] = mux %[[VAL_49:.*]]#5 {{\[}}%[[VAL_50:.*]], %[[VAL_46]]] : index, index
// CHECK:           %[[VAL_51:.*]]:2 = fork [2] %[[VAL_48]] : index
// CHECK:           %[[VAL_52:.*]] = mux %[[VAL_49]]#4 {{\[}}%[[VAL_53:.*]], %[[VAL_47]]] : index, index
// CHECK:           %[[VAL_54:.*]] = mux %[[VAL_49]]#3 {{\[}}%[[VAL_55:.*]], %[[VAL_42]]] : index, index
// CHECK:           %[[VAL_56:.*]] = mux %[[VAL_49]]#2 {{\[}}%[[VAL_57:.*]], %[[VAL_43]]] : index, index
// CHECK:           %[[VAL_58:.*]] = mux %[[VAL_49]]#1 {{\[}}%[[VAL_59:.*]], %[[VAL_44]]] : index, index
// CHECK:           %[[VAL_60:.*]], %[[VAL_61:.*]] = control_merge %[[VAL_62:.*]], %[[VAL_45]] : none
// CHECK:           %[[VAL_49]]:6 = fork [6] %[[VAL_61]] : index
// CHECK:           %[[VAL_63:.*]] = mux %[[VAL_49]]#0 {{\[}}%[[VAL_64:.*]], %[[VAL_41]]] : index, index
// CHECK:           %[[VAL_65:.*]]:2 = fork [2] %[[VAL_63]] : index
// CHECK:           %[[VAL_66:.*]] = arith.cmpi slt, %[[VAL_65]]#1, %[[VAL_51]]#1 : index
// CHECK:           %[[VAL_67:.*]]:7 = fork [7] %[[VAL_66]] : i1
// CHECK:           %[[VAL_68:.*]], %[[VAL_69:.*]] = cond_br %[[VAL_67]]#6, %[[VAL_51]]#0 : index
// CHECK:           sink %[[VAL_69]] : index
// CHECK:           %[[VAL_70:.*]], %[[VAL_71:.*]] = cond_br %[[VAL_67]]#5, %[[VAL_52]] : index
// CHECK:           sink %[[VAL_71]] : index
// CHECK:           %[[VAL_72:.*]], %[[VAL_73:.*]] = cond_br %[[VAL_67]]#4, %[[VAL_54]] : index
// CHECK:           %[[VAL_74:.*]], %[[VAL_75:.*]] = cond_br %[[VAL_67]]#3, %[[VAL_56]] : index
// CHECK:           %[[VAL_76:.*]], %[[VAL_77:.*]] = cond_br %[[VAL_67]]#2, %[[VAL_58]] : index
// CHECK:           %[[VAL_78:.*]], %[[VAL_79:.*]] = cond_br %[[VAL_67]]#1, %[[VAL_60]] : none
// CHECK:           %[[VAL_80:.*]], %[[VAL_81:.*]] = cond_br %[[VAL_67]]#0, %[[VAL_65]]#0 : index
// CHECK:           sink %[[VAL_81]] : index
// CHECK:           %[[VAL_82:.*]] = merge %[[VAL_80]] : index
// CHECK:           %[[VAL_83:.*]] = merge %[[VAL_70]] : index
// CHECK:           %[[VAL_84:.*]]:2 = fork [2] %[[VAL_83]] : index
// CHECK:           %[[VAL_85:.*]] = merge %[[VAL_68]] : index
// CHECK:           %[[VAL_86:.*]] = merge %[[VAL_72]] : index
// CHECK:           %[[VAL_87:.*]] = merge %[[VAL_74]] : index
// CHECK:           %[[VAL_88:.*]] = merge %[[VAL_76]] : index
// CHECK:           %[[VAL_89:.*]], %[[VAL_90:.*]] = control_merge %[[VAL_78]] : none
// CHECK:           sink %[[VAL_90]] : index
// CHECK:           %[[VAL_91:.*]] = arith.addi %[[VAL_82]], %[[VAL_84]]#1 : index
// CHECK:           %[[VAL_53]] = br %[[VAL_84]]#0 : index
// CHECK:           %[[VAL_50]] = br %[[VAL_85]] : index
// CHECK:           %[[VAL_55]] = br %[[VAL_86]] : index
// CHECK:           %[[VAL_57]] = br %[[VAL_87]] : index
// CHECK:           %[[VAL_59]] = br %[[VAL_88]] : index
// CHECK:           %[[VAL_62]] = br %[[VAL_89]] : none
// CHECK:           %[[VAL_64]] = br %[[VAL_91]] : index
// CHECK:           %[[VAL_92:.*]] = merge %[[VAL_73]] : index
// CHECK:           %[[VAL_93:.*]] = merge %[[VAL_75]] : index
// CHECK:           %[[VAL_94:.*]]:2 = fork [2] %[[VAL_93]] : index
// CHECK:           %[[VAL_95:.*]] = merge %[[VAL_77]] : index
// CHECK:           %[[VAL_96:.*]], %[[VAL_97:.*]] = control_merge %[[VAL_79]] : none
// CHECK:           sink %[[VAL_97]] : index
// CHECK:           %[[VAL_98:.*]] = arith.addi %[[VAL_92]], %[[VAL_94]]#1 : index
// CHECK:           %[[VAL_15]] = br %[[VAL_94]]#0 : index
// CHECK:           %[[VAL_12]] = br %[[VAL_95]] : index
// CHECK:           %[[VAL_18]] = br %[[VAL_96]] : none
// CHECK:           %[[VAL_20]] = br %[[VAL_98]] : index
// CHECK:           %[[VAL_99:.*]], %[[VAL_100:.*]] = control_merge %[[VAL_29]] : none
// CHECK:           sink %[[VAL_100]] : index
// CHECK:           return %[[VAL_99]] : none
// CHECK:         }
  func @affine_apply_loops_shorthand(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    br ^bb1(%c0 : index)
  ^bb1(%0: index):      // 2 preds: ^bb0, ^bb5
    %1 = arith.cmpi slt, %0, %arg0 : index
    cond_br %1, ^bb2, ^bb6
  ^bb2: // pred: ^bb1
    %c42 = arith.constant 42 : index
    %c1_0 = arith.constant 1 : index
    br ^bb3(%0 : index)
  ^bb3(%2: index):      // 2 preds: ^bb2, ^bb4
    %3 = arith.cmpi slt, %2, %c42 : index
    cond_br %3, ^bb4, ^bb5
  ^bb4: // pred: ^bb3
    %4 = arith.addi %2, %c1_0 : index
    br ^bb3(%4 : index)
  ^bb5: // pred: ^bb3
    %5 = arith.addi %0, %c1 : index
    br ^bb1(%5 : index)
  ^bb6: // pred: ^bb1
    return
  }
