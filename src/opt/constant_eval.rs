use std::collections::HashMap;

use lang_c::ast::UnaryOperator;

use crate::ir::interp::calculator::{
    calculate_binary_operator_expression, calculate_typecast, calculate_unary_operator_expression,
};
use crate::ir::{FunctionDefinition, Operand, RegisterId, Value};
use crate::opt::domtree::DomTree;
use crate::{ir, FunctionPass, Optimize};

pub type ConstantEval = FunctionPass<ConstantEvalInner>;

/// only replace int
/// float is strange in ir
#[derive(Default, Clone, Copy, Debug)]
pub struct ConstantEvalInner {}

impl Optimize<FunctionDefinition> for ConstantEvalInner {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let rpo = code.reverse_post_order();

        // key : RegisterId::Temp{}
        let mut int_mp: HashMap<RegisterId, Value> = Default::default();

        for bid in rpo {
            let block = &code.blocks[&bid];
            for (iid, instr) in block.instructions.iter().enumerate() {
                match &**instr {
                    ir::Instruction::UnaryOp {
                        op: UnaryOperator::Minus,
                        ..
                    } => {
                        // ignore minus because of unknown bugs
                    }
                    ir::Instruction::UnaryOp { op, operand, .. } => {
                        let Some(operand) = downturn(operand, &int_mp) else {
                            continue;
                        };
                        let Ok(v) = calculate_unary_operator_expression(op, operand) else {
                            continue;
                        };
                        let None = int_mp.insert(RegisterId::Temp { bid, iid }, v) else {
                            unreachable!()
                        };
                    }
                    ir::Instruction::BinOp { op, lhs, rhs, .. } => {
                        let Some(lhs) = downturn(lhs, &int_mp) else {
                            continue;
                        };
                        let Some(rhs) = downturn(rhs, &int_mp) else {
                            continue;
                        };
                        let Ok(v) = calculate_binary_operator_expression(op, lhs, rhs) else {
                            continue;
                        };
                        let None = int_mp.insert(RegisterId::Temp { bid, iid }, v) else {
                            unreachable!()
                        };
                    }
                    ir::Instruction::TypeCast {
                        value,
                        target_dtype,
                    } => {
                        let Some(value) = downturn(value, &int_mp) else {
                            continue;
                        };
                        let Ok(v) = calculate_typecast(value, target_dtype.clone()) else {
                            continue;
                        };
                        let None = int_mp.insert(RegisterId::Temp { bid, iid }, v) else {
                            unreachable!()
                        };
                    }
                    ir::Instruction::Nop
                    | ir::Instruction::GetElementPtr { .. }
                    | ir::Instruction::Store { .. }
                    | ir::Instruction::Load { .. }
                    | ir::Instruction::Call { .. } => {}
                }
            }
        }

        let mut b = false;
        for (rid, value) in int_mp.into_iter() {
            match &value {
                Value::Int {
                    value,
                    width,
                    is_signed,
                } => {
                    let operand: Operand = Operand::Constant(ir::Constant::Int {
                        value: *value,
                        width: *width,
                        is_signed: *is_signed,
                    });
                    b |= code.replace(rid, operand);
                }
                Value::Float { .. } => {}
                Value::Undef { .. } => unreachable!(),
                Value::Unit => unreachable!(),
                Value::Pointer { .. } => unreachable!(),
                Value::Array { .. } => unreachable!(),
                Value::Struct { .. } => unreachable!(),
            }
        }

        b
    }
}

fn downturn(operand: &Operand, int_mp: &HashMap<RegisterId, Value>) -> Option<Value> {
    match operand {
        Operand::Constant(c) => match Value::try_from(c.clone()) {
            Ok(value) => Some(value),
            Err(_) => None,
        },
        Operand::Register { rid, .. } => int_mp.get(rid).cloned(),
    }
}

impl FunctionDefinition {
    fn replace(&mut self, rid_from: RegisterId, operand: Operand) -> bool {
        let mut b = false;
        for x in self.walk_operand_mut() {
            match x {
                Operand::Constant(_) => {}
                Operand::Register { rid, .. } => {
                    if *rid == rid_from {
                        *x = operand.clone();
                        b = true;
                    }
                }
            }
        }
        b
    }
}
