use std::{
    iter::{empty, once},
    ops::Deref,
};

use bimap::BiMap;
use frozen::Frozen;

use crate::{
    asm::{self, Pseudo, Register, RegisterType},
    asmgen::{DirectOrInDirect, RegOrStack},
    ir::{
        self, BlockExit, BlockId, Dtype, FunctionDefinition, HasDtype, JumpArg, Operand, RegisterId,
    },
};

use super::FunctionAbi;

impl Into<regalloc2::RegClass> for RegisterType {
    fn into(self) -> regalloc2::RegClass {
        match self {
            RegisterType::Integer => regalloc2::RegClass::Int,
            RegisterType::FloatingPoint => regalloc2::RegClass::Float,
        }
    }
}

impl From<regalloc2::RegClass> for RegisterType {
    fn from(value: regalloc2::RegClass) -> Self {
        match value {
            regalloc2::RegClass::Int => Self::Integer,
            regalloc2::RegClass::Float => Self::FloatingPoint,
            regalloc2::RegClass::Vector => unreachable!(),
        }
    }
}

impl Into<regalloc2::PReg> for Register {
    fn into(self) -> regalloc2::PReg {
        match self {
            Register::Zero => unreachable!(),
            Register::Ra => unreachable!(),
            Register::Sp => unreachable!(),
            Register::Gp => unreachable!(),
            Register::Tp => unreachable!(),
            Register::Temp(register_type, offset) => {
                regalloc2::PReg::new(offset << 2, register_type.into())
            }
            Register::Saved(register_type, offset) => {
                regalloc2::PReg::new(offset << 2 | 1, register_type.into())
            }
            Register::Arg(register_type, offset) => {
                regalloc2::PReg::new(offset << 2 | 2, register_type.into())
            }
        }
    }
}
impl From<regalloc2::PReg> for Register {
    fn from(value: regalloc2::PReg) -> Self {
        let hw_enc = value.hw_enc();
        match hw_enc & 0b11 {
            0 => Register::Temp(value.class().into(), hw_enc >> 2),
            1 => Register::Saved(value.class().into(), hw_enc >> 2),
            2 => Register::Arg(value.class().into(), hw_enc >> 2),
            _ => unreachable!(),
        }
    }
}

impl Into<regalloc2::RegClass> for Dtype {
    fn into(self) -> regalloc2::RegClass {
        match self {
            Dtype::Int { .. } => regalloc2::RegClass::Int,
            Dtype::Float { .. } => regalloc2::RegClass::Float,
            Dtype::Pointer { .. }
            | Dtype::Struct { .. }
            | Dtype::Array { .. }
            | Dtype::Function { .. }
            | Dtype::Typedef { .. }
            | Dtype::Unit { .. } => unreachable!(),
        }
    }
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum Yank {
    BeforeFirst,
    Instruction(usize),
    BlockExit,
}

pub struct Gape<'a> {
    definition: &'a FunctionDefinition,
    abi: FunctionAbi,
    pub inst_mp: Frozen<BiMap<(BlockId, Yank), regalloc2::Inst>>,
    pub reg_mp: Frozen<BiMap<RegisterId, regalloc2::VReg>>,
    block_mp: Frozen<BiMap<BlockId, regalloc2::Block>>,
}

impl<'a> Gape<'a> {
    fn init_inst_mp(definition: &'a FunctionDefinition) -> BiMap<(BlockId, Yank), regalloc2::Inst> {
        let mut inst_mp: BiMap<(BlockId, Yank), regalloc2::Inst> = BiMap::new();
        let mut insn = regalloc2::Inst::new(0);

        for (&bid, block) in &definition.blocks {
            inst_mp
                .insert_no_overwrite((bid, Yank::BeforeFirst), insn)
                .unwrap();
            insn = insn.next();
            for offset in 0..block.instructions.len() {
                inst_mp
                    .insert_no_overwrite((bid, Yank::Instruction(offset)), insn)
                    .unwrap();
                insn = insn.next();
            }
            inst_mp
                .insert_no_overwrite((bid, Yank::BlockExit), insn)
                .unwrap();
            insn = insn.next();
        }
        inst_mp
    }

    fn init_reg_mp(definition: &'a FunctionDefinition) -> BiMap<RegisterId, regalloc2::VReg> {
        let mut reg_mp: BiMap<RegisterId, regalloc2::VReg> = BiMap::new();

        let mut index = 0usize;

        for (rid, dtype) in definition
            .blocks
            .iter()
            .flat_map(|(_, b)| b.walk_register())
            .filter(|(rid, _)| match rid {
                RegisterId::Local { .. } => false,
                RegisterId::Arg { .. } | RegisterId::Temp { .. } => true,
            })
        {
            match dtype {
                Dtype::Int { .. } | Dtype::Pointer { .. } => {
                    match reg_mp.insert_no_overwrite(
                        rid,
                        regalloc2::VReg::new(index, regalloc2::RegClass::Int),
                    ) {
                        Ok(_) => index += 1,
                        Err(_) => {}
                    }
                }
                Dtype::Float { .. } => {
                    match reg_mp.insert_no_overwrite(
                        rid,
                        regalloc2::VReg::new(index, regalloc2::RegClass::Float),
                    ) {
                        Ok(_) => index += 1,
                        Err(_) => {}
                    }
                }
                Dtype::Array { .. } | Dtype::Unit { .. } | Dtype::Struct { .. } => {}
                Dtype::Function { .. } => unreachable!(),
                Dtype::Typedef { .. } => unreachable!(),
            }
        }

        reg_mp
    }

    fn init_block_mp(definition: &'a FunctionDefinition) -> BiMap<BlockId, regalloc2::Block> {
        let mut block_mp: BiMap<BlockId, regalloc2::Block> = BiMap::new();
        let mut block = regalloc2::Block::new(0);

        for bid in definition.blocks.keys() {
            block_mp.insert_no_overwrite(*bid, block).unwrap();
            block = block.next()
        }
        block_mp
    }

    pub fn from_definition(definition: &'a FunctionDefinition, abi: FunctionAbi) -> Self {
        Self {
            definition,
            abi,
            inst_mp: Frozen::freeze(Self::init_inst_mp(definition)),
            reg_mp: Frozen::freeze(Self::init_reg_mp(definition)),
            block_mp: Frozen::freeze(Self::init_block_mp(definition)),
        }
    }
}

impl<'a> regalloc2::Function for Gape<'a> {
    fn num_insts(&self) -> usize {
        self.definition
            .blocks
            .values()
            .map(|b| b.instructions.len() + 1 + 1) // + 1 : blockexit
            // +1 : BeforeFirst
            .fold(0, |a, b| a + b)
    }

    fn num_blocks(&self) -> usize {
        self.definition.blocks.len()
    }

    fn entry_block(&self) -> regalloc2::Block {
        *self
            .block_mp
            .get_by_left(&self.definition.bid_init)
            .unwrap()
    }

    fn block_insns(&self, block: regalloc2::Block) -> regalloc2::InstRange {
        let block_id: BlockId = *self.block_mp.get_by_right(&block).unwrap();
        regalloc2::InstRange::forward(
            *self
                .inst_mp
                .get_by_left(&(block_id, Yank::BeforeFirst))
                .unwrap(),
            self.inst_mp
                .get_by_left(&(block_id, Yank::BlockExit))
                .unwrap()
                .next(),
        )
    }

    fn block_succs(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        let block_id: BlockId = *self.block_mp.get_by_right(&block).unwrap();
        let block_exit = &self.definition.blocks[&block_id].exit;
        let v: Vec<regalloc2::Block> = block_exit
            .walk_jump_bid()
            .map(|x| *self.block_mp.get_by_left(&x).unwrap())
            .collect();
        v.leak()
    }

    fn block_preds(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        let bid: BlockId = *self.block_mp.get_by_right(&block).unwrap();
        self.definition
            .blocks
            .iter()
            .filter(|(_, v)| {
                v.exit
                    .walk_jump_args_1()
                    .any(|jump_arg| jump_arg.bid == bid)
            })
            .map(|(k, _)| *self.block_mp.get_by_left(k).unwrap())
            .collect::<Vec<_>>()
            .leak()
    }

    fn block_params(&self, block: regalloc2::Block) -> &[regalloc2::VReg] {
        if block == self.entry_block() {
            return &[];
        }
        let bid: BlockId = *self.block_mp.get_by_right(&block).unwrap();
        let block = &self.definition.blocks[&bid];
        block
            .phinodes
            .iter()
            .enumerate()
            .map(|(aid, dtype)| {
                let rid = RegisterId::Arg { bid, aid };
                *self.reg_mp.get_by_left(&rid).unwrap()
            })
            .collect::<Vec<_>>()
            .leak()
    }

    fn is_ret(&self, insn: regalloc2::Inst) -> bool {
        let (bid, yank) = self.inst_mp.get_by_right(&insn).unwrap();
        match yank {
            Yank::BeforeFirst | Yank::Instruction(_) => false,
            Yank::BlockExit => match self.definition.blocks[&bid].exit {
                ir::BlockExit::Return { .. } => true,
                _ => false,
            },
        }
    }

    fn is_branch(&self, insn: regalloc2::Inst) -> bool {
        let (bid, yank) = self.inst_mp.get_by_right(&insn).unwrap();
        match yank {
            Yank::BeforeFirst | Yank::Instruction(_) => false,
            Yank::BlockExit => match self.definition.blocks[&bid].exit {
                ir::BlockExit::Jump { .. } => true,
                ir::BlockExit::ConditionalJump { .. } => true,
                ir::BlockExit::Switch { .. } => true,
                ir::BlockExit::Return { .. } => false,
                ir::BlockExit::Unreachable => false,
            },
        }
    }

    fn branch_blockparams(
        &self,
        block: regalloc2::Block,
        insn: regalloc2::Inst,
        succ_idx: usize,
    ) -> &[regalloc2::VReg] {
        let (bid, yank) = self.inst_mp.get_by_right(&insn).unwrap();
        let block_id_1: BlockId = *self.block_mp.get_by_right(&block).unwrap();
        assert_eq!(block_id_1, *bid);
        let Yank::BlockExit = yank else {
            unreachable!()
        };
        let Some(jump_arg) = self.definition.blocks[&bid]
            .exit
            .walk_jump_args_1()
            .nth(succ_idx)
        else {
            unreachable!()
        };
        jump_arg
            .args
            .iter()
            .map(|operand| match operand {
                Operand::Constant(_) => unreachable!(),
                Operand::Register { rid, dtype } => *self.reg_mp.get_by_left(&rid).unwrap(),
            })
            .collect::<Vec<_>>()
            .leak()
    }

    fn inst_operands(&self, insn: regalloc2::Inst) -> &[regalloc2::Operand] {
        let (bid, yank) = *self.inst_mp.get_by_right(&insn).unwrap();
        let block = &self.definition.blocks[&bid];

        match yank {
            Yank::BeforeFirst => {
                if self.entry_block() == *self.block_mp.get_by_left(&bid).unwrap() {
                    block
                        .phinodes
                        .iter()
                        .zip(self.abi.params_alloc.iter())
                        .enumerate()
                        .flat_map(|(aid, (dtype, param_alloc))| {
                            let rid = RegisterId::Arg { bid, aid };
                            match param_alloc {
                                crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::Direct(RegOrStack::Reg(reg)),
                                )
                                | crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::InDirect(RegOrStack::Reg(reg)),
                                ) => {
                                    let x = regalloc2::Operand::new(
                                        *self.reg_mp.get_by_left(&rid).unwrap(),
                                        regalloc2::OperandConstraint::FixedReg((*reg).into()),
                                        regalloc2::OperandKind::Def,
                                        regalloc2::OperandPos::Late,
                                    );
                                    Some(x)
                                }
                                crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::Direct(RegOrStack::Stack { .. }),
                                )
                                | crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::InDirect(RegOrStack::Stack { .. }),
                                ) => None,
                                crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. }),
                                )
                                | crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure {
                                        ..
                                    }),
                                )
                                | crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure {
                                        ..
                                    }),
                                )
                                | crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure {
                                        ..
                                    }),
                                ) => unreachable!(),
                                crate::asmgen::ParamAlloc::StructInRegister(_) => None,
                            }
                        })
                        .collect::<Vec<_>>()
                        .leak()
                } else {
                    &[]
                }
            }
            Yank::BlockExit => match block.exit {
                BlockExit::Jump { .. }
                | BlockExit::ConditionalJump { .. }
                | BlockExit::Switch { .. } => block
                    .exit
                    .walk_register()
                    .filter_map(|(rid, dtype)| match rid {
                        RegisterId::Local { .. } => None,
                        RegisterId::Arg { .. } | RegisterId::Temp { .. } => {
                            Some(regalloc2::Operand::new(
                                *self.reg_mp.get_by_left(&rid).unwrap(),
                                regalloc2::OperandConstraint::Any,
                                regalloc2::OperandKind::Use,
                                regalloc2::OperandPos::Early,
                            ))
                        }
                    })
                    .collect::<Vec<_>>()
                    .leak(),

                BlockExit::Return {
                    value:
                        Operand::Register {
                            rid,
                            dtype: Dtype::Int { .. } | Dtype::Pointer { .. },
                        },
                } => {
                    let x = regalloc2::Operand::new(
                        *self.reg_mp.get_by_left(&rid).unwrap(),
                        regalloc2::OperandConstraint::FixedReg(Register::A0.into()),
                        regalloc2::OperandKind::Use,
                        regalloc2::OperandPos::Early,
                    );
                    vec![x].leak()
                }
                BlockExit::Return {
                    value:
                        Operand::Register {
                            rid,
                            dtype: Dtype::Float { .. },
                        },
                } => {
                    let x = regalloc2::Operand::new(
                        *self.reg_mp.get_by_left(&rid).unwrap(),
                        regalloc2::OperandConstraint::FixedReg(Register::FA0.into()),
                        regalloc2::OperandKind::Use,
                        regalloc2::OperandPos::Early,
                    );
                    vec![x].leak()
                }
                BlockExit::Return {
                    value:
                        Operand::Register {
                            dtype: Dtype::Struct { .. },
                            ..
                        },
                }
                | BlockExit::Return {
                    value: Operand::Constant(..),
                } => &[],
                BlockExit::Unreachable => unreachable!(),
                _ => unreachable!(),
            },
            Yank::Instruction(iid) => {
                let instruction: &ir::Instruction = &block.instructions[iid];
                let mut v: Vec<regalloc2::Operand> = Vec::new();

                let rid = RegisterId::Temp { bid, iid };
                match instruction.dtype() {
                    Dtype::Unit { .. } => {}
                    Dtype::Int { .. } | Dtype::Pointer { .. } => {
                        match self.reg_mp.get_by_left(&rid) {
                            Some(dest) => v.push(regalloc2::Operand::new(
                                *dest,
                                regalloc2::OperandConstraint::Any,
                                regalloc2::OperandKind::Def,
                                regalloc2::OperandPos::Late,
                            )),
                            None => {
                                // result never used
                                return &[];
                            }
                        }
                    }
                    Dtype::Float { .. } => v.push(regalloc2::Operand::new(
                        *self.reg_mp.get_by_left(&rid).unwrap(),
                        regalloc2::OperandConstraint::Any,
                        regalloc2::OperandKind::Def,
                        regalloc2::OperandPos::Late,
                    )),

                    Dtype::Array { .. } => unreachable!(),
                    Dtype::Struct { .. } => {}
                    Dtype::Function { .. } => unreachable!(),
                    Dtype::Typedef { .. } => unreachable!(),
                }

                for (rid, dtype) in instruction.walk_register().filter(|(rid, _)| match rid {
                    RegisterId::Local { .. } => false,
                    RegisterId::Arg { .. } | RegisterId::Temp { .. } => true,
                }) {
                    match dtype {
                        Dtype::Unit { .. } => unreachable!(),
                        Dtype::Int { .. } | Dtype::Pointer { .. } => {
                            v.push(regalloc2::Operand::new(
                                *self.reg_mp.get_by_left(&rid).unwrap(),
                                regalloc2::OperandConstraint::Any,
                                regalloc2::OperandKind::Use,
                                regalloc2::OperandPos::Early,
                            ))
                        }
                        Dtype::Float { .. } => v.push(regalloc2::Operand::new(
                            *self.reg_mp.get_by_left(&rid).unwrap(),
                            regalloc2::OperandConstraint::Any,
                            regalloc2::OperandKind::Use,
                            regalloc2::OperandPos::Early,
                        )),
                        Dtype::Array { .. } => unreachable!(),
                        Dtype::Struct { .. } => {}
                        Dtype::Function { .. } => unreachable!(),
                        Dtype::Typedef { .. } => unreachable!(),
                    }
                }

                v.leak()
            }
        }
    }

    fn inst_clobbers(&self, insn: regalloc2::Inst) -> regalloc2::PRegSet {
        let (block_id, yank) = *self.inst_mp.get_by_right(&insn).unwrap();
        let block = &self.definition.blocks[&block_id.into()];
        match yank {
            Yank::Instruction(offset) => match block.instructions[offset].deref() {
                ir::Instruction::TypeCast { .. }
                | ir::Instruction::GetElementPtr { .. }
                | ir::Instruction::Store { .. }
                | ir::Instruction::UnaryOp { .. }
                | ir::Instruction::BinOp { .. }
                | ir::Instruction::Nop
                | ir::Instruction::Load { .. } => regalloc2::PRegSet::empty(),
                ir::Instruction::Call { .. } => {
                    let mut res = regalloc2::PRegSet::empty();
                    res.add(Register::T0.into());
                    res.add(Register::T1.into());
                    res.add(Register::T2.into());
                    res.add(Register::T3.into());
                    res.add(Register::T4.into());
                    res.add(Register::T5.into());
                    res.add(Register::T6.into());

                    res.add(Register::FT0.into());
                    res.add(Register::FT1.into());
                    res.add(Register::FT2.into());
                    res.add(Register::FT3.into());
                    res.add(Register::FT4.into());
                    res.add(Register::FT5.into());
                    res.add(Register::FT6.into());
                    res.add(Register::FT7.into());
                    res.add(Register::FT8.into());
                    res.add(Register::FT9.into());
                    res.add(Register::FT10.into());
                    res.add(Register::FT11.into());

                    res.add(Register::A0.into());
                    res.add(Register::A1.into());
                    res.add(Register::A2.into());
                    res.add(Register::A3.into());
                    res.add(Register::A4.into());
                    res.add(Register::A5.into());
                    res.add(Register::A6.into());
                    res.add(Register::A7.into());

                    res.add(Register::FA0.into());
                    res.add(Register::FA1.into());
                    res.add(Register::FA2.into());
                    res.add(Register::FA3.into());
                    res.add(Register::FA4.into());
                    res.add(Register::FA5.into());
                    res.add(Register::FA6.into());
                    res.add(Register::FA7.into());
                    res
                }
            },
            Yank::BeforeFirst | Yank::BlockExit => regalloc2::PRegSet::empty(),
        }
    }

    fn num_vregs(&self) -> usize {
        self.definition
            .blocks
            .values()
            .fold(self.definition.allocations.len(), |acc, b| {
                acc + b.instructions.len() + b.phinodes.len()
            })
    }

    fn spillslot_size(&self, regclass: regalloc2::RegClass) -> usize {
        match regclass {
            regalloc2::RegClass::Int => 1,
            regalloc2::RegClass::Float => 1,
            regalloc2::RegClass::Vector => unreachable!(),
        }
    }
}

impl BlockExit {
    pub fn walk_jump_args_1(&self) -> Box<dyn Iterator<Item = &JumpArg> + '_> {
        match self {
            Self::Jump { arg } => Box::new(once(arg)),
            Self::ConditionalJump {
                arg_then, arg_else, ..
            } => Box::new(once(arg_then).chain(once(arg_else))),
            Self::Switch { default, cases, .. } => {
                Box::new(once(default).chain(cases.iter().map(|x| &x.1)))
            }
            Self::Return { .. } | Self::Unreachable => Box::new(empty()),
        }
    }
}

pub fn edit_2_instruction(edit: &regalloc2::Edit) -> Vec<asm::Instruction> {
    match edit {
        regalloc2::Edit::Move { from, to } => match (from.kind(), to.kind()) {
            (regalloc2::AllocationKind::None, regalloc2::AllocationKind::None) => todo!(),
            (regalloc2::AllocationKind::None, regalloc2::AllocationKind::Reg) => todo!(),
            (regalloc2::AllocationKind::None, regalloc2::AllocationKind::Stack) => todo!(),
            (regalloc2::AllocationKind::Reg, regalloc2::AllocationKind::None) => todo!(),
            (regalloc2::AllocationKind::Reg, regalloc2::AllocationKind::Reg) => {
                let rs = from.as_reg().unwrap().into();
                let rd = to.as_reg().unwrap().into();
                vec![asm::Instruction::Pseudo(Pseudo::Mv { rd, rs })]
            }
            (regalloc2::AllocationKind::Reg, regalloc2::AllocationKind::Stack) => todo!(),
            (regalloc2::AllocationKind::Stack, regalloc2::AllocationKind::None) => todo!(),
            (regalloc2::AllocationKind::Stack, regalloc2::AllocationKind::Reg) => todo!(),
            (regalloc2::AllocationKind::Stack, regalloc2::AllocationKind::Stack) => todo!(),
        },
    }
}
