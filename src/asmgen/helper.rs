use std::{
    iter::{empty, once},
    ops::Deref,
};

use regalloc2::Function;

use crate::{
    asm::{Register, RegisterType},
    ir::{
        self, Block, BlockExit, BlockId, Dtype, FunctionDefinition, JumpArg, Operand, RegisterId,
    },
};

impl From<regalloc2::Block> for BlockId {
    fn from(value: regalloc2::Block) -> Self {
        Self(value.index())
    }
}

impl Into<regalloc2::Block> for BlockId {
    fn into(self) -> regalloc2::Block {
        regalloc2::Block::new(self.0)
    }
}

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

// to regalloc2::VReg
impl Into<usize> for RegisterId {
    fn into(self) -> usize {
        match self {
            RegisterId::Local { aid } => aid << 2,
            // 30 + 32 + 2
            RegisterId::Arg { bid, aid } => bid.0 << 34 | aid << 2 | 1,
            RegisterId::Temp { bid, iid } => bid.0 << 34 | iid << 2 | 2,
        }
    }
}

impl From<usize> for RegisterId {
    fn from(value: usize) -> Self {
        match value & 0b11 {
            0 => RegisterId::Local { aid: value >> 2 },
            1 => {
                let value = value >> 2;
                let aid = value & ((1 << 32) - 1);
                let bid = value >> 32;
                RegisterId::Arg {
                    bid: BlockId(bid),
                    aid,
                }
            }
            2 => {
                let value = value >> 2;
                let iid = value & ((1 << 32) - 1);
                let bid = value >> 32;
                RegisterId::Temp {
                    bid: BlockId(bid),
                    iid,
                }
            }
            _ => unreachable!(),
        }
    }
}

impl From<regalloc2::VReg> for RegisterId {
    fn from(value: regalloc2::VReg) -> Self {
        let index = value.vreg();
        index.into()
    }
}

enum Yank {
    Instruction(usize),
    BlockExit,
}

impl Block {
    fn encode_yank(&self, insn: usize) -> Yank {
        assert!(insn <= u16::MAX.into());
        match self.instructions.len().cmp(&insn) {
            std::cmp::Ordering::Less => unreachable!(),
            std::cmp::Ordering::Equal => Yank::BlockExit,
            std::cmp::Ordering::Greater => Yank::Instruction(insn),
        }
    }

    fn decode_yank(&self, yank: Yank) -> u16 {
        let x = match yank {
            Yank::Instruction(y) => y,
            Yank::BlockExit => self.instructions.len(),
        };
        x.try_into().unwrap()
    }
}

impl FunctionDefinition {
    fn decode_inst(&self, insn: regalloc2::Inst) -> Riddle {
        self.encode_riddle(insn)
    }

    fn encode_riddle(&self, insn: regalloc2::Inst) -> Riddle {
        let value: u32 = insn.raw_u32();
        let block_id = BlockId((value >> 16).try_into().unwrap());
        let offset: usize = (value & ((1 << 16) - 1)).try_into().unwrap();
        let yank = self.blocks[&block_id].encode_yank(offset);
        Riddle { block_id, yank }
    }

    fn decode_riddle(&self, riddle: Riddle) -> regalloc2::Inst {
        let block_id: u16 = riddle.block_id.0.try_into().unwrap();
        let offset = self.blocks[&riddle.block_id].decode_yank(riddle.yank);
        regalloc2::Inst::new(((block_id << 16) | offset).try_into().unwrap())
    }

    fn get_first_ridder(&self, block_id: BlockId) -> Riddle {
        let block = &self.blocks[&block_id];
        let yank = if block.instructions.is_empty() {
            Yank::BlockExit
        } else {
            Yank::Instruction(0)
        };
        Riddle { block_id, yank }
    }
}

struct Riddle {
    block_id: BlockId,
    yank: Yank,
}

struct Gape<'a> {
    definition: &'a FunctionDefinition,
}

impl<'a> Gape<'a> {
    fn new(definition: &'a FunctionDefinition) -> Self {
        Self { definition }
    }
}

impl<'a> regalloc2::Function for Gape<'a> {
    fn num_insts(&self) -> usize {
        self.definition
            .blocks
            .values()
            .map(|b| b.instructions.len() + 1) // + 1 : blockexit
            .fold(0, |a, b| a + b)
    }

    fn num_blocks(&self) -> usize {
        self.definition.blocks.len()
    }

    fn entry_block(&self) -> regalloc2::Block {
        self.definition.bid_init.into()
    }

    fn block_insns(&self, block: regalloc2::Block) -> regalloc2::InstRange {
        let block_id: BlockId = block.into();
        let from = self.definition.get_first_ridder(block_id);
        let to = Riddle {
            block_id,
            yank: Yank::BlockExit,
        };
        regalloc2::InstRange::forward(
            self.definition.decode_riddle(from),
            self.definition.decode_riddle(to),
        )
    }

    fn block_succs(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        let block_id: BlockId = block.into();
        let block_exit = &self.definition.blocks[&block_id].exit;
        let v: Vec<regalloc2::Block> = block_exit.walk_jump_bid().map(|x| x.into()).collect();
        v.leak()
    }

    fn block_preds(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        let bid: BlockId = block.into();
        self.definition
            .blocks
            .iter()
            .filter(|(_, v)| {
                v.exit
                    .walk_jump_args_1()
                    .any(|jump_arg| jump_arg.bid == bid)
            })
            .map(|(&k, _)| k.into())
            .collect::<Vec<_>>()
            .leak()
    }

    fn block_params(&self, block: regalloc2::Block) -> &[regalloc2::VReg] {
        let bid: BlockId = block.into();
        let block = &self.definition.blocks[&bid];
        block
            .phinodes
            .iter()
            .enumerate()
            .map(|(aid, dtype)| {
                let rid = RegisterId::Arg { bid, aid };
                regalloc2::VReg::new(rid.into(), dtype.deref().clone().into())
            })
            .collect::<Vec<_>>()
            .leak()
    }

    fn is_ret(&self, insn: regalloc2::Inst) -> bool {
        let riddle = self.definition.decode_inst(insn);
        match riddle.yank {
            Yank::Instruction(_) => false,
            Yank::BlockExit => match self.definition.blocks[&riddle.block_id].exit {
                ir::BlockExit::Return { .. } => true,
                _ => false,
            },
        }
    }

    fn is_branch(&self, insn: regalloc2::Inst) -> bool {
        let riddle = self.definition.decode_inst(insn);
        match riddle.yank {
            Yank::Instruction(_) => false,
            Yank::BlockExit => match self.definition.blocks[&riddle.block_id].exit {
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
        let Riddle { block_id, yank } = self.definition.decode_inst(insn);
        let block_id_1: BlockId = block.into();
        assert_eq!(block_id_1, block_id);
        let Yank::BlockExit = yank else {
            unreachable!()
        };
        let Some(jump_arg) = self.definition.blocks[&block_id]
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
                Operand::Register { rid, dtype } => {
                    regalloc2::VReg::new((*rid).into(), dtype.clone().into())
                }
            })
            .collect::<Vec<_>>()
            .leak()
    }

    fn inst_operands(&self, insn: regalloc2::Inst) -> &[regalloc2::Operand] {
        let Riddle { block_id : bid, yank } = self.definition.decode_inst(insn);
        let block = &self.definition.blocks[&bid];
        match yank {
            Yank::BlockExit => block
                .exit
                .walk_operand()
                .flat_map(|operand| match operand {
                    Operand::Constant(_) => None,
                    Operand::Register { rid, dtype } => {
                        let operand = regalloc2::Operand::new(
                            regalloc2::VReg::new(rid.clone().into(), dtype.clone().into()),
                            regalloc2::OperandConstraint::Any,
                            regalloc2::OperandKind::Use,
                            regalloc2::OperandPos::Early,
                        );
                        Some(operand)
                    }
                })
                .collect::<Vec<_>>()
                .leak(),
            Yank::Instruction(offset) => {
                let mut v = Vec::new();
                if offset == 0 && self.entry_block() == bid.into() {
                    // visit abi
                    let x = block
                        .phinodes
                        .iter()
                        .enumerate()
                        .map(|(aid, dtype)| {
                            let rid = RegisterId::Arg { bid, aid };
                            regalloc2::Operand::new(
                                regalloc2::VReg::new(rid.into(), dtype.deref().clone().into()),
                                regalloc2::OperandConstraint::Any, // TODO:
                                regalloc2::OperandKind::Use,
                                regalloc2::OperandPos::Early,
                            );
                        })
                        .collect::<Vec<_>>();
                }
                todo!()
            }
        }
    }

    fn inst_clobbers(&self, insn: regalloc2::Inst) -> regalloc2::PRegSet {
        let Riddle { block_id, yank } = self.definition.decode_inst(insn);
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
            Yank::BlockExit => regalloc2::PRegSet::empty(),
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
