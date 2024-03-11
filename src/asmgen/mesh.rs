use std::iter::{empty, once};

use crate::asm::{Instruction, Pseudo, Register};

impl Instruction {
    pub fn walk_register(&self) -> Box<dyn Iterator<Item = Register> + '_> {
        match self {
            Instruction::RType {
                instr: _instr,
                rd,
                rs1,
                rs2,
            } => Box::new(rs2.iter().chain(once(rd)).chain(once(rs1)).copied()),
            Instruction::IType {
                instr: _instr,
                rd,
                rs1,
                imm: _imm,
            } => Box::new(once(*rd).chain(once(*rs1))),
            Instruction::SType {
                instr: _instr,
                rs1,
                rs2,
                imm: _imm,
            } => Box::new(once(*rs1).chain(once(*rs2))),
            Instruction::BType {
                instr: _instr,
                rs1,
                rs2,
                imm: _imm,
            } => Box::new(once(*rs1).chain(once(*rs2))),
            Instruction::UType {
                instr: _instr,
                rd,
                imm: _imm,
            } => Box::new(once(*rd)),
            Instruction::Pseudo(pseudo) => pseudo.walk_register(),
        }
    }
}

impl Pseudo {
    pub fn walk_register(&self) -> Box<dyn Iterator<Item = Register> + '_> {
        match self {
            Pseudo::La {
                rd,
                symbol: _symbol,
            } => Box::new(once(*rd)),
            Pseudo::Li { rd, imm: _imm } => Box::new(once(*rd)),
            Pseudo::Fneg {
                data_size: _data_size,
                rd,
                rs,
            }
            | Pseudo::Fmv {
                data_size: _data_size,
                rd,
                rs,
            }
            | Pseudo::Neg {
                data_size: _data_size,
                rd,
                rs,
            } => Box::new(once(*rd).chain(once(*rs))),
            Pseudo::Mv { rd, rs }
            | Pseudo::SextW { rd, rs }
            | Pseudo::Seqz { rd, rs }
            | Pseudo::Snez { rd, rs } => Box::new(once(*rd).chain(once(*rs))),
            Pseudo::Jr { rs } | Pseudo::Jalr { rs } => Box::new(once(*rs)),
            Pseudo::Ret => Box::new(empty()),
            Pseudo::J { offset: _offset } | Pseudo::Call { offset: _offset } => Box::new(empty()),
        }
    }
}
