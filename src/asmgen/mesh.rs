use std::iter::{empty, once};

use crate::asm::{self, Block, Instruction, Label, Pseudo, Register};

impl asm::Function {
    pub fn walk_label(&self) -> Box<dyn Iterator<Item = &Label> + '_> {
        Box::new(self.blocks.iter().flat_map(Block::walk_label))
    }

    pub fn walk_label_mut(&mut self) -> Box<dyn Iterator<Item = &mut Label> + '_> {
        Box::new(self.blocks.iter_mut().flat_map(Block::walk_label_mut))
    }
}

impl Block {
    pub fn walk_label(&self) -> Box<dyn Iterator<Item = &Label> + '_> {
        Box::new(self.instructions.iter().flat_map(Instruction::walk_label))
    }

    pub fn walk_label_mut(&mut self) -> Box<dyn Iterator<Item = &mut Label> + '_> {
        Box::new(
            self.instructions
                .iter_mut()
                .flat_map(Instruction::walk_label_mut),
        )
    }
}

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

    pub fn walk_label(&self) -> Box<dyn Iterator<Item = &Label> + '_> {
        match self {
            Instruction::RType { .. }
            | Instruction::IType { .. }
            | Instruction::SType { .. }
            | Instruction::UType { .. } => Box::new(empty()),
            Instruction::BType { imm, .. } => Box::new(once(imm)),
            Instruction::Pseudo(pseudo) => pseudo.walk_label(),
        }
    }

    pub fn walk_label_mut(&mut self) -> Box<dyn Iterator<Item = &mut Label> + '_> {
        match self {
            Instruction::RType { .. }
            | Instruction::IType { .. }
            | Instruction::SType { .. }
            | Instruction::UType { .. } => Box::new(empty()),
            Instruction::BType { imm, .. } => Box::new(once(imm)),
            Instruction::Pseudo(pseudo) => pseudo.walk_label_mut(),
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

    pub fn walk_label(&self) -> Box<dyn Iterator<Item = &Label> + '_> {
        match self {
            Pseudo::La { rd, symbol } => Box::new(once(symbol)),
            Pseudo::J { offset } | Pseudo::Call { offset } => Box::new(once(offset)),
            Pseudo::Li { .. }
            | Pseudo::Mv { .. }
            | Pseudo::Fmv { .. }
            | Pseudo::Neg { .. }
            | Pseudo::SextW { .. }
            | Pseudo::Seqz { .. }
            | Pseudo::Snez { .. }
            | Pseudo::Fneg { .. }
            | Pseudo::Jr { .. }
            | Pseudo::Jalr { .. }
            | Pseudo::Ret => Box::new(empty()),
        }
    }

    pub fn walk_label_mut(&mut self) -> Box<dyn Iterator<Item = &mut Label> + '_> {
        match self {
            Pseudo::La { rd, symbol } => Box::new(once(symbol)),
            Pseudo::J { offset } | Pseudo::Call { offset } => Box::new(once(offset)),
            Pseudo::Li { .. }
            | Pseudo::Mv { .. }
            | Pseudo::Fmv { .. }
            | Pseudo::Neg { .. }
            | Pseudo::SextW { .. }
            | Pseudo::Seqz { .. }
            | Pseudo::Snez { .. }
            | Pseudo::Fneg { .. }
            | Pseudo::Jr { .. }
            | Pseudo::Jalr { .. }
            | Pseudo::Ret => Box::new(empty()),
        }
    }
}
