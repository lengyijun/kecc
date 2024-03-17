use crate::asm::{self, Pseudo, RType, TranslationUnit};

impl asm::Block {
    // rm `mv a0, a0`
    // rm `fmv fa0, fa0`
    fn rm_needless_mv(&mut self) {
        self.instructions.retain(|instr| match instr {
            asm::Instruction::Pseudo(Pseudo::Fmv { rd, rs, .. }) => {
                if rd == rs {
                    false
                } else {
                    true
                }
            }
            asm::Instruction::Pseudo(Pseudo::Mv { rd, rs }) => {
                if rd == rs {
                    false
                } else {
                    true
                }
            }
            asm::Instruction::RType {
                instr: RType::Add(_),
                rd,
                rs1: asm::Register::Zero,
                rs2: Some(x),
            }
            | asm::Instruction::RType {
                instr: RType::Add(_),
                rd,
                rs1: x,
                rs2: Some(asm::Register::Zero),
            } => {
                if rd == x {
                    false
                } else {
                    true
                }
            }
            _ => true,
        });
    }
}

impl asm::Function {
    fn rm_needless_mv(&mut self) {
        for block in self.blocks.iter_mut() {
            block.rm_needless_mv();
        }
    }
}

impl TranslationUnit {
    pub fn rm_needless_mv(&mut self) {
        for function in self.functions.iter_mut() {
            let function = &mut function.body;
            function.rm_needless_mv();
        }
    }
}
