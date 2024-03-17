use crate::asm::{self, TranslationUnit};

impl asm::Function {
    fn rm_needless_jump(&mut self) {
        let next_label: Vec<_> = self
            .blocks
            .iter()
            .skip(1)
            .map(|b| &b.label)
            .cloned()
            .collect();
        for (curr, next_label) in self.blocks.iter_mut().zip(next_label.into_iter()) {
            let Some(next_label) = next_label else {
                continue;
            };
            let Some(exit) = curr.instructions.last() else {
                continue;
            };
            match exit {
                asm::Instruction::Pseudo(asm::Pseudo::J { offset }) => {
                    if *offset == next_label {
                        let _j = curr.instructions.pop().unwrap();
                    }
                }
                _ => {}
            }
        }
    }
}

impl TranslationUnit {
    pub fn rm_needless_jump(&mut self) {
        for function in self.functions.iter_mut() {
            function.body.rm_needless_jump();
        }
    }
}
