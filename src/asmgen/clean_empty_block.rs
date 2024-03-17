use crate::asm::{self, TranslationUnit};

impl asm::Function {
    fn rm_empty_block(&mut self) {
        while let Some((curr, next)) = self
            .blocks
            .iter()
            .skip(1)
            .zip(self.blocks.iter().skip(2))
            .find_map(|(curr, next)| {
                if curr.instructions.is_empty()
                    && let Some(curr) = &curr.label
                    && let Some(next) = &next.label
                {
                    Some((curr.clone(), next.clone()))
                } else {
                    None
                }
            })
        {
            for from in self.walk_label_mut() {
                if from == &curr {
                    *from = next.clone();
                }
            }
            self.blocks.retain(|b| match &b.label {
                Some(label) => label != &curr,
                None => true,
            });
        }
    }
}

impl TranslationUnit {
    pub fn rm_empty_block(&mut self) {
        for function in self.functions.iter_mut() {
            function.body.rm_empty_block();
        }
    }
}
