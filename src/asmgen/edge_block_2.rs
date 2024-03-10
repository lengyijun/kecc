use super::helper::Gape;
use crate::ir::{Block, BlockId, JumpArg, Operand};
use itertools::Itertools;
use std::ops::Deref;

impl Gape {
    /// deal with examples/c/cond.c
    /// found this bug in `test_examples_end_to_end`
    pub fn foo(mut self) -> Self {
        let mut block_id_index = self
            .blocks
            .keys()
            .fold(0, |acc, bid| usize::max(acc, bid.0))
            + 1;
        let mut next_block_id = || -> BlockId {
            let bid = BlockId(block_id_index);
            block_id_index += 1;
            bid
        };

        let blocks = self.blocks.clone();

        let mut new_blocks: Vec<(BlockId, Block)> = Vec::new();
        for (_, block) in self.blocks.iter_mut() {
            for (_, group) in block
                .exit
                .walk_jump_args_mut()
                .filter(|jump_arg| !jump_arg.args.is_empty())
                .group_by(|jump_arg| jump_arg.bid)
                .into_iter()
            {
                let mut v: Vec<_> = group.collect();
                if v.len() <= 1 {
                    continue;
                }
                for x in v.iter_mut().skip(1) {
                    let target_block = &blocks[&x.bid];
                    let new_bid = next_block_id();
                    let block = Block {
                        phinodes: target_block.phinodes.clone(),
                        instructions: Vec::new(),
                        exit: crate::ir::BlockExit::Jump {
                            arg: JumpArg {
                                bid: x.bid,
                                args: target_block
                                    .phinodes
                                    .iter()
                                    .enumerate()
                                    .map(|(aid, dtype)| Operand::Register {
                                        rid: crate::ir::RegisterId::Arg { bid: new_bid, aid },
                                        dtype: dtype.deref().clone(),
                                    })
                                    .collect(),
                            },
                        },
                    };
                    x.bid = new_bid;
                    new_blocks.push((new_bid, block));
                }
            }
        }

        self.blocks.extend(new_blocks);

        Self::new(self.blocks, self.bid_init, self.abi)
    }
}
