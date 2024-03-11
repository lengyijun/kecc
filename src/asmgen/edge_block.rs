use crate::{
    ir::{Block, BlockExit, BlockId, JumpArg},
    SimplifyCfgReach,
};

use super::helper::Gape;

impl Gape {
    /// avoid regalloc2::RegAllocError::DisallowedBranchArg
    pub fn add_edge_block(mut self) -> Self {
        let max_bid =
            |x: &Self| -> usize { x.blocks.keys().fold(0, |a, BlockId(b)| usize::max(a, *b)) };
        loop {
            let x = max_bid(&self);
            self = self.add_edge_block_inner();
            match max_bid(&self).cmp(&x) {
                std::cmp::Ordering::Less => unreachable!(),
                std::cmp::Ordering::Equal => return self,
                std::cmp::Ordering::Greater => {
                    // continue
                }
            }
        }
    }

    fn add_edge_block_inner(mut self) -> Self {
        let mut id = self
            .blocks
            .iter()
            .fold(0, |acc, (bid, _)| usize::max(acc, bid.0))
            + 1;

        // 1. deal with branch
        // by inserting empty blocks
        loop {
            if let Some((&bid, _)) = self.blocks.iter().find(|(bid, bb)| {
                need_edge_block(&self, *self.block_mp.get_by_left(bid).unwrap())
                    && matches!(
                        bb.exit,
                        BlockExit::Switch { .. } | BlockExit::ConditionalJump { .. }
                    )
            }) {
                let mut v: Vec<(BlockId, Block)> = Vec::new();

                for jump_arg in self.blocks.get_mut(&bid).unwrap().exit.walk_jump_args_mut() {
                    let temp_bid = BlockId(id);
                    id += 1;
                    v.push((
                        temp_bid,
                        Block {
                            phinodes: Vec::new(),
                            instructions: Vec::new(),
                            exit: BlockExit::Jump {
                                arg: jump_arg.clone(),
                            },
                        },
                    ));
                    *jump_arg = JumpArg {
                        bid: temp_bid,
                        args: Vec::new(),
                    };
                }
                for (bid, block) in v {
                    let None = self.blocks.insert(bid, block) else {
                        unreachable!()
                    };
                }
                self = Self::new(self.blocks, self.bid_init, self.abi);
            } else {
                break;
            }
        }
        self
    }
}

fn need_edge_block<F>(f: &F, block: regalloc2::Block) -> bool
where
    F: regalloc2::Function,
{
    let mut require_no_branch_args = false;
    for &succ in f.block_succs(block) {
        let preds = f.block_preds(succ).len() + if succ == f.entry_block() { 1 } else { 0 };
        if preds > 1 {
            require_no_branch_args = true;
        }
    }
    if require_no_branch_args {
        let last = f.block_insns(block).last();
        if !f.inst_operands(last).is_empty() {
            return true;
        }
    }
    false
}
