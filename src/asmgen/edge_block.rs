use crate::{
    ir::{Block, BlockId, JumpArg},
    SimplifyCfgReach,
};

use super::helper::Gape;

impl Gape {
    pub fn add_edge_block(mut self) -> Self {
        let mut id = self
            .blocks
            .iter()
            .fold(0, |acc, (bid, _)| usize::max(acc, bid.0))
            + 1;

        loop {
            if let Some((&bid, _)) = self
                .blocks
                .iter()
                .find(|(bid, _)| need_edge_block(&self, *self.block_mp.get_by_left(bid).unwrap()))
            {
                let mut v: Vec<(BlockId, Block)> = Vec::new();
                let blocks = self.blocks.clone();

                for jump_arg in self.blocks.get_mut(&bid).unwrap().exit.walk_jump_args_mut() {
                    let temp_bid = BlockId(id);
                    id += 1;
                    v.push((
                        temp_bid,
                        clone_block(
                            blocks.get(&jump_arg.bid).unwrap().clone(),
                            jump_arg.bid,
                            temp_bid,
                        ),
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
                // remove unreachable blocks
                let _ = SimplifyCfgReach::optimize_inner(self.bid_init, &mut self.blocks);
                self = Self::new(self.blocks, self.bid_init, self.abi);
            } else {
                return self;
            }
        }
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

fn clone_block(mut block: Block, from: BlockId, to: BlockId) -> Block {
    for x in block.walk_register_mut() {
        match x {
            crate::ir::RegisterId::Local { .. } => {}
            crate::ir::RegisterId::Arg { bid, .. } | crate::ir::RegisterId::Temp { bid, .. } => {
                if *bid == from {
                    *bid = to;
                }
            }
        }
    }
    block
}
