use std::collections::{BTreeMap, HashSet, VecDeque};

use crate::ir::*;
use crate::opt::FunctionPass;
use crate::*;

pub type SimplifyCfg = FunctionPass<
    Repeat<(
        SimplifyCfgConstProp,
        (SimplifyCfgReach, (SimplifyCfgMerge, SimplifyCfgEmpty)),
    )>,
>;

/// Simplifies block exits by propagating constants.
#[derive(Default, Clone, Copy, Debug)]
pub struct SimplifyCfgConstProp {}

/// Retains only those blocks that are reachable from the init.
#[derive(Default, Clone, Copy, Debug)]
pub struct SimplifyCfgReach {}

/// Merges two blocks if a block is pointed to only by another
#[derive(Default, Clone, Copy, Debug)]
pub struct SimplifyCfgMerge {}

/// Removes empty blocks
#[derive(Default, Clone, Copy, Debug)]
pub struct SimplifyCfgEmpty {}

impl Optimize<FunctionDefinition> for SimplifyCfgConstProp {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let mut res = false;
        'outer: for block in code.blocks.values_mut() {
            match &block.exit {
                BlockExit::ConditionalJump {
                    condition: ir::Operand::Constant(ir::Constant::Int { value, .. }),
                    arg_then,
                    arg_else,
                } => {
                    res = true;
                    if (value & 1) == 1 {
                        block.exit = BlockExit::Jump {
                            arg: arg_then.clone(),
                        };
                    } else {
                        block.exit = BlockExit::Jump {
                            arg: arg_else.clone(),
                        };
                    }
                }
                BlockExit::ConditionalJump {
                    arg_then, arg_else, ..
                } => {
                    if arg_then == arg_else {
                        res = true;
                        block.exit = BlockExit::Jump {
                            arg: arg_then.clone(),
                        };
                    }
                }
                BlockExit::Switch {
                    value: ir::Operand::Constant(c),
                    default,
                    cases,
                } => {
                    res = true;
                    for (x, arg) in cases {
                        if c == x {
                            block.exit = BlockExit::Jump { arg: arg.clone() };
                            continue 'outer;
                        }
                    }
                    block.exit = BlockExit::Jump {
                        arg: default.clone(),
                    };
                }
                BlockExit::Switch { default, cases, .. } => {
                    let b = cases
                        .iter()
                        .map(|(_, jump_arg)| jump_arg == default)
                        .all(|x| x);
                    if b {
                        res = true;
                        block.exit = BlockExit::Jump {
                            arg: default.clone(),
                        };
                    }
                }
                _ => {}
            }
        }
        res
    }
}

impl Optimize<FunctionDefinition> for SimplifyCfgReach {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let mut queue: VecDeque<BlockId> = VecDeque::new();
        queue.push_back(code.bid_init);
        let mut set: HashSet<BlockId> = HashSet::new();

        while let Some(bid) = queue.pop_front() {
            if !set.insert(bid) {
                continue;
            }
            match &code.blocks.get(&bid).unwrap().exit {
                BlockExit::Jump { arg } => queue.push_back(arg.bid),
                BlockExit::ConditionalJump {
                    arg_then, arg_else, ..
                } => {
                    queue.push_back(arg_then.bid);
                    queue.push_back(arg_else.bid);
                }
                BlockExit::Switch { cases, default, .. } => {
                    for (_, jump_arg) in cases {
                        queue.push_back(jump_arg.bid);
                    }
                    queue.push_back(default.bid);
                }
                BlockExit::Return { .. } | BlockExit::Unreachable => {}
            }
        }

        let unreachable_bids = code
            .blocks
            .keys()
            .filter(|bid| !set.contains(bid))
            .cloned()
            .collect::<Vec<_>>();
        if unreachable_bids.is_empty() {
            return false;
        }

        for bid in unreachable_bids {
            let _x = code.blocks.remove(&bid).unwrap();
        }
        true
    }
}

impl Optimize<FunctionDefinition> for SimplifyCfgMerge {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let mut prev: BTreeMap<BlockId, Option<(BlockId, Vec<Operand>)>> = BTreeMap::new();
        for (bid, block) in &code.blocks {
            match &block.exit {
                BlockExit::Jump { arg } => match prev.entry(arg.bid) {
                    std::collections::btree_map::Entry::Vacant(e) => {
                        let _x = e.insert(Some((*bid, arg.args.clone())));
                    }
                    std::collections::btree_map::Entry::Occupied(mut e) => {
                        let _x = e.insert(None);
                    }
                },
                BlockExit::ConditionalJump {
                    arg_then, arg_else, ..
                } => {
                    let _x = prev.insert(arg_then.bid, None);
                    let _x = prev.insert(arg_else.bid, None);
                }
                BlockExit::Switch { default, cases, .. } => {
                    let _x = prev.insert(default.bid, None);
                    for (_, jump_arg) in cases {
                        let _x = prev.insert(jump_arg.bid, None);
                    }
                }
                BlockExit::Return { .. } | BlockExit::Unreachable => {}
            }
        }

        let mut res = false;
        let iter = prev.into_iter().filter_map(|(bid, a)| a.map(|x| (bid, x)));
        for (next_bid, (prev_bid, args)) in iter {
            let Some(mut next_block ) = code.blocks.remove(&next_bid) else {continue };
            let Some(prev_block ) = code.blocks.get_mut(&prev_bid) else {
                let None = code.blocks.insert(next_bid, next_block) else {unreachable!()};
                continue
            };
            res = true;

            for operand in next_block
                .instructions
                .iter_mut()
                .flat_map(|x| x.walk_operand_mut())
                .chain(next_block.exit.walk_operand_mut())
            {
                match operand {
                    Operand::Register {
                        rid: ir::RegisterId::Arg { bid, aid },
                        ..
                    } => {
                        if *bid == next_bid {
                            *operand = args[*aid].clone();
                        }
                    }
                    Operand::Register {
                        rid: ir::RegisterId::Temp { bid, iid },
                        dtype,
                    } => {
                        if *bid == next_bid {
                            *operand = Operand::Register {
                                rid: ir::RegisterId::Temp {
                                    bid: prev_bid,
                                    iid: prev_block.instructions.len() + *iid,
                                },
                                dtype: dtype.clone(),
                            };
                        }
                    }
                    _ => {}
                }
            }

            prev_block.exit = next_block.exit;
            prev_block.instructions.extend(next_block.instructions);
        }
        res
    }
}

impl Optimize<FunctionDefinition> for SimplifyCfgEmpty {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let blocks_ro = code.blocks.clone();

        let mut res = false;

        for block in code.blocks.values_mut() {
            match &mut block.exit {
                BlockExit::Jump { arg: arg1 } => {
                    let block_next = blocks_ro.get(&arg1.bid).unwrap();
                    if !block_next.phinodes.is_empty() || !block_next.instructions.is_empty() {
                        continue;
                    }
                    match &block_next.exit {
                        BlockExit::Jump { arg: arg2 } => {
                            if arg2.bid == arg1.bid {
                                continue;
                            }
                            block.exit = block_next.exit.clone();
                            res = true;
                        }
                        BlockExit::ConditionalJump { .. }
                        | BlockExit::Switch { .. }
                        | BlockExit::Return { .. }
                        | BlockExit::Unreachable => {
                            block.exit = block_next.exit.clone();
                            res = true;
                        }
                    }
                }
                BlockExit::ConditionalJump {
                    arg_then, arg_else, ..
                } => {
                    for b in [arg_then, arg_else]
                        .into_iter()
                        .map(|x| merge_jump(x, &blocks_ro))
                    {
                        if b {
                            res = true;
                        }
                    }
                }
                BlockExit::Switch { default, cases, .. } => {
                    for b in cases
                        .iter_mut()
                        .map(|(_, x)| x)
                        .chain(std::iter::once(default))
                        .map(|x| merge_jump(x, &blocks_ro))
                    {
                        if b {
                            res = true;
                        }
                    }
                }
                BlockExit::Return { .. } | BlockExit::Unreachable => {}
            }
        }
        res
    }
}

impl Instruction {
    pub fn walk_operand_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut Operand> + 'a> {
        match self {
            ir::Instruction::Nop => Box::new(std::iter::empty()),
            ir::Instruction::BinOp { lhs, rhs, .. } => {
                Box::new(std::iter::once(lhs).chain(std::iter::once(rhs)))
            }
            ir::Instruction::UnaryOp { operand, .. } => Box::new(std::iter::once(operand)),
            ir::Instruction::Store { ptr, value } => {
                Box::new(std::iter::once(ptr).chain(std::iter::once(value)))
            }
            ir::Instruction::Load { ptr } => Box::new(std::iter::once(ptr)),
            ir::Instruction::Call { callee, args, .. } => {
                Box::new(std::iter::once(callee).chain(args.iter_mut()))
            }
            ir::Instruction::TypeCast { value, .. } => Box::new(std::iter::once(value)),
            ir::Instruction::GetElementPtr { ptr, offset, .. } => {
                Box::new(std::iter::once(ptr).chain(std::iter::once(offset)))
            }
        }
    }
}

impl BlockExit {
    pub fn walk_operand_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut Operand> + 'a> {
        match self {
            BlockExit::Jump { .. } => Box::new(std::iter::empty()),
            BlockExit::ConditionalJump { condition, .. } => Box::new(std::iter::once(condition)),
            BlockExit::Switch { value, .. } => Box::new(std::iter::once(value)),
            BlockExit::Return { value } => Box::new(std::iter::once(value)),
            BlockExit::Unreachable => Box::new(std::iter::empty()),
        }
    }
}

// return true: modified
// return false: unmodified
fn merge_jump(jump_arg: &mut JumpArg, blocks_ro: &BTreeMap<BlockId, Block>) -> bool {
    let block_next = blocks_ro.get(&jump_arg.bid).unwrap();
    if !block_next.phinodes.is_empty() || !block_next.instructions.is_empty() {
        return false;
    }
    match &block_next.exit {
        BlockExit::Jump { arg: arg2 } => {
            if arg2.bid == jump_arg.bid {
                return false;
            }
            *jump_arg = arg2.clone();
            true
        }
        BlockExit::ConditionalJump { .. }
        | BlockExit::Switch { .. }
        | BlockExit::Return { .. }
        | BlockExit::Unreachable => false,
    }
}
