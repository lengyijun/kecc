use std::collections::{HashMap, HashSet};

use crate::ir::*;
use crate::*;

pub type Deadcode = FunctionPass<Repeat<DeadcodeInner>>;

#[derive(Default, Clone, Copy, Debug)]
pub struct DeadcodeInner {}

impl Optimize<FunctionDefinition> for DeadcodeInner {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let b3 = clear_nop(code);
        let b2 = clear_instruction(code);
        let b4 = clear_phinode(code);
        let b1 = clear_allocation(code);
        b1 || b2 || b3 || b4
    }
}

fn clear_nop(code: &mut FunctionDefinition) -> bool {
    let mut nop_instr: HashMap<BlockId, Vec<usize>> = HashMap::new();

    for (bid, block) in code.blocks.iter() {
        for (i, instr) in block.instructions.iter().enumerate() {
            if let ir::Instruction::Nop = &**instr {
                match nop_instr.entry(*bid) {
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        e.get_mut().push(i);
                    }
                    std::collections::hash_map::Entry::Vacant(v) => {
                        let _ = v.insert(vec![i]);
                    }
                }
            }
        }
    }

    {
        let nop_instr: HashSet<(BlockId, usize)> = nop_instr
            .clone()
            .into_iter()
            .flat_map(|(bid, v)| v.into_iter().map(move |x| (bid, x)))
            .collect();

        for operand in code.walk_operand_mut() {
            if let ir::Operand::Register {
                rid: ir::RegisterId::Temp { bid, iid },
                ..
            } = operand
            {
                if nop_instr.contains(&(*bid, *iid)) {
                    *operand = ir::Operand::Constant(ir::Constant::unit());
                }
            }
        }
    }

    shift_operand_and_rm_instr(code, nop_instr.clone())
}

fn clear_allocation(code: &mut FunctionDefinition) -> bool {
    let mut used_allocation: HashSet<usize> = HashSet::new();

    for operand in code.walk_operand_mut() {
        if let Operand::Register {
            rid: ir::RegisterId::Local { aid },
            ..
        } = operand
        {
            let _ = used_allocation.insert(*aid);
        }
    }

    let mut t: Vec<usize> = vec![0; code.allocations.len()];
    let mut should_remove: Vec<usize> = (0..code.allocations.len())
        .filter(|aid| !used_allocation.contains(aid))
        .collect();
    should_remove.sort();
    for i in &should_remove {
        t[*i] = 1;
    }
    if t.iter().all(|aid| *aid == 0) {
        return false;
    }
    for i in 1..t.len() {
        t[i] += t[i - 1];
    }

    let mut modified = false;

    for operand in code.walk_operand_mut() {
        if let Operand::Register {
            rid: ir::RegisterId::Local { aid },
            ..
        } = operand
        {
            if t[*aid] != 0 {
                *aid -= t[*aid];
                modified = true;
            }
        }
    }

    for x in should_remove.into_iter().rev() {
        let _x = code.allocations.remove(x);
    }

    modified
}

fn clear_instruction(code: &mut FunctionDefinition) -> bool {
    let mut used_opearand: HashSet<(BlockId, usize)> = HashSet::new();

    for (bid, block) in code.blocks.iter_mut() {
        for (i, instr) in block.instructions.iter_mut().enumerate() {
            if let ir::Instruction::Store { .. } | ir::Instruction::Call { .. } = &**instr {
                let _ = used_opearand.insert((*bid, i));
            }
        }
    }

    for operand in code.walk_operand_mut() {
        if let ir::Operand::Register {
            rid: ir::RegisterId::Temp { bid, iid },
            ..
        } = operand
        {
            let _ = used_opearand.insert((*bid, *iid));
        }
    }

    let all_temp_rid = code
        .blocks
        .iter()
        .flat_map(|(bid, block)| (0..block.instructions.len()).map(|i| (*bid, i)));

    let mut unused_instr: HashMap<BlockId, Vec<usize>> = HashMap::new();

    all_temp_rid
        .filter(|x| !used_opearand.contains(x))
        .for_each(|(bid, iid)| match unused_instr.entry(bid) {
            std::collections::hash_map::Entry::Occupied(mut o) => {
                o.get_mut().push(iid);
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                let _ = v.insert(vec![iid]);
            }
        });

    shift_operand_and_rm_instr(code, unused_instr)
}

fn shift_operand_and_rm_instr(
    code: &mut FunctionDefinition,
    mut unused_instr: HashMap<BlockId, Vec<usize>>,
) -> bool {
    for v in unused_instr.values_mut() {
        v.sort();
    }

    let mut modified = false;

    for (bid, v) in unused_instr.clone().into_iter() {
        let mut x = vec![0; code.blocks.get(&bid).unwrap().instructions.len()];
        for i in &v {
            x[*i] = 1;
        }
        for i in 1..x.len() {
            x[i] += x[i - 1];
        }

        for operand in code.walk_operand_mut() {
            if let ir::Operand::Register {
                rid: ir::RegisterId::Temp { bid: bid_1, iid },
                ..
            } = operand
            {
                if bid == *bid_1 && x[*iid] != 0 {
                    *iid -= x[*iid];
                    modified = true;
                }
            }
        }
    }

    for (bid, v) in &unused_instr {
        let b = &mut code.blocks.get_mut(bid).unwrap().instructions;
        for iid in v.iter().rev() {
            let _instr = b.remove(*iid);
            modified = true;
        }
    }

    modified
}

fn clear_phinode(code: &mut FunctionDefinition) -> bool {
    let mut modified = false;

    let mut used_phinode: HashSet<(BlockId, usize)> =
        (0..code.blocks.get(&code.bid_init).unwrap().phinodes.len())
            .map(|aid| (code.bid_init, aid))
            .collect();

    // collect all usage of phinode
    for operand in code.walk_operand_mut() {
        if let ir::Operand::Register {
            rid: ir::RegisterId::Arg { bid, aid },
            ..
        } = operand
        {
            let _ = used_phinode.insert((*bid, *aid));
        }
    }

    let mut unused_phinodes: HashMap<BlockId, Vec<usize>> = HashMap::new();

    for (bid, aid) in code
        .blocks
        .iter()
        .flat_map(|(bid, block)| (0..block.phinodes.len()).map(|x| (*bid, x)))
        .filter(|x| !used_phinode.contains(x))
    {
        match unused_phinodes.entry(bid) {
            std::collections::hash_map::Entry::Occupied(mut o) => {
                o.get_mut().push(aid);
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                let _ = v.insert(vec![aid]);
            }
        }
    }

    drop(used_phinode);

    let None = unused_phinodes.remove(&code.bid_init) else {
        unreachable!()
    };
    let unused_phinodes = unused_phinodes;

    for (bid, v) in &unused_phinodes {
        let mut x: Vec<usize> = vec![0; code.blocks.get(bid).unwrap().phinodes.len()];
        for aid in v {
            x[*aid] = 1;
        }
        for i in 1..x.len() {
            x[i] += x[i - 1];
        }
        for operand in code.walk_operand_mut() {
            if let ir::Operand::Register {
                rid: ir::RegisterId::Arg { bid: bid_1, aid },
                ..
            } = operand
            {
                if bid == bid_1 && x[*aid] != 0 {
                    *aid -= x[*aid];
                    modified = true;
                }
            }
        }
    }

    let pred = code.calculate_pred();

    for (bid, mut v) in unused_phinodes.into_iter() {
        v.sort();
        v.reverse();

        let block = code.blocks.get_mut(&bid).unwrap();
        for x in &v {
            let _x = block.phinodes.remove(*x);
        }

        let Some(prev_bids) = pred.get(&bid) else {
            continue;
        };
        for prev_bid in prev_bids {
            code.blocks
                .get_mut(prev_bid)
                .unwrap()
                .exit
                .walk_jump_args(|jump_arg| {
                    if jump_arg.bid == bid {
                        for x in &v {
                            let _x = jump_arg.args.remove(*x);
                        }
                    }
                });
        }
    }

    modified
}

impl FunctionDefinition {
    pub fn walk_operand_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut Operand> + 'a> {
        Box::new(
            self.blocks
                .iter_mut()
                .flat_map(|(_, b)| b.walk_operand_mut()),
        )
    }
}
