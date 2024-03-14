use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::DerefMut;

use crate::ir::*;
use crate::*;

use super::domtree::DomTree;

pub type Mem2reg = FunctionPass<Mem2regInner>;

#[derive(Default, Clone, Copy, Debug)]
pub struct Mem2regInner {}

impl Optimize<FunctionDefinition> for Mem2regInner {
    fn optimize(&mut self, code: &mut FunctionDefinition) -> bool {
        let rpo = code.reverse_post_order();
        let pred = code.calculate_pred();
        let dom = code.calculate_dominator(&pred);
        let idom = code.calculate_idom(&dom, &rpo);
        let df = dominance_frontier(&dom, &pred);

        // search location can't remove
        // used as a pointer
        let mut impromotable: HashSet<usize> = HashSet::new();

        let mut stores: HashMap<usize, Vec<BlockId>> = HashMap::new();

        for (bid, b) in &code.blocks {
            for instruction in &b.instructions {
                match &**instruction {
                    ir::Instruction::Nop => {}
                    ir::Instruction::BinOp { lhs, rhs, .. } => {
                        mark_impromotable(lhs, &mut impromotable);
                        mark_impromotable(rhs, &mut impromotable);
                    }
                    ir::Instruction::UnaryOp { operand, .. } => {
                        mark_impromotable(operand, &mut impromotable);
                    }
                    ir::Instruction::Store { ptr, value } => {
                        mark_impromotable(value, &mut impromotable);
                        if let ir::Operand::Register {
                            rid: ir::RegisterId::Local { aid },
                            ..
                        } = ptr
                        {
                            stores.entry(*aid).or_default().push(*bid);
                        }
                    }
                    ir::Instruction::Load { .. } => {}
                    ir::Instruction::Call { callee, args, .. } => {
                        mark_impromotable(callee, &mut impromotable);
                        for arg in args {
                            mark_impromotable(arg, &mut impromotable);
                        }
                    }
                    ir::Instruction::TypeCast { value, .. } => {
                        mark_impromotable(value, &mut impromotable);
                    }
                    ir::Instruction::GetElementPtr { ptr, offset, .. } => {
                        mark_impromotable(ptr, &mut impromotable);
                        mark_impromotable(offset, &mut impromotable);
                    }
                }
            }
        }
        if code.allocations.len() == impromotable.len() {
            return false;
        }
        for bid in impromotable.iter() {
            let _x = stores.remove(bid);
        }

        // can't mut
        let stores = stores;
        let impromotable = impromotable;

        // location -> blocks need phinodes
        // 并不是 join_1 join_2 里出现的 block, 都需要插入对应的 phinode
        // 只有当 block 里有 load 的时候, 才需要
        // 所以会过滤出一个  phinode_indexes
        let joins_1: HashMap<usize, HashSet<BlockId>> = stores
            .iter()
            .map(|(aid, bids)| {
                let mut vec = bids.clone();
                let mut visited: HashSet<BlockId> = HashSet::new();
                while let Some(bid) = vec.pop() {
                    for frontier in df.get(&bid).unwrap().iter() {
                        if visited.insert(*frontier) {
                            vec.push(*frontier);
                        }
                    }
                }
                (*aid, visited)
            })
            .collect();
        let joins_2: HashSet<(usize, BlockId)> = joins_1
            .into_iter()
            .flat_map(|(location, hs)| hs.into_iter().map(move |bid| (location, bid)))
            .collect();

        // must be subset of join_2
        let mut need_phinode: HashSet<(usize, BlockId)> = HashSet::new();

        let mut replaces: HashMap<RegisterId, OperandVar> = HashMap::new();

        let mut end_values: HashMap<(usize, BlockId), OperandVar> = HashMap::new();

        for bid in &rpo {
            let block = code.blocks.get(bid).unwrap();
            for (i, instr) in block.instructions.iter().enumerate() {
                match &**instr {
                    Instruction::Store {
                        ptr:
                            Operand::Register {
                                rid: ir::RegisterId::Local { aid },
                                ..
                            },
                        value,
                    } => {
                        if !impromotable.contains(aid) {
                            let _x =
                                end_values.insert((*aid, *bid), OperandVar::Operand(value.clone()));
                        }
                    }
                    Instruction::Load {
                        ptr:
                            Operand::Register {
                                rid: ir::RegisterId::Local { aid },
                                ..
                            },
                    } => {
                        if !impromotable.contains(aid) {
                            let mut vec: Vec<BlockId> = vec![];
                            let var = match end_values.get(&(*aid, *bid)).cloned() {
                                Some(v) => v,
                                None => {
                                    if joins_2.contains(&(*aid, *bid)) {
                                        let _b = need_phinode.insert((*aid, *bid));
                                        OperandVar::Phi(*aid, *bid)
                                    } else {
                                        let mut p = bid;
                                        let res: OperandVar = loop {
                                            match idom.get(p) {
                                                Some(x) => match end_values.get(&(*aid, *x)) {
                                                    Some(operand_var) => {
                                                        break operand_var.clone();
                                                    }
                                                    None => {
                                                        if joins_2.contains(&(*aid, *x)) {
                                                            let _b =
                                                                need_phinode.insert((*aid, *x));
                                                            break OperandVar::Phi(*aid, *x);
                                                        } else {
                                                            vec.push(*x);
                                                            p = x;
                                                        }
                                                    }
                                                },
                                                None => {
                                                    assert_eq!(p, &code.bid_init);
                                                    let dtype = code
                                                        .allocations
                                                        .get(*aid)
                                                        .unwrap()
                                                        .clone()
                                                        .into_inner();
                                                    let x =
                                                        OperandVar::Operand(ir::Operand::Constant(
                                                            ir::Constant::Undef { dtype },
                                                        ));
                                                    break x;
                                                }
                                            }
                                        };
                                        let None = end_values.insert((*aid, *bid), res.clone())
                                        else {
                                            unreachable!()
                                        };
                                        res
                                    }
                                }
                            };

                            for bid in vec.into_iter() {
                                let None = end_values.insert((*aid, bid), var.clone()) else {
                                    unreachable!()
                                };
                            }
                            let None = replaces.insert(RegisterId::Temp { bid: *bid, iid: i }, var)
                            else {
                                unreachable!()
                            };
                        }
                    }
                    _ => {}
                }
            }
        }

        // (location, block_id) -> offset
        let mut visited: HashSet<(usize, BlockId)> = HashSet::new();

        let mut btree_set: BTreeSet<(usize, BlockId)> = BTreeSet::new();
        let mut vec: Vec<(usize, BlockId)> = need_phinode.into_iter().collect();

        while let Some((aid, bid)) = vec.pop() {
            if visited.insert((aid, bid)) {
                let true = btree_set.insert((aid, bid)) else {
                    unreachable!()
                };
                match end_values.entry((aid, bid)) {
                    std::collections::hash_map::Entry::Occupied(o) => {
                        let x = o.get();
                        if let OperandVar::Phi(_, _) = x {
                            assert_eq!(x, &OperandVar::Phi(aid, bid));
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(v) => {
                        let _x = v.insert(OperandVar::Phi(aid, bid));
                    }
                }
                for pred_bid in pred.get(&bid).unwrap() {
                    let bid = find_nearest_end_value(
                        aid,
                        *pred_bid,
                        &end_values,
                        &idom,
                        &joins_2,
                        code.bid_init,
                    );
                    if bid == code.bid_init {
                    } else if !end_values.contains_key(&(aid, bid)) {
                        vec.push((aid, bid));
                    }
                }
            }
        }

        // value: Vec<location>
        // jump to bid need additional parameters
        let mut jiting: HashMap<BlockId, Vec<usize>> = HashMap::new();

        // used in OperandVar replacement
        let mut jiqian: HashMap<(usize, BlockId), usize> = HashMap::new();

        for (aid, bid) in btree_set {
            jiting.entry(bid).or_default().push(aid);
            let name = code.allocations.get(aid).unwrap().name();
            let dtype = code.allocations.get(aid).unwrap().clone().into_inner();
            code.blocks
                .get_mut(&bid)
                .unwrap()
                .phinodes
                .push(Named::new(name.cloned(), dtype));
            let None = jiqian.insert(
                (aid, bid),
                code.blocks.get(&bid).unwrap().phinodes.len() - 1,
            ) else {
                unreachable!()
            };
        }

        // only use the allocation part
        let code_clone = code.clone();

        // modify all jump
        for (bid, block) in code.blocks.iter_mut() {
            block.exit.walk_jump_args(|jump_arg| {
                if let Some(vec_aid) = jiting.get(&jump_arg.bid) {
                    let iter = vec_aid
                        .iter()
                        .map(|aid| {
                            end_values.get(&(*aid, *bid)).cloned().unwrap_or_else(|| {
                                let dtype = code_clone
                                    .allocations
                                    .get(*aid)
                                    .unwrap()
                                    .clone()
                                    .into_inner();
                                let mut p = bid;
                                loop {
                                    match idom.get(p) {
                                        Some(parent) => match end_values.get(&(*aid, *parent)) {
                                            Some(x) => break x.clone(),
                                            None => {
                                                p = parent;
                                            }
                                        },
                                        None => {
                                            assert_eq!(p, &code.bid_init);
                                            break OperandVar::Operand(ir::Operand::Constant(
                                                ir::Constant::Undef { dtype },
                                            ));
                                        }
                                    }
                                }
                            })
                        })
                        .map(|x| x.into_operand(&code_clone, &jiqian, &replaces));
                    jump_arg.args.extend(iter);
                }
            });
        }

        // utilize replacement

        for operand in code.walk_operand_mut() {
            if let Operand::Register { rid, .. } = operand {
                if let Some(v) = replaces.get(rid) {
                    *operand = v.clone().into_operand(&code_clone, &jiqian, &replaces);
                }
            }
        }

        // replace load and store
        for block in code.blocks.values_mut() {
            for instr in block.instructions.iter_mut() {
                match &**instr {
                    Instruction::Store {
                        ptr:
                            Operand::Register {
                                rid: ir::RegisterId::Local { aid },
                                ..
                            },
                        ..
                    } => {
                        if !impromotable.contains(aid) {
                            let x = instr.deref_mut();
                            *x = ir::Instruction::Nop;
                        }
                    }
                    Instruction::Load {
                        ptr:
                            Operand::Register {
                                rid: ir::RegisterId::Local { aid },
                                ..
                            },
                    } => {
                        if !impromotable.contains(aid) {
                            let x = instr.deref_mut();
                            *x = ir::Instruction::Nop;
                        }
                    }
                    _ => {}
                }
            }
        }

        true
    }
}

fn find_nearest_end_value(
    aid: usize,
    mut bid: BlockId,
    end_values: &HashMap<(usize, BlockId), OperandVar>,
    idom: &HashMap<BlockId, BlockId>,
    joins_2: &HashSet<(usize, BlockId)>,
    bid_init: BlockId,
) -> BlockId {
    loop {
        match end_values.get(&(aid, bid)) {
            Some(_) => return bid,
            None => {
                if joins_2.contains(&(aid, bid)) {
                    return bid;
                }
                match idom.get(&bid) {
                    Some(p) => {
                        bid = *p;
                    }
                    None => {
                        assert_eq!(bid, bid_init);
                        return bid_init;
                    }
                }
            }
        }
    }
}

fn mark_impromotable(lhs: &Operand, impromotable: &mut HashSet<usize>) {
    if let Operand::Register {
        rid: ir::RegisterId::Local { aid },
        ..
    } = lhs
    {
        let _b = impromotable.insert(*aid);
    }
}

/*
impl FunctionDefinition {
    pub fn dom_tree(&self) -> HashMap<BlockId, Vec<BlockId>> {
        let rpo = self.reverse_post_order();
        let pred = self.calculate_pred();
        let dom = calculate_dominator(self, &pred);
        let idom = self.calculate_idom(&dom, rpo);

        let mut res: HashMap<BlockId, Vec<BlockId>> = HashMap::new();

        for (child, parent) in idom.into_iter() {
            match res.entry(parent) {
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    o.get_mut().push(child);
                }
                std::collections::hash_map::Entry::Vacant(x) => {
                    let _ = x.insert(vec![child]);
                }
            }
        }

        res
    }
}
 */

fn dominance_frontier(
    dom: &HashMap<BlockId, HashSet<BlockId>>,
    pred: &HashMap<BlockId, HashSet<BlockId>>,
) -> HashMap<BlockId, HashSet<BlockId>> {
    let mut dom = dom.clone();
    for (k, v) in dom.iter_mut() {
        let true = v.remove(k) else { unreachable!() };
    }
    let mut df: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
    for x in dom.keys() {
        let mut hs: HashSet<BlockId> = HashSet::new();
        for (y, dom_y) in &dom {
            if !dom_y.contains(x) {
                let mut b = false;
                for z in pred.get(y).unwrap_or_else(|| panic!("failed to find {y}")) {
                    if dom.get(z).unwrap().contains(x) || x == z {
                        b = true;
                    }
                }
                if b {
                    let true = hs.insert(*y) else { unreachable!() };
                }
            }
        }
        let None = df.insert(*x, hs) else {
            unreachable!()
        };
    }
    df
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum OperandVar {
    Operand(Operand),
    Phi(usize, BlockId),
}

impl OperandVar {
    fn into_operand(
        self,
        code: &FunctionDefinition,
        jiqian: &HashMap<(usize, BlockId), usize>,
        replaces: &HashMap<RegisterId, OperandVar>,
    ) -> Operand {
        match self {
            OperandVar::Operand(mut o) => loop {
                match &o {
                    Operand::Constant(_) => return o,
                    Operand::Register { rid, .. } => match replaces.get(rid) {
                        Some(x) => o = x.clone().into_operand(code, jiqian, replaces),
                        None => return o,
                    },
                }
            },
            OperandVar::Phi(aid, bid) => {
                let offset = jiqian.get(&(aid, bid)).unwrap();
                Operand::Register {
                    rid: RegisterId::Arg { bid, aid: *offset },
                    dtype: code.allocations.get(aid).unwrap().clone().into_inner(),
                }
            }
        }
    }
}
