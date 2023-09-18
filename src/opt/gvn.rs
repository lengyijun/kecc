use std::collections::HashMap;
use std::hash::Hash;

use lang_c::ast::{BinaryOperator, UnaryOperator};

use crate::ir::{BlockId, Dtype, Operand, RegisterId};
use crate::opt::{mem2reg, FunctionPass};
use crate::*;

pub type Gvn = FunctionPass<GvnInner>;

#[derive(Default, Clone, Copy, Debug)]
pub struct GvnInner {}

impl Optimize<ir::FunctionDefinition> for GvnInner {
    fn optimize(&mut self, code: &mut ir::FunctionDefinition) -> bool {
        let mut modified = false;

        let mut num: Num = Num(0);

        let mut register_table: HashMap<RegisterId, NumOrConstant> = HashMap::new();
        let mut expression_table: HashMap<Expr<NumOrConstant>, Num> = HashMap::new();
        // TODO: only save the last one
        let mut leader_tables: HashMap<BlockId, Vec<HashMap<Num, Operand>>> = HashMap::new();

        let rpo = code.reverse_post_order();
        let pred = code.calculate_pred();
        let dom = mem2reg::calculate_dominator(code, &pred);
        let idom = mem2reg::calculate_idom(code, &dom, rpo.clone());

        for bid in rpo.into_iter() {
            let mut leader_table_vec: Vec<HashMap<Num, Operand>> = match idom.get(&bid) {
                Some(idom_bid) => {
                    vec![leader_tables.get(idom_bid).unwrap().last().unwrap().clone()]
                }
                None => {
                    vec![HashMap::new()]
                }
            };

            let mut phinode_offset: HashMap<BlockId, usize> = code
                .blocks
                .iter()
                .map(|(bid, block)| (*bid, block.phinodes.len()))
                .collect();
            let mut phinode_vec: Vec<(Num, Dtype)> = Vec::new();

            let block = code.blocks.get(&bid).unwrap();

            for (aid, phinode) in block.phinodes.iter().enumerate() {
                let rid = ir::RegisterId::Arg { bid, aid };
                let operand = {
                    let vv: Vec<Option<Operand>> = pred
                        .get(&bid)
                        .unwrap()
                        .iter()
                        .flat_map(|prev_bid| {
                            let mut exit = code.blocks.get(prev_bid).unwrap().exit.clone();
                            let mut v = vec![];
                            exit.walk_jump_args(|jump_arg| {
                                if jump_arg.bid == bid {
                                    match &jump_arg.args[aid] {
                                        Operand::Constant(c) => {
                                            v.push(Some(Operand::Constant(c.clone())))
                                        }
                                        Operand::Register { rid, .. } => {
                                            let x = register_table.get(rid).and_then(
                                                |num_or_constant| match num_or_constant {
                                                    NumOrConstant::Num(num) => leader_tables
                                                        .get(prev_bid)
                                                        .unwrap()
                                                        .last()
                                                        .unwrap()
                                                        .get(num)
                                                        .cloned(),
                                                    NumOrConstant::Constant(c) => {
                                                        Some(ir::Operand::Constant(c.clone()))
                                                    }
                                                },
                                            );
                                            v.push(x);
                                        }
                                    }
                                }
                            });
                            v.into_iter()
                        })
                        .collect();

                    match consists_only_one_value(vv) {
                        ListItem::Empty => {
                            assert_eq!(bid, code.bid_init);
                            ir::Operand::Register {
                                rid,
                                dtype: phinode.clone().into_inner(),
                            }
                        }
                        ListItem::Same(Some(operand)) => operand,
                        ListItem::Same(None) | ListItem::Different => ir::Operand::Register {
                            rid,
                            dtype: phinode.clone().into_inner(),
                        },
                    }
                };
                match operand {
                    Operand::Constant(c) => {
                        let None = register_table.insert(rid, NumOrConstant::Constant(c)) else {unreachable!()};
                    }
                    Operand::Register { .. } => {
                        let num = num.next();
                        let None = register_table.insert(rid, NumOrConstant::Num(num)) else {unreachable!()};
                        let None = leader_table_vec.last_mut().unwrap().insert(num, operand ) else {unreachable!()};
                    }
                }
            }

            for (i, instr) in block.instructions.iter().enumerate() {
                leader_table_vec.push(leader_table_vec.last().unwrap().clone());
                let leader_table = leader_table_vec.last_mut().unwrap();
                let (e, dtype) = match &**instr {
                    // not sure BinaryOperator::Assign
                    ir::Instruction::BinOp {
                        op,
                        lhs,
                        rhs,
                        dtype,
                    } => {
                        let x1 = operand2num(lhs, &mut num, &mut register_table);
                        let x2 = operand2num(rhs, &mut num, &mut register_table);
                        let e = Expr::binop(op.clone(), x1, x2);
                        (e, dtype.clone())
                    }
                    ir::Instruction::TypeCast {
                        value,
                        target_dtype,
                    } => {
                        let value = operand2num(value, &mut num, &mut register_table);
                        let e = Expr::TypeCast {
                            value,
                            target_dtype: target_dtype.clone(),
                        };
                        (e, target_dtype.clone())
                    }
                    ir::Instruction::GetElementPtr { ptr, offset, dtype } => {
                        let ptr = operand2num(ptr, &mut num, &mut register_table);
                        let offset = operand2num(offset, &mut num, &mut register_table);
                        let e = Expr::GetElementPtr {
                            ptr,
                            offset,
                            dtype: dtype.clone(),
                        };
                        (e, dtype.clone())
                    }
                    ir::Instruction::UnaryOp {
                        op: UnaryOperator::Minus,
                        operand,
                        dtype,
                    } => {
                        let operand = operand2num(operand, &mut num, &mut register_table);
                        let e = Expr::UnaryOp {
                            op: UnaryOperator::Minus,
                            operand,
                        };
                        (e, dtype.clone())
                    }
                    ir::Instruction::UnaryOp {
                        op: UnaryOperator::Plus,
                        operand,
                        dtype,
                    } => {
                        let operand = operand2num(operand, &mut num, &mut register_table);
                        let e = Expr::UnaryOp {
                            op: UnaryOperator::Plus,
                            operand,
                        };
                        (e, dtype.clone())
                    }
                    _ => {
                        continue;
                    }
                };
                let operand = ir::Operand::Register {
                    rid: ir::RegisterId::Temp { bid, iid: i },
                    dtype: dtype.clone(),
                };
                let num = match expression_table.entry(e.clone()) {
                    std::collections::hash_map::Entry::Occupied(o) => {
                        let num = *o.get();
                        match leader_table.entry(num) {
                            std::collections::hash_map::Entry::Occupied(o) => {
                                assert_ne!(o.get(), &operand);
                            }
                            std::collections::hash_map::Entry::Vacant(v) => {
                                let vv: Vec<Option<Operand>> = pred
                                    .get(&bid)
                                    .unwrap()
                                    .iter()
                                    .map(|prev_bid| {
                                        leader_tables
                                            .get(prev_bid)
                                            .unwrap()
                                            .last()
                                            .unwrap()
                                            .get(&num)
                                            .cloned()
                                    })
                                    .collect();
                                match consists_only_one_value(vv.clone()) {
                                    ListItem::Empty => {
                                        assert_eq!(bid, code.bid_init);
                                        let _ = v.insert(operand);
                                    }
                                    ListItem::Same(Some(operand)) => {
                                        for x in leader_table_vec.iter_mut() {
                                            let None = x.insert(num, operand.clone()) else {unreachable!()};
                                        }
                                    }
                                    ListItem::Same(None) => {
                                        let _ = v.insert(operand);
                                    }
                                    ListItem::Different => {
                                        if vv.into_iter().all(|x| x.is_some()) {
                                            let aid = *phinode_offset.get(&bid).unwrap();
                                            let _ = phinode_offset.insert(bid, aid + 1).unwrap();
                                            for x in leader_table_vec.iter_mut() {
                                                let None = x.insert(num,
                                                    Operand::Register {
                                                        rid: RegisterId::arg(bid, aid),
                                                        dtype: dtype.clone(),
                                                    }
                                                ) else {unreachable!()};
                                            }
                                            phinode_vec.push((num, dtype.clone()));
                                        } else {
                                            let _ = v.insert(operand);
                                        }
                                    }
                                }
                            }
                        }
                        num
                    }
                    std::collections::hash_map::Entry::Vacant(v) => {
                        let num = num.next();
                        let _ = v.insert(num);
                        let None = leader_table.insert(
                                        num,
                                        operand
                                    ) else {unreachable!()};
                        num
                    }
                };
                assert!(leader_table_vec.last_mut().unwrap().get(&num).is_some());
                let None = register_table
                                    .insert(ir::RegisterId::Temp { bid, iid: i }, NumOrConstant::Num(num)) else {unreachable!()};
            }

            for (num, dtype) in phinode_vec.into_iter() {
                let block = code.blocks.get_mut(&bid).unwrap();
                block.phinodes.push(ir::Named::new(None, dtype));
                for prev_bid in pred.get(&bid).unwrap() {
                    let prev_block = code.blocks.get_mut(prev_bid).unwrap();
                    prev_block.exit.walk_jump_args(|jump_arg| {
                        if jump_arg.bid == bid {
                            match leader_tables
                                .get(prev_bid)
                                .unwrap()
                                .last()
                                .unwrap()
                                .get(&num)
                            {
                                Some(x) => {
                                    jump_arg.args.push(x.clone());
                                    modified = true;
                                }
                                None => {
                                    panic!("{:?}", num);
                                }
                            }
                        }
                    });
                }
            }

            let leader_table_vec = leader_table_vec;

            let block = code.blocks.get_mut(&bid).unwrap();
            for (i, instr) in block.instructions.iter_mut().enumerate() {
                let leader_table = &leader_table_vec[i];
                for operand in instr.walk_operand_mut() {
                    if let ir::Operand::Register { rid, .. } = operand {
                        match register_table.get(rid) {
                            Some(NumOrConstant::Constant(c)) => {
                                *operand = ir::Operand::Constant(c.clone());
                                modified = true;
                            }
                            Some(NumOrConstant::Num(num)) => {
                                if let Some(replacement) = leader_table.get(num) {
                                    if operand != replacement {
                                        *operand = replacement.clone();
                                        modified = true;
                                    }
                                }
                            }
                            None => {}
                        }
                    }
                }
            }
            {
                let leader_table = leader_table_vec.last().unwrap();
                for operand in block.exit.walk_operand_mut() {
                    if let ir::Operand::Register { rid, .. } = operand {
                        match register_table.get(rid) {
                            Some(NumOrConstant::Constant(c)) => {
                                *operand = ir::Operand::Constant(c.clone());
                                modified = true;
                            }
                            Some(NumOrConstant::Num(num)) => {
                                if let Some(replacement) = leader_table.get(num) {
                                    if operand != replacement {
                                        *operand = replacement.clone();
                                        modified = true;
                                    }
                                }
                            }
                            None => {}
                        }
                    }
                }
                block.exit.walk_jump_args(|jump_arg| {
                    for operand in jump_arg.args.iter_mut() {
                        if let ir::Operand::Register { rid, .. } = operand {
                            match register_table.get(rid) {
                                Some(NumOrConstant::Constant(c)) => {
                                    *operand = ir::Operand::Constant(c.clone());
                                    modified = true;
                                }
                                Some(NumOrConstant::Num(num)) => {
                                    if let Some(replacement) = leader_table.get(num) {
                                        if operand != replacement {
                                            *operand = replacement.clone();
                                            modified = true;
                                        }
                                    }
                                }
                                None => {}
                            }
                        }
                    }
                });
            }

            let None = leader_tables.insert(bid, leader_table_vec) else {unreachable!()};
        }

        modified
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Num(usize);

impl Num {
    fn next(&mut self) -> Self {
        self.0 += 1;
        *self
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum NumOrConstant {
    Num(Num),
    Constant(ir::Constant),
}
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum Expr<T: Clone + Hash + Eq + Ord> {
    BinOp { op: BinaryOperator, lhs: T, rhs: T },
    UnaryOp { op: UnaryOperator, operand: T },
    TypeCast { value: T, target_dtype: Dtype },
    GetElementPtr { ptr: T, offset: T, dtype: Dtype },
}

impl<T: Clone + Hash + Eq + Ord> Expr<T> {
    fn binop(op: BinaryOperator, lhs: T, rhs: T) -> Self {
        match &op {
            BinaryOperator::Plus
            | BinaryOperator::Multiply
            | BinaryOperator::Equals
            | BinaryOperator::NotEquals
            | BinaryOperator::BitwiseAnd
            | BinaryOperator::BitwiseXor
            | BinaryOperator::BitwiseOr
            | BinaryOperator::LogicalAnd
            | BinaryOperator::LogicalOr => {
                let mut v = [lhs, rhs];
                v.sort();
                let [lhs, rhs] = v;
                Self::BinOp { op, lhs, rhs }
            }

            _ => Self::BinOp { op, lhs, rhs },
        }
    }
}

fn operand2num(
    operand: &Operand,
    num: &mut Num,
    register_table: &mut HashMap<RegisterId, NumOrConstant>,
) -> NumOrConstant {
    match operand {
        ir::Operand::Constant(c) => NumOrConstant::Constant(c.clone()),
        ir::Operand::Register { rid, .. } => match register_table.entry(*rid) {
            std::collections::hash_map::Entry::Occupied(o) => {
                return o.get().clone();
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                let num = num.next();
                let res = NumOrConstant::Num(num);
                let _x = v.insert(res.clone());
                res
            }
        },
    }
}

enum ListItem<T> {
    Empty,
    Same(T),
    Different,
}

fn consists_only_one_value<T: Eq>(mut v: Vec<T>) -> ListItem<T> {
    if v.is_empty() {
        ListItem::Empty
    } else if v.iter().all(|x| x == &v[0]) {
        ListItem::Same(v.pop().unwrap())
    } else {
        ListItem::Different
    }
}
