use std::{
    collections::{BTreeMap, HashMap},
    iter::{empty, once},
    ops::Deref,
};

use bimap::BiMap;
use frozen::Frozen;
use itertools::{izip, Itertools};

use crate::{
    asm::{self, DataSize, Pseudo, Register, RegisterType},
    asmgen::{clown, DirectOrInDirect, RegOrStack, RegisterCouple},
    ir::{
        self, BlockExit, BlockId, Constant, Dtype, FunctionDefinition, HasDtype, JumpArg, Named,
        Operand, RegisterId,
    },
    opt::domtree::{calculate_pred_inner, DomTree},
    SimplifyCfgReach,
};

use super::{load_float_to_reg, load_int_to_reg, FloatMp, FunctionAbi, ParamAlloc};

impl Into<regalloc2::RegClass> for RegisterType {
    fn into(self) -> regalloc2::RegClass {
        match self {
            RegisterType::Integer => regalloc2::RegClass::Int,
            RegisterType::FloatingPoint => regalloc2::RegClass::Float,
        }
    }
}

impl From<regalloc2::RegClass> for RegisterType {
    fn from(value: regalloc2::RegClass) -> Self {
        match value {
            regalloc2::RegClass::Int => Self::Integer,
            regalloc2::RegClass::Float => Self::FloatingPoint,
            regalloc2::RegClass::Vector => unreachable!(),
        }
    }
}

impl Into<regalloc2::PReg> for Register {
    fn into(self) -> regalloc2::PReg {
        match self {
            Register::Zero => unreachable!(),
            Register::Ra => unreachable!(),
            Register::Sp => unreachable!(),
            Register::Gp => unreachable!(),
            Register::Tp => unreachable!(),
            Register::Temp(register_type, offset) => {
                regalloc2::PReg::new(offset << 2, register_type.into())
            }
            Register::Saved(register_type, offset) => {
                regalloc2::PReg::new(offset << 2 | 1, register_type.into())
            }
            Register::Arg(register_type, offset) => {
                regalloc2::PReg::new(offset << 2 | 2, register_type.into())
            }
        }
    }
}
impl From<regalloc2::PReg> for Register {
    fn from(value: regalloc2::PReg) -> Self {
        let hw_enc = value.hw_enc();
        match hw_enc & 0b11 {
            0 => Register::Temp(value.class().into(), hw_enc >> 2),
            1 => Register::Saved(value.class().into(), hw_enc >> 2),
            2 => Register::Arg(value.class().into(), hw_enc >> 2),
            _ => unreachable!(),
        }
    }
}

impl TryInto<regalloc2::RegClass> for Dtype {
    type Error = ();

    fn try_into(self) -> Result<regalloc2::RegClass, Self::Error> {
        match self {
            Dtype::Int { .. } => Ok(regalloc2::RegClass::Int),
            Dtype::Float { .. } => Ok(regalloc2::RegClass::Float),
            Dtype::Pointer { .. }
            | Dtype::Struct { .. }
            | Dtype::Array { .. }
            | Dtype::Function { .. }
            | Dtype::Typedef { .. }
            | Dtype::Unit { .. } => Err(()),
        }
    }
}

/// Yank is bimap to regalloc2::Inst
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum Yank {
    /// init_block: deal with arguments
    /// otherwise: do nothing
    BlockEntry(BlockId),
    BeforeInstruction(BlockId, usize),
    Instruction(BlockId, usize),
    AllocateConstBeforeJump(BlockId),
    BlockExit(BlockId),
}

#[derive(Debug)]
pub struct Gape<'a> {
    /// invariant:
    /// 1. no unreachable block
    /// 2. no switch(constant)
    /// 3. no ConditionalJump(constant)
    pub blocks: BTreeMap<BlockId, ir::Block>,
    pub bid_init: BlockId,

    /// copy from environmen
    pub allocations: Vec<Named<Dtype>>,
    pub function_abi_mp: &'a HashMap<String, FunctionAbi>,
    pub source: &'a ir::TranslationUnit,

    pub pred_mp: Frozen<BTreeMap<BlockId, Vec<BlockId>>>,
    /// abi of current function
    pub abi: FunctionAbi,
    pub inst_mp: Frozen<BiMap<Yank, regalloc2::Inst>>,
    pub reg_mp: Frozen<BiMap<RegisterId, regalloc2::VReg>>,
    pub block_mp: Frozen<BiMap<BlockId, regalloc2::Block>>,
    /// only make sense in ConditionalJump/Switch
    pub constant_in_jumparg_mp: Frozen<BTreeMap<BlockId, Vec<Vec<regalloc2::VReg>>>>,
    /// Call/Noop: empty
    pub constant_in_instruction_mp: Frozen<BTreeMap<(BlockId, usize), Vec<regalloc2::VReg>>>,

    num_vregs: usize,
}

impl<'a> Gape<'a> {
    fn init_inst_mp(blocks: &BTreeMap<BlockId, ir::Block>) -> BiMap<Yank, regalloc2::Inst> {
        let mut inst_mp: BiMap<Yank, regalloc2::Inst> = BiMap::new();
        let mut insn = regalloc2::Inst::new(0);

        for (&bid, block) in blocks {
            inst_mp
                .insert_no_overwrite(Yank::BlockEntry(bid), insn)
                .unwrap();
            insn = insn.next();
            for offset in 0..block.instructions.len() {
                inst_mp
                    .insert_no_overwrite(Yank::BeforeInstruction(bid, offset), insn)
                    .unwrap();
                insn = insn.next();
                inst_mp
                    .insert_no_overwrite(Yank::Instruction(bid, offset), insn)
                    .unwrap();
                insn = insn.next();
            }
            inst_mp
                .insert_no_overwrite(Yank::AllocateConstBeforeJump(bid), insn)
                .unwrap();
            insn = insn.next();

            inst_mp
                .insert_no_overwrite(Yank::BlockExit(bid), insn)
                .unwrap();
            insn = insn.next();
        }
        inst_mp
    }

    fn init_reg_mp(blocks: &BTreeMap<BlockId, ir::Block>) -> BiMap<RegisterId, regalloc2::VReg> {
        let mut reg_mp: BiMap<RegisterId, regalloc2::VReg> = BiMap::new();

        let mut index = 0usize;

        // we must allocate parameter register
        let arg_iter = blocks.iter().flat_map(|(&bid, bb)| {
            bb.phinodes
                .iter()
                .enumerate()
                .map(move |(aid, dtype)| (RegisterId::Arg { bid, aid }, dtype.deref().clone()))
        });

        let temp_iter = blocks.iter().flat_map(|(bid, block)| {
            block
                .instructions
                .iter()
                .enumerate()
                .filter_map(|(iid, instr)| {
                    let dtype = instr.dtype();
                    match &dtype {
                        Dtype::Unit { .. } => None,
                        Dtype::Int { .. } | Dtype::Float { .. } | Dtype::Pointer { .. } => {
                            Some((RegisterId::Temp { bid: *bid, iid }, dtype))
                        }
                        Dtype::Struct { .. } => None,
                        Dtype::Array { .. } | Dtype::Function { .. } | Dtype::Typedef { .. } => {
                            unreachable!()
                        }
                    }
                })
        });

        for (rid, dtype) in arg_iter.chain(temp_iter) {
            match dtype {
                // struct : block paramter may be indirect register
                Dtype::Int { .. } | Dtype::Pointer { .. } | Dtype::Struct { .. } => {
                    match reg_mp.insert_no_overwrite(
                        rid,
                        regalloc2::VReg::new(index, regalloc2::RegClass::Int),
                    ) {
                        Ok(()) => index += 1,
                        Err(_) => {}
                    }
                }
                Dtype::Float { .. } => {
                    match reg_mp.insert_no_overwrite(
                        rid,
                        regalloc2::VReg::new(index, regalloc2::RegClass::Float),
                    ) {
                        Ok(()) => index += 1,
                        Err(_) => {}
                    }
                }
                Dtype::Array { .. } | Dtype::Unit { .. } => {}
                Dtype::Function { .. } => unreachable!(),
                Dtype::Typedef { .. } => unreachable!(),
            }
        }

        reg_mp
    }

    fn init_block_mp(blocks: &BTreeMap<BlockId, ir::Block>) -> BiMap<BlockId, regalloc2::Block> {
        let mut block_mp: BiMap<BlockId, regalloc2::Block> = BiMap::new();
        let mut block = regalloc2::Block::new(0);

        for bid in blocks.keys() {
            block_mp.insert_no_overwrite(*bid, block).unwrap();
            block = block.next()
        }
        block_mp
    }

    fn init_constant_in_jumparg(
        blocks: &BTreeMap<BlockId, ir::Block>,
        mut next_virt_reg: impl FnMut() -> usize,
    ) -> BTreeMap<BlockId, Vec<Vec<regalloc2::VReg>>> {
        blocks
            .iter()
            .map(|(bid, bb)| {
                let vv = bb
                    .exit
                    .walk_jump_args_1()
                    .map(|jump_arg| {
                        jump_arg
                            .walk_constant_arg()
                            .map(|c| match c {
                                Constant::Unit | Constant::Undef { .. } => unreachable!(),
                                Constant::Int { .. } => {
                                    let x = regalloc2::VReg::new(
                                        next_virt_reg(),
                                        regalloc2::RegClass::Int,
                                    );
                                    x
                                }
                                Constant::Float { .. } => {
                                    let x = regalloc2::VReg::new(
                                        next_virt_reg(),
                                        regalloc2::RegClass::Float,
                                    );
                                    x
                                }
                                Constant::GlobalVariable { .. } => unreachable!(),
                            })
                            .collect_vec()
                    })
                    .collect();
                (*bid, vv)
            })
            .collect()
    }

    fn init_constant_in_instruction(
        blocks: &BTreeMap<BlockId, ir::Block>,
        mut next_virt_reg: impl FnMut() -> usize,
    ) -> BTreeMap<(BlockId, usize), Vec<regalloc2::VReg>> {
        let mut res = BTreeMap::new();
        for (bid, block) in blocks {
            for (iid, instruction) in block.instructions.iter().enumerate() {
                match &**instruction {
                    ir::Instruction::Nop | ir::Instruction::Call { .. } => {}
                    ir::Instruction::BinOp { .. }
                    | ir::Instruction::UnaryOp { .. }
                    | ir::Instruction::Store { .. }
                    | ir::Instruction::Load { .. }
                    | ir::Instruction::TypeCast { .. }
                    | ir::Instruction::GetElementPtr { .. } => {
                        let mut v = Vec::new();
                        for c in instruction.walk_constant() {
                            match c {
                                Constant::GlobalVariable { .. } => {
                                    // skip global variable
                                }
                                Constant::Unit | Constant::Undef { .. } => unreachable!(),
                                Constant::Int { .. } => {
                                    v.push(regalloc2::VReg::new(
                                        next_virt_reg(),
                                        regalloc2::RegClass::Int,
                                    ));
                                }
                                Constant::Float { .. } => {
                                    v.push(regalloc2::VReg::new(
                                        next_virt_reg(),
                                        regalloc2::RegClass::Float,
                                    ));
                                }
                            }
                        }
                        let None = res.insert((*bid, iid), v) else {
                            unreachable!()
                        };
                    }
                }
            }
        }
        res
    }

    pub fn from_definition(
        definition: &FunctionDefinition,
        abi: FunctionAbi,
        function_abi_mp: &'a HashMap<String, FunctionAbi>,
        source: &'a ir::TranslationUnit,
    ) -> Self {
        Self::new(
            definition.blocks.clone(),
            definition.bid_init,
            abi,
            function_abi_mp,
            source,
            definition.allocations.clone(),
        )
    }

    pub fn new(
        mut blocks: BTreeMap<BlockId, ir::Block>,
        bid_init: BlockId,
        abi: FunctionAbi,
        function_abi_mp: &'a HashMap<String, FunctionAbi>,
        source: &'a ir::TranslationUnit,
        allocations: Vec<Named<Dtype>>,
    ) -> Self {
        for b in blocks.values_mut() {
            let _ = b.exit.optimize();
        }
        // remove unreachable blocks
        let _ = SimplifyCfgReach::optimize_inner(bid_init, &mut blocks);

        let reg_mp = Frozen::freeze(Self::init_reg_mp(&blocks));
        let mut a = reg_mp.len();
        let f = || -> usize {
            let b = a;
            a += 1;
            b
        };

        Self {
            bid_init,
            constant_in_jumparg_mp: Frozen::freeze(Self::init_constant_in_jumparg(&blocks, f)),
            inst_mp: Frozen::freeze(Self::init_inst_mp(&blocks)),
            block_mp: Frozen::freeze(Self::init_block_mp(&blocks)),
            pred_mp: Frozen::freeze(
                calculate_pred_inner(bid_init, &blocks)
                    .into_iter()
                    .map(|(x, y)| (x, y.into_iter().collect()))
                    .collect(),
            ),
            constant_in_instruction_mp: Frozen::freeze(Self::init_constant_in_instruction(
                &blocks,
                || -> usize {
                    let b = a;
                    a += 1;
                    b
                },
            )),
            blocks,
            reg_mp,
            abi,
            function_abi_mp,
            source,
            allocations,
            num_vregs: a,
        }
    }
}

impl<'a> regalloc2::Function for Gape<'a> {
    fn num_insts(&self) -> usize {
        self.blocks
            .values()
            .map(|b| b.instructions.len() * 2 + 1 + 1 + 1) // + 1 : blockexit
            // +1 : BeforeFirst
            // +1 : AllocateConstBeforeJump
            .fold(0, |a, b| a + b)
    }

    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn entry_block(&self) -> regalloc2::Block {
        *self.block_mp.get_by_left(&self.bid_init).unwrap()
    }

    fn block_insns(&self, block: regalloc2::Block) -> regalloc2::InstRange {
        let block_id: BlockId = *self.block_mp.get_by_right(&block).unwrap();
        regalloc2::InstRange::new(
            *self
                .inst_mp
                .get_by_left(&Yank::BlockEntry(block_id))
                .unwrap(),
            self.inst_mp
                .get_by_left(&Yank::BlockExit(block_id))
                .unwrap()
                .next(),
        )
    }

    fn block_succs(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        let block_id: BlockId = *self.block_mp.get_by_right(&block).unwrap();
        let block_exit = &self.blocks[&block_id].exit;
        let v: Vec<regalloc2::Block> = block_exit
            .walk_jump_bid()
            .map(|x| *self.block_mp.get_by_left(&x).unwrap())
            .collect();
        v.leak()
    }

    fn block_preds(&self, block: regalloc2::Block) -> &[regalloc2::Block] {
        let bid: BlockId = *self.block_mp.get_by_right(&block).unwrap();
        self.pred_mp
            .get(&bid)
            .map(Deref::deref)
            .unwrap_or_default()
            .iter()
            .map(|k| *self.block_mp.get_by_left(k).unwrap())
            .collect::<Vec<_>>()
            .leak()
    }

    fn block_params(&self, block: regalloc2::Block) -> &[regalloc2::VReg] {
        if block == self.entry_block() {
            return &[];
        }
        let bid: BlockId = *self.block_mp.get_by_right(&block).unwrap();
        let block = &self.blocks[&bid];
        block
            .phinodes
            .iter()
            .enumerate()
            .map(|(aid, dtype)| {
                let rid = RegisterId::Arg { bid, aid };
                *self.reg_mp.get_by_left(&rid).unwrap()
            })
            .collect::<Vec<_>>()
            .leak()
    }

    fn is_ret(&self, insn: regalloc2::Inst) -> bool {
        let yank = self.inst_mp.get_by_right(&insn).unwrap();
        match yank {
            Yank::BlockEntry(_)
            | Yank::BeforeInstruction(_, _)
            | Yank::Instruction(_, _)
            | Yank::AllocateConstBeforeJump(_) => false,
            Yank::BlockExit(bid) => match self.blocks[&bid].exit {
                ir::BlockExit::Return { .. } => true,
                _ => false,
            },
        }
    }

    fn is_branch(&self, insn: regalloc2::Inst) -> bool {
        let yank = self.inst_mp.get_by_right(&insn).unwrap();
        match yank {
            Yank::BlockEntry(_)
            | Yank::BeforeInstruction(_, _)
            | Yank::Instruction(_, _)
            | Yank::AllocateConstBeforeJump(_) => false,
            Yank::BlockExit(bid) => match self.blocks[&bid].exit {
                ir::BlockExit::Jump { .. } => true,
                ir::BlockExit::ConditionalJump { .. } => true,
                ir::BlockExit::Switch { .. } => true,
                ir::BlockExit::Return { .. } => false,
                ir::BlockExit::Unreachable => false,
            },
        }
    }

    fn branch_blockparams(
        &self,
        block: regalloc2::Block,
        insn: regalloc2::Inst,
        succ_idx: usize,
    ) -> &[regalloc2::VReg] {
        let block_id_1: BlockId = *self.block_mp.get_by_right(&block).unwrap();
        let Yank::BlockExit(bid) = self.inst_mp.get_by_right(&insn).unwrap() else {
            unreachable!()
        };
        assert_eq!(block_id_1, *bid);
        let Some(jump_arg) = self.blocks[&bid].exit.walk_jump_args_1().nth(succ_idx) else {
            unreachable!()
        };
        let mut constant_mp = self.constant_in_jumparg_mp[&block_id_1][succ_idx].iter();
        jump_arg
            .args
            .iter()
            .map(|operand| match operand {
                Operand::Constant(_) => *constant_mp.next().unwrap(),
                Operand::Register { rid, dtype } => *self.reg_mp.get_by_left(&rid).unwrap(),
            })
            .collect::<Vec<_>>()
            .leak()
    }

    fn inst_operands(&self, insn: regalloc2::Inst) -> &[regalloc2::Operand] {
        let yank = *self.inst_mp.get_by_right(&insn).unwrap();

        match yank {
            Yank::BlockEntry(bid) => {
                let block = &self.blocks[&bid];
                if self.entry_block() == *self.block_mp.get_by_left(&bid).unwrap() {
                    block
                        .phinodes
                        .iter()
                        .zip(self.abi.params_alloc.iter())
                        .enumerate()
                        .flat_map(|(aid, (dtype, param_alloc))| {
                            let rid = RegisterId::Arg { bid, aid };
                            match param_alloc {
                                crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::Direct(RegOrStack::Reg(reg)),
                                )
                                | crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::InDirect(RegOrStack::Reg(reg)),
                                ) => self.reg_mp.get_by_left(&rid).map(|vreg| {
                                    regalloc2::Operand::new(
                                        *vreg,
                                        regalloc2::OperandConstraint::FixedReg((*reg).into()),
                                        regalloc2::OperandKind::Def,
                                        regalloc2::OperandPos::Late,
                                    )
                                }),
                                crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::Direct(RegOrStack::Stack { .. }),
                                )
                                | crate::asmgen::ParamAlloc::PrimitiveType(
                                    DirectOrInDirect::InDirect(RegOrStack::Stack { .. }),
                                ) => None,
                                crate::asmgen::ParamAlloc::StructInRegister(_) => None,
                            }
                        })
                        .collect::<Vec<_>>()
                        .leak()
                } else {
                    &[]
                }
            }
            Yank::BeforeInstruction(bid, iid) => {
                match self.constant_in_instruction_mp.get(&(bid, iid)) {
                    Some(v) => v
                        .iter()
                        .map(|vreg| {
                            regalloc2::Operand::new(
                                *vreg,
                                regalloc2::OperandConstraint::Reg,
                                regalloc2::OperandKind::Def,
                                regalloc2::OperandPos::Late,
                            )
                        })
                        .collect::<Vec<_>>()
                        .leak(),
                    None => {
                        // Call / Nop
                        &[]
                    }
                }
            }
            Yank::Instruction(bid, iid) => {
                // TODO: totally wrong: match instruction first
                let block = &self.blocks[&bid];
                let instruction: &ir::Instruction = &block.instructions[iid];
                let mut v: Vec<regalloc2::Operand> = Vec::new();

                // 1. add read
                // 2. add write

                match instruction {
                    ir::Instruction::Call {
                        callee,
                        args,
                        return_type,
                    } => {
                        let FunctionAbi { params_alloc, .. } =
                            clown(callee, self.function_abi_mp, self.source);
                        for (operand, alloc) in izip!(args, params_alloc) {
                            match (alloc, operand) {
                                (
                                    ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(
                                        RegOrStack::Reg(target_reg),
                                    )),
                                    ir::Operand::Constant(c),
                                ) => {}
                                (
                                    ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(
                                        RegOrStack::Reg(target_reg),
                                    )),
                                    ir::Operand::Register { rid, dtype },
                                ) => {
                                    v.push(regalloc2::Operand::new(
                                        *self.reg_mp.get_by_left(rid).unwrap(),
                                        regalloc2::OperandConstraint::FixedReg(target_reg.into()),
                                        regalloc2::OperandKind::Use,
                                        regalloc2::OperandPos::Early,
                                    ));
                                }
                                (
                                    ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(
                                        RegOrStack::Stack { offset_to_s0 },
                                    )),
                                    ir::Operand::Constant(..),
                                ) => {}
                                (
                                    ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(
                                        RegOrStack::Stack { .. },
                                    )),
                                    ir::Operand::Register { .. },
                                ) => {
                                    unreachable!("{}", operand.dtype())
                                }
                                (
                                    ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(
                                        RegOrStack::Reg(target_reg),
                                    )),
                                    ir::Operand::Register { rid, .. },
                                ) => match self.reg_mp.get_by_left(rid) {
                                    Some(vreg) => {
                                        v.push(regalloc2::Operand::new(
                                            *vreg,
                                            regalloc2::OperandConstraint::FixedReg(
                                                target_reg.into(),
                                            ),
                                            regalloc2::OperandKind::Use,
                                            regalloc2::OperandPos::Early,
                                        ));
                                    }
                                    None => {}
                                },
                                (
                                    ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(
                                        RegOrStack::Stack {
                                            offset_to_s0: target_offset,
                                        },
                                    )),
                                    ir::Operand::Register { rid, .. },
                                ) => match self.reg_mp.get_by_left(rid) {
                                    Some(vreg) => {
                                        v.push(regalloc2::Operand::new(
                                            *vreg,
                                            regalloc2::OperandConstraint::Reg,
                                            regalloc2::OperandKind::Use,
                                            regalloc2::OperandPos::Early,
                                        ));
                                    }
                                    None => todo!(),
                                },
                                (
                                    ParamAlloc::StructInRegister(_),
                                    ir::Operand::Register {
                                        rid,
                                        dtype: ir::Dtype::Struct { .. },
                                    },
                                ) => match self.reg_mp.get_by_left(rid) {
                                    Some(vreg) => {
                                        v.push(regalloc2::Operand::new(
                                            *vreg,
                                            regalloc2::OperandConstraint::Reg,
                                            regalloc2::OperandKind::Use,
                                            regalloc2::OperandPos::Early,
                                        ));
                                    }
                                    None => todo!(),
                                },
                                _ => unreachable!(),
                            }
                        }

                        // deal with operand last
                        match callee {
                            Operand::Constant(_) => {}
                            Operand::Register { rid, dtype } => {
                                v.push(regalloc2::Operand::new(
                                    *self.reg_mp.get_by_left(rid).unwrap(),
                                    regalloc2::OperandConstraint::Any,
                                    regalloc2::OperandKind::Use,
                                    regalloc2::OperandPos::Early,
                                ));
                            }
                        }
                    }
                    ir::Instruction::BinOp { .. }
                    | ir::Instruction::UnaryOp { .. }
                    | ir::Instruction::Store { .. }
                    | ir::Instruction::Load { .. }
                    | ir::Instruction::TypeCast { .. }
                    | ir::Instruction::GetElementPtr { .. } => {
                        let mut constant_iter = self.constant_in_instruction_mp[&(bid, iid)].iter();
                        for operand in instruction.walk_operand() {
                            match operand {
                                Operand::Constant(Constant::GlobalVariable { .. }) => {
                                    // skip GlobalVariable
                                }
                                Operand::Constant(_) => {
                                    let vreg = constant_iter.next().unwrap();
                                    v.push(regalloc2::Operand::new(
                                        *vreg,
                                        regalloc2::OperandConstraint::Any,
                                        regalloc2::OperandKind::Use,
                                        regalloc2::OperandPos::Early,
                                    ));
                                }
                                Operand::Register {
                                    rid: RegisterId::Local { .. },
                                    ..
                                } => {}
                                Operand::Register {
                                    rid,
                                    dtype:
                                        Dtype::Int { .. } | Dtype::Float { .. } | Dtype::Pointer { .. },
                                } => {
                                    v.push(regalloc2::Operand::new(
                                        *self.reg_mp.get_by_left(&rid).unwrap(),
                                        regalloc2::OperandConstraint::Any,
                                        regalloc2::OperandKind::Use,
                                        regalloc2::OperandPos::Early,
                                    ));
                                }
                                Operand::Register {
                                    rid: RegisterId::Temp { .. },
                                    dtype: Dtype::Struct { .. },
                                } => {
                                    // always on stack
                                }
                                Operand::Register {
                                    rid: rid @ RegisterId::Arg { bid, aid },
                                    dtype: Dtype::Struct { .. },
                                } => {
                                    if *bid == self.bid_init {
                                        // check in abi
                                        match self.abi.params_alloc[*aid] {
                                            ParamAlloc::PrimitiveType(
                                                DirectOrInDirect::Direct(_),
                                            ) => unreachable!(),
                                            ParamAlloc::PrimitiveType(
                                                DirectOrInDirect::InDirect(RegOrStack::Reg(reg)),
                                            ) => v.push(regalloc2::Operand::new(
                                                *self.reg_mp.get_by_left(rid).unwrap(),
                                                regalloc2::OperandConstraint::Any,
                                                regalloc2::OperandKind::Use,
                                                regalloc2::OperandPos::Early,
                                            )),
                                            ParamAlloc::PrimitiveType(
                                                DirectOrInDirect::InDirect(RegOrStack::Stack {
                                                    ..
                                                }),
                                            ) => {}
                                            ParamAlloc::StructInRegister(_) => {
                                                // always on stack
                                            }
                                        }
                                    } else {
                                        unreachable!()
                                    }
                                }
                                Operand::Register { .. } => {
                                    unreachable!()
                                }
                            }
                        }
                    }
                    ir::Instruction::Nop => {}
                }

                let rid = RegisterId::Temp { bid, iid };
                match instruction {
                    ir::Instruction::Call {
                        callee,
                        args,
                        return_type: Dtype::Int { .. } | Dtype::Pointer { .. },
                    } => match self.reg_mp.get_by_left(&rid) {
                        Some(dest) => v.push(regalloc2::Operand::new(
                            *dest,
                            regalloc2::OperandConstraint::FixedReg(Register::A0.into()),
                            regalloc2::OperandKind::Def,
                            regalloc2::OperandPos::Late,
                        )),
                        None => {
                            unreachable!("{rid}")
                        }
                    },
                    ir::Instruction::Call {
                        callee,
                        args,
                        return_type: Dtype::Float { .. },
                    } => match self.reg_mp.get_by_left(&rid) {
                        Some(dest) => v.push(regalloc2::Operand::new(
                            *dest,
                            regalloc2::OperandConstraint::FixedReg(Register::FA0.into()),
                            regalloc2::OperandKind::Def,
                            regalloc2::OperandPos::Late,
                        )),
                        None => {
                            unreachable!("{rid}")
                        }
                    },
                    ir::Instruction::Call {
                        return_type: Dtype::Struct { .. } | Dtype::Unit { .. },
                        ..
                    } => {}
                    ir::Instruction::Call { .. } => unreachable!("{instruction}"),
                    ir::Instruction::BinOp { .. }
                    | ir::Instruction::UnaryOp { .. }
                    | ir::Instruction::Store { .. }
                    | ir::Instruction::Load { .. }
                    | ir::Instruction::TypeCast { .. }
                    | ir::Instruction::GetElementPtr { .. } => match instruction.dtype() {
                        Dtype::Unit { .. } => {}
                        Dtype::Int { .. } | Dtype::Pointer { .. } | Dtype::Float { .. } => {
                            match self.reg_mp.get_by_left(&rid) {
                                Some(dest) => v.push(regalloc2::Operand::new(
                                    *dest,
                                    regalloc2::OperandConstraint::Any,
                                    regalloc2::OperandKind::Def,
                                    regalloc2::OperandPos::Late,
                                )),
                                None => {
                                    unreachable!("{rid}")
                                }
                            }
                        }

                        Dtype::Struct { .. } => {}
                        Dtype::Array { .. } => unreachable!(),
                        Dtype::Function { .. } => unreachable!(),
                        Dtype::Typedef { .. } => unreachable!(),
                    },
                    ir::Instruction::Nop => {}
                }

                v.leak()
            }
            Yank::AllocateConstBeforeJump(bid) => self.constant_in_jumparg_mp[&bid]
                .iter()
                .flatten()
                .map(|&vreg| {
                    regalloc2::Operand::new(
                        vreg,
                        regalloc2::OperandConstraint::Any,
                        regalloc2::OperandKind::Def,
                        regalloc2::OperandPos::Late,
                    )
                })
                .collect_vec()
                .leak(),
            Yank::BlockExit(bid) => {
                let mut v: Vec<_> = Vec::new();
                let block = &self.blocks[&bid];
                match &block.exit {
                    BlockExit::Jump { .. } => &[],
                    BlockExit::ConditionalJump {
                        condition:
                            Operand::Register {
                                rid: rid @ (RegisterId::Arg { .. } | RegisterId::Temp { .. }),
                                dtype,
                            },
                        ..
                    }
                    | BlockExit::Switch {
                        value:
                            Operand::Register {
                                rid: rid @ (RegisterId::Arg { .. } | RegisterId::Temp { .. }),
                                dtype,
                            },
                        ..
                    } => {
                        let x = regalloc2::Operand::new(
                            *self.reg_mp.get_by_left(&rid).unwrap(),
                            regalloc2::OperandConstraint::Any,
                            regalloc2::OperandKind::Use,
                            regalloc2::OperandPos::Early,
                        );
                        v.push(x);
                        v.leak()
                    }
                    BlockExit::Return {
                        value:
                            Operand::Register {
                                rid,
                                dtype: Dtype::Int { .. } | Dtype::Pointer { .. },
                            },
                    } => {
                        let x = regalloc2::Operand::new(
                            *self.reg_mp.get_by_left(&rid).unwrap(),
                            regalloc2::OperandConstraint::FixedReg(Register::A0.into()),
                            regalloc2::OperandKind::Use,
                            regalloc2::OperandPos::Early,
                        );
                        v.push(x);
                        v.leak()
                    }
                    BlockExit::Return {
                        value:
                            Operand::Register {
                                rid,
                                dtype: Dtype::Float { .. },
                            },
                    } => {
                        let x = regalloc2::Operand::new(
                            *self.reg_mp.get_by_left(&rid).unwrap(),
                            regalloc2::OperandConstraint::FixedReg(Register::FA0.into()),
                            regalloc2::OperandKind::Use,
                            regalloc2::OperandPos::Early,
                        );
                        v.push(x);
                        v.leak()
                    }
                    BlockExit::Return {
                        value:
                            Operand::Register {
                                dtype: Dtype::Struct { .. },
                                ..
                            },
                    }
                    | BlockExit::Return {
                        value: Operand::Constant(..),
                    } => &[],
                    BlockExit::Unreachable => unreachable!(),
                    _ => &[],
                }
            }
        }
    }

    fn inst_clobbers(&self, insn: regalloc2::Inst) -> regalloc2::PRegSet {
        let yank = *self.inst_mp.get_by_right(&insn).unwrap();
        // let block = &self.blocks[&block_id.into()];
        match yank {
            Yank::Instruction(bid, offset) => {
                let instr = self.blocks[&bid.into()].instructions[offset].deref();
                match instr {
                    ir::Instruction::TypeCast { .. }
                    | ir::Instruction::GetElementPtr { .. }
                    | ir::Instruction::Store { .. }
                    | ir::Instruction::UnaryOp { .. }
                    | ir::Instruction::BinOp { .. }
                    | ir::Instruction::Nop
                    | ir::Instruction::Load { .. } => regalloc2::PRegSet::empty(),
                    ir::Instruction::Call { .. } => {
                        let mut x = whole_pregset();
                        match instr.dtype() {
                            Dtype::Int { .. } | Dtype::Pointer { .. } => {
                                x.remove(Register::A0.into());
                                x
                            }
                            Dtype::Float { .. } => {
                                x.remove(Register::FA0.into());
                                x
                            }
                            _ => x,
                        }
                    }
                }
            }
            Yank::BlockEntry(_)
            | Yank::BeforeInstruction(_, _)
            | Yank::AllocateConstBeforeJump(_)
            | Yank::BlockExit(_) => regalloc2::PRegSet::empty(),
        }
    }

    fn num_vregs(&self) -> usize {
        self.num_vregs
    }

    fn spillslot_size(&self, regclass: regalloc2::RegClass) -> usize {
        match regclass {
            regalloc2::RegClass::Int => 1,
            regalloc2::RegClass::Float => 1,
            regalloc2::RegClass::Vector => unreachable!(),
        }
    }
}

impl BlockExit {
    pub fn walk_jump_args_1(&self) -> Box<dyn Iterator<Item = &JumpArg> + '_> {
        match self {
            Self::Jump { arg } => Box::new(once(arg)),
            Self::ConditionalJump {
                arg_then, arg_else, ..
            } => Box::new(once(arg_then).chain(once(arg_else))),
            Self::Switch { default, cases, .. } => {
                Box::new(once(default).chain(cases.iter().map(|x| &x.1)))
            }
            Self::Return { .. } | Self::Unreachable => Box::new(empty()),
        }
    }

    pub fn walk_jump_args_mut(&mut self) -> Box<dyn Iterator<Item = &mut JumpArg> + '_> {
        match self {
            Self::Jump { arg } => Box::new(once(arg)),
            Self::ConditionalJump {
                arg_then, arg_else, ..
            } => Box::new(once(arg_then).chain(once(arg_else))),
            Self::Switch { default, cases, .. } => {
                Box::new(once(default).chain(cases.iter_mut().map(|x| &mut x.1)))
            }
            Self::Return { .. } | Self::Unreachable => Box::new(empty()),
        }
    }
}

impl JumpArg {
    pub fn walk_constant_arg(&self) -> Box<dyn Iterator<Item = &Constant> + '_> {
        Box::new(self.args.iter().flat_map(|operand| match operand {
            Operand::Constant(c) => Some(c),
            Operand::Register { .. } => None,
        }))
    }
}

impl<'a> Gape<'a> {
    fn get_dtype(&self, register_id: RegisterId) -> Dtype {
        match register_id {
            RegisterId::Local { .. } => unreachable!(),
            RegisterId::Arg { bid, aid } => {
                let b = &self.blocks[&bid];
                b.phinodes[aid].deref().clone()
            }
            RegisterId::Temp { bid, iid } => {
                let b = &self.blocks[&bid];
                b.instructions[iid].dtype()
            }
        }
    }

    pub fn edit_2_instruction(
        &self,
        edit: &regalloc2::Edit,
        slate: &mut BiMap<RegisterId, regalloc2::Allocation>,
    ) -> Vec<asm::Instruction> {
        match edit {
            regalloc2::Edit::Move { from, to } => {
                if from == to {
                    return Vec::new();
                }
                match (from.kind(), to.kind()) {
                    (regalloc2::AllocationKind::None, regalloc2::AllocationKind::None) => todo!(),
                    (regalloc2::AllocationKind::None, regalloc2::AllocationKind::Reg) => todo!(),
                    (regalloc2::AllocationKind::None, regalloc2::AllocationKind::Stack) => todo!(),
                    (regalloc2::AllocationKind::Reg, regalloc2::AllocationKind::None) => todo!(),
                    (regalloc2::AllocationKind::Reg, regalloc2::AllocationKind::Reg) => {
                        let rs = from.as_reg().unwrap().into();
                        let rd = to.as_reg().unwrap().into();

                        match rs {
                            Register::Zero
                            | Register::Ra
                            | Register::Sp
                            | Register::Gp
                            | Register::Tp => {
                                unreachable!()
                            }
                            Register::Temp(RegisterType::Integer, _)
                            | Register::Saved(RegisterType::Integer, _)
                            | Register::Arg(RegisterType::Integer, _) => {
                                vec![asm::Instruction::Pseudo(Pseudo::Mv { rd, rs })]
                            }
                            Register::Temp(RegisterType::FloatingPoint, _)
                            | Register::Saved(RegisterType::FloatingPoint, _)
                            | Register::Arg(RegisterType::FloatingPoint, _) => {
                                let register_id = *slate.get_by_right(from).unwrap();
                                let dtype = self.get_dtype(register_id);
                                match &dtype {
                                    Dtype::Float { .. } => {
                                        let _ = slate.insert(register_id, *to);
                                        return vec![asm::Instruction::Pseudo(Pseudo::Fmv {
                                            rd,
                                            rs,
                                            data_size: DataSize::try_from(dtype).unwrap(),
                                        })];
                                    }
                                    _ => unreachable!(),
                                }
                            }
                        }
                    }
                    (regalloc2::AllocationKind::Reg, regalloc2::AllocationKind::Stack) => todo!(),
                    (regalloc2::AllocationKind::Stack, regalloc2::AllocationKind::None) => todo!(),
                    (regalloc2::AllocationKind::Stack, regalloc2::AllocationKind::Reg) => todo!(),
                    (regalloc2::AllocationKind::Stack, regalloc2::AllocationKind::Stack) => todo!(),
                }
            }
        }
    }
}

pub fn constant_2_allocation(
    c: Constant,
    allocation: regalloc2::Allocation,
    res: &mut Vec<asm::Instruction>,
    float_mp: &mut FloatMp,
) -> Register {
    match allocation.kind() {
        regalloc2::AllocationKind::None => unreachable!(),
        regalloc2::AllocationKind::Reg => {
            let reg: Register = allocation.as_reg().unwrap().into();
            match c {
                Constant::Undef { .. } | Constant::Unit => unreachable!(),
                Constant::Float { .. } => {
                    res.extend(load_float_to_reg(c, reg, float_mp));
                    reg
                }
                Constant::Int { .. } => {
                    res.extend(load_int_to_reg(c, reg));
                    reg
                }
                Constant::GlobalVariable { .. } => unreachable!(),
            }
        }
        regalloc2::AllocationKind::Stack => unimplemented!(),
    }
}

pub fn allocation_2_reg(allocation: regalloc2::Allocation, or_register: Register) -> Register {
    match allocation.kind() {
        regalloc2::AllocationKind::None => unreachable!(),
        regalloc2::AllocationKind::Reg => allocation.as_reg().unwrap().into(),
        regalloc2::AllocationKind::Stack => todo!(),
    }
}

fn whole_pregset() -> regalloc2::PRegSet {
    let mut res = regalloc2::PRegSet::empty();
    res.add(Register::T0.into());
    res.add(Register::T1.into());
    res.add(Register::T2.into());
    res.add(Register::T3.into());
    res.add(Register::T4.into());
    res.add(Register::T5.into());
    res.add(Register::T6.into());

    res.add(Register::FT0.into());
    res.add(Register::FT1.into());
    res.add(Register::FT2.into());
    res.add(Register::FT3.into());
    res.add(Register::FT4.into());
    res.add(Register::FT5.into());
    res.add(Register::FT6.into());
    res.add(Register::FT7.into());
    res.add(Register::FT8.into());
    res.add(Register::FT9.into());
    res.add(Register::FT10.into());
    res.add(Register::FT11.into());

    res.add(Register::A0.into());
    res.add(Register::A1.into());
    res.add(Register::A2.into());
    res.add(Register::A3.into());
    res.add(Register::A4.into());
    res.add(Register::A5.into());
    res.add(Register::A6.into());
    res.add(Register::A7.into());

    res.add(Register::FA0.into());
    res.add(Register::FA1.into());
    res.add(Register::FA2.into());
    res.add(Register::FA3.into());
    res.add(Register::FA4.into());
    res.add(Register::FA5.into());
    res.add(Register::FA6.into());
    res.add(Register::FA7.into());
    res
}

impl DomTree for Gape<'_> {
    fn init_block(&self) -> BlockId {
        self.bid_init
    }

    fn blocks(&self) -> &BTreeMap<BlockId, ir::Block> {
        &self.blocks
    }
}
