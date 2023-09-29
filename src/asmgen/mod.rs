#![allow(clippy::too_many_arguments)]

use crate::asm::{
    self, DataSize, Directive, IType, Immediate, Label, Pseudo, RType, Register, SType, Section,
    TranslationUnit,
};
use crate::ir::{self, BlockId, Declaration, FunctionSignature, HasDtype, RegisterId, Value};
use crate::Translate;
use itertools::izip;
use lang_c::ast::{BinaryOperator, Expression, Initializer, UnaryOperator};
use ordered_float::OrderedFloat;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::{HashMap, HashSet};

static INT_REGISTERS: [Register; 11] = [
    Register::S1,
    Register::S2,
    Register::S3,
    Register::S4,
    Register::S5,
    Register::S6,
    Register::S7,
    Register::S8,
    Register::S9,
    Register::S10,
    Register::S11,
];

static FLOAT_REGISTERS: [Register; 12] = [
    Register::FS0,
    Register::FS1,
    Register::FS2,
    Register::FS3,
    Register::FS4,
    Register::FS5,
    Register::FS6,
    Register::FS7,
    Register::FS8,
    Register::FS9,
    Register::FS10,
    Register::FS11,
];

#[derive(Default, Clone, Copy, Debug)]
pub struct Asmgen {}

impl Translate<ir::TranslationUnit> for Asmgen {
    type Target = asm::Asm;
    type Error = ();

    fn translate(&mut self, source: &ir::TranslationUnit) -> Result<Self::Target, Self::Error> {
        let mut asm = asm::Asm {
            unit: TranslationUnit {
                functions: vec![],
                variables: vec![],
            },
        };

        let mut float_mp = FloatMp(HashMap::new());

        // init global variable first
        for (label, decl) in &source.decls {
            let Declaration::Variable { dtype, initializer } = decl else { continue };
            let (_, align) = dtype.size_align_of(&source.structs).unwrap();

            let directives = initializer_2_directive(dtype.clone(), initializer.clone(), source);

            asm.unit.variables.push(Section {
                header: vec![
                    Directive::Section(asm::SectionType::Data),
                    Directive::Align(align),
                ],
                body: asm::Variable {
                    label: Label(label.clone()),
                    directives,
                },
            });
        }

        let function_abi_mp: HashMap<String, FunctionAbi> = source
            .decls
            .iter()
            .filter_map(|(label, decl)| match decl {
                Declaration::Variable { .. } => None,
                Declaration::Function { signature, .. } => {
                    Some((label.clone(), signature.try_alloc(source)))
                }
            })
            .collect();

        for (func_name, decl) in source.decls.iter() {
            let Declaration::Function{ signature, definition } = decl else { continue };
            asm.unit.functions.push(Section {
                header: vec![Directive::Globl(Label(func_name.to_owned()))],
                body: translate_function(
                    func_name,
                    signature,
                    definition.as_ref().unwrap(),
                    &function_abi_mp,
                    source,
                    &mut float_mp,
                ),
            });
        }

        for (f, index) in float_mp.0.into_iter() {
            let directive = f.to_directive();
            asm.unit.variables.push(Section {
                header: vec![],
                body: asm::Variable {
                    label: Label(format!(".LCPI1_{index}")),
                    directives: vec![directive],
                },
            });
        }

        Ok(asm)
    }
}

fn translate_function(
    func_name: &str,
    signature: &FunctionSignature,
    definition: &ir::FunctionDefinition,
    function_abi_mp: &HashMap<String, FunctionAbi>,
    source: &ir::TranslationUnit,
    float_mp: &mut FloatMp,
) -> asm::Function {
    let mut function: asm::Function = asm::Function { blocks: vec![] };

    let abi @ FunctionAbi {
        params_alloc: params,
        ..
    } = function_abi_mp.get(func_name).unwrap();

    let mut stack_offset_2_s0: i64 = 0;

    // ra
    stack_offset_2_s0 -= 8;

    // backup s0 s1 s2 ...
    // backup fs0 fs1 fs2 ...
    stack_offset_2_s0 -= 8 * 12 * 2;

    let mut register_mp: HashMap<RegisterId, DirectOrInDirect<RegOrStack>> = HashMap::new();

    let mut alloc_arg = vec![];

    for (aid, (alloc, dtype)) in izip!(params, &signature.params).enumerate() {
        let register_id = RegisterId::Arg {
            bid: definition.bid_init,
            aid,
        };
        match alloc {
            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(_))) => {
                match dtype {
                    ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => {
                        let None = register_mp.insert(register_id,  DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)) else {unreachable!()};
                    }
                    ir::Dtype::Float { .. } => {
                        let None = register_mp.insert(register_id,  DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure)) else {unreachable!()};
                    }
                    _ => unreachable!(),
                }
            }
            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Stack {
                offset_to_s0,
            })) => {
                let None = register_mp.insert(register_id, DirectOrInDirect::Direct( RegOrStack::Stack { offset_to_s0: *offset_to_s0 })) else {unreachable!()};
            }
            ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Reg(_))) => {
                match dtype {
                    ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => {
                        let None = register_mp.insert(register_id,  DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure )) else {unreachable!()};
                    }
                    ir::Dtype::Float { .. } => {
                        let None = register_mp.insert(register_id,  DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure )) else {unreachable!()};
                    }
                    _ => unreachable!(),
                }
            }
            ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Stack {
                offset_to_s0,
            })) => {
                let None = register_mp.insert(register_id, DirectOrInDirect::InDirect(
                    RegOrStack::Stack { offset_to_s0: *offset_to_s0 })) else {unreachable!()};
            }
            ParamAlloc::StructInRegister(v) => {
                let ir::Dtype::Struct { name, size_align_offsets , fields, ..} = dtype else {unreachable!()};
                let Some((size, align, offsets)) = (if size_align_offsets.is_some() {
                    size_align_offsets.clone()
                } else {
                    source
                        .structs
                        .get(name.as_ref().unwrap())
                        .and_then(|x| x.as_ref())
                        .and_then(|x| x.get_struct_size_align_offsets())
                        .and_then(|x| x.as_ref()).cloned()
                } ) else {unreachable!()};

                let Some(fields) = (if fields.is_some() {
                    fields.clone()
                } else {
                    source
                        .structs
                        .get(name.as_ref().unwrap())
                        .and_then(|x| x.as_ref())
                        .and_then(|x| x.get_struct_fields())
                        .and_then(|x| x.as_ref()).cloned()
                } ) else {unreachable!()};

                let align: i64 = align.max(4).try_into().unwrap();
                while stack_offset_2_s0 % align != 0 {
                    stack_offset_2_s0 -= 1;
                }
                stack_offset_2_s0 -= <usize as TryInto<i64>>::try_into(size.max(4)).unwrap();

                for (x, dtype, offset) in izip!(v, fields, offsets) {
                    match x {
                        RegisterCouple::Single(register) => {
                            alloc_arg.extend(mk_stype(
                                SType::store(dtype.clone().into_inner()),
                                Register::S0,
                                *register,
                                (stack_offset_2_s0
                                    + <usize as TryInto<i64>>::try_into(offset).unwrap())
                                    as u64,
                            ));
                        }
                        RegisterCouple::Double(register) => {
                            alloc_arg.extend(mk_stype(
                                SType::SD,
                                Register::S0,
                                *register,
                                (stack_offset_2_s0
                                    + <usize as TryInto<i64>>::try_into(offset).unwrap())
                                    as u64,
                            ));
                        }
                        RegisterCouple::MergedToPrevious => {}
                    }
                }
                let None = register_mp.insert(register_id, DirectOrInDirect::Direct( RegOrStack::Stack { offset_to_s0: stack_offset_2_s0 })) else {unreachable!()};
            }
            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::IntRegNotSure))
            | ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure))
            | ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure))
            | ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure)) => {
                unreachable!()
            }
        }
    }

    let mut init_allocation = vec![];
    for (aid, dtype) in definition.allocations.iter().enumerate() {
        let (size, align) = dtype.size_align_of(&source.structs).unwrap();
        let align: i64 = align.max(4).try_into().unwrap();
        while stack_offset_2_s0 % align != 0 {
            stack_offset_2_s0 -= 1;
        }
        stack_offset_2_s0 -= <usize as TryInto<i64>>::try_into(size.max(4)).unwrap();
        init_allocation.extend(mk_itype(
            IType::Addi(DataSize::Double),
            Register::T0,
            Register::S0,
            stack_offset_2_s0 as u64,
        ));
        while stack_offset_2_s0 % 8 != 0 {
            stack_offset_2_s0 -= 1;
        }
        stack_offset_2_s0 -= 8;
        init_allocation.extend(mk_stype(
            SType::SD,
            Register::S0,
            Register::T0,
            stack_offset_2_s0 as u64,
        ));
        let None = register_mp.insert(RegisterId::Local { aid }, DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 : stack_offset_2_s0 } )) else {unreachable!()};
    }

    for (&bid, block) in definition
        .blocks
        .iter()
        .filter(|(&bid, _)| bid != definition.bid_init)
    {
        for (aid, dtype) in block.phinodes.iter().enumerate() {
            match &**dtype {
                ir::Dtype::Unit { .. } => unreachable!(),
                ir::Dtype::Pointer { .. } | ir::Dtype::Int { .. } => {
                    let None = register_mp.insert(RegisterId::Arg { bid, aid }, DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)) else {unreachable!()};
                }
                ir::Dtype::Float { .. } => {
                    let None = register_mp.insert(RegisterId::Arg { bid, aid }, DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure)) else {unreachable!()};
                }
                ir::Dtype::Array { .. } => unreachable!(),
                ir::Dtype::Struct { .. } => {
                    // if necessary, alloc on stack
                    unreachable!()
                }
                ir::Dtype::Function { .. } => unreachable!(),
                ir::Dtype::Typedef { .. } => unreachable!(),
            }
        }
    }

    for (&bid, block) in definition.blocks.iter() {
        for (iid, instr) in block.instructions.iter().enumerate() {
            let dtype = instr.dtype();
            match &dtype {
                ir::Dtype::Unit { .. } => {}
                ir::Dtype::Pointer { .. } | ir::Dtype::Int { .. } => {
                    let None = register_mp.insert(RegisterId::Temp { bid , iid },DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)) else {unreachable!()};
                }
                ir::Dtype::Float { .. } => {
                    let None = register_mp.insert(RegisterId::Temp { bid , iid },DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure)) else {unreachable!()};
                }
                ir::Dtype::Array { .. } => unreachable!(),
                ir::Dtype::Struct { .. } => {
                    let (size, align) = dtype.size_align_of(&source.structs).unwrap();
                    let align: i64 = align.max(4).try_into().unwrap();
                    while stack_offset_2_s0 % align != 0 {
                        stack_offset_2_s0 -= 1;
                    }
                    stack_offset_2_s0 -= size.max(4) as i64;
                    let None = register_mp.insert(RegisterId::Temp { bid , iid },DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0: stack_offset_2_s0 } )) else {unreachable!()};
                }
                ir::Dtype::Function { .. } => unreachable!(),
                ir::Dtype::Typedef { .. } => unreachable!(),
            }
        }
    }

    // before gen detailed asm::Instruction
    // we need to allocate register first
    alloc_register(definition, &mut register_mp, &mut stack_offset_2_s0);

    #[allow(clippy::all)]
    for (_, v) in &register_mp {
        match v {
            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure)
            | DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure)
            | DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure) => unreachable!(),
            _ => {}
        }
    }

    let mut register_remap: Vec<(Register, Register, ir::Dtype)> = vec![];
    for (aid, (alloc, dtype)) in izip!(params, &signature.params).enumerate() {
        let register_id = RegisterId::Arg {
            bid: definition.bid_init,
            aid,
        };
        match (alloc, register_mp.get(&register_id).unwrap()) {
            (
                ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(reg))),
                DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
            ) => {
                alloc_arg.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    *reg,
                    *offset_to_s0 as u64,
                ));
            }
            (
                ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(_))),
                DirectOrInDirect::InDirect(RegOrStack::Stack { .. }),
            ) => {
                unreachable!()
            }
            (
                ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Reg(reg))),
                DirectOrInDirect::InDirect(RegOrStack::Stack { offset_to_s0 }),
            ) => {
                alloc_arg.extend(mk_stype(
                    SType::SD,
                    Register::S0,
                    *reg,
                    *offset_to_s0 as u64,
                ));
            }
            (
                // register to stack
                ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Reg(_))),
                DirectOrInDirect::Direct(RegOrStack::Stack { .. }),
            ) => {
                unreachable!()
            }

            (
                ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(origin_reg))),
                DirectOrInDirect::Direct(RegOrStack::Reg(target_reg)),
            )
            | (
                ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Reg(origin_reg))),
                DirectOrInDirect::InDirect(RegOrStack::Reg(target_reg)),
            ) => {
                if target_reg != origin_reg {
                    register_remap.push((*origin_reg, *target_reg, dtype.clone()));
                }
            }
            _ => {}
        }
    }

    alloc_arg.extend(cp_parallel(register_remap));

    // the stack pointer is always kept 16-byte aligned
    while stack_offset_2_s0 % 16 != 0 {
        stack_offset_2_s0 -= 1;
    }
    let stack_offset_2_s0 = stack_offset_2_s0;

    let backup_ra: Vec<crate::asm::Instruction> = mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::Ra,
        (-stack_offset_2_s0 - 8) as u64,
    );
    let restore_ra: Vec<crate::asm::Instruction> = mk_itype(
        asm::IType::LD,
        Register::Ra,
        Register::Sp,
        (-stack_offset_2_s0 - 8) as u64,
    );

    let mut backup_sx: Vec<crate::asm::Instruction> = vec![];
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S0,
        (-stack_offset_2_s0 - 16) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S1,
        (-stack_offset_2_s0 - 24) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S2,
        (-stack_offset_2_s0 - 32) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S3,
        (-stack_offset_2_s0 - 40) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S4,
        (-stack_offset_2_s0 - 48) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S5,
        (-stack_offset_2_s0 - 56) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S6,
        (-stack_offset_2_s0 - 64) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S7,
        (-stack_offset_2_s0 - 72) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S8,
        (-stack_offset_2_s0 - 80) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S9,
        (-stack_offset_2_s0 - 88) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S10,
        (-stack_offset_2_s0 - 96) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S11,
        (-stack_offset_2_s0 - 104) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS0,
        (-stack_offset_2_s0 - 112) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS1,
        (-stack_offset_2_s0 - 120) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS2,
        (-stack_offset_2_s0 - 128) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS3,
        (-stack_offset_2_s0 - 136) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS4,
        (-stack_offset_2_s0 - 144) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS5,
        (-stack_offset_2_s0 - 152) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS6,
        (-stack_offset_2_s0 - 160) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS7,
        (-stack_offset_2_s0 - 168) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS8,
        (-stack_offset_2_s0 - 176) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS9,
        (-stack_offset_2_s0 - 184) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS10,
        (-stack_offset_2_s0 - 192) as u64,
    ));
    backup_sx.extend(mk_stype(
        asm::SType::Store(DataSize::DoublePrecision),
        Register::Sp,
        Register::FS11,
        (-stack_offset_2_s0 - 200) as u64,
    ));

    let mut restore_sx: Vec<crate::asm::Instruction> = vec![];
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S0,
        Register::Sp,
        (-stack_offset_2_s0 - 16) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S1,
        Register::Sp,
        (-stack_offset_2_s0 - 24) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S2,
        Register::Sp,
        (-stack_offset_2_s0 - 32) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S3,
        Register::Sp,
        (-stack_offset_2_s0 - 40) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S4,
        Register::Sp,
        (-stack_offset_2_s0 - 48) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S5,
        Register::Sp,
        (-stack_offset_2_s0 - 56) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S6,
        Register::Sp,
        (-stack_offset_2_s0 - 64) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S7,
        Register::Sp,
        (-stack_offset_2_s0 - 72) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S8,
        Register::Sp,
        (-stack_offset_2_s0 - 80) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S9,
        Register::Sp,
        (-stack_offset_2_s0 - 88) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S10,
        Register::Sp,
        (-stack_offset_2_s0 - 96) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::LD,
        Register::S11,
        Register::Sp,
        (-stack_offset_2_s0 - 104) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS0,
        Register::Sp,
        (-stack_offset_2_s0 - 112) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS1,
        Register::Sp,
        (-stack_offset_2_s0 - 120) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS2,
        Register::Sp,
        (-stack_offset_2_s0 - 128) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS3,
        Register::Sp,
        (-stack_offset_2_s0 - 136) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS4,
        Register::Sp,
        (-stack_offset_2_s0 - 144) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS5,
        Register::Sp,
        (-stack_offset_2_s0 - 152) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS6,
        Register::Sp,
        (-stack_offset_2_s0 - 160) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS7,
        Register::Sp,
        (-stack_offset_2_s0 - 168) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS8,
        Register::Sp,
        (-stack_offset_2_s0 - 176) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS9,
        Register::Sp,
        (-stack_offset_2_s0 - 184) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS10,
        Register::Sp,
        (-stack_offset_2_s0 - 192) as u64,
    ));
    restore_sx.extend(mk_itype(
        asm::IType::Load {
            data_size: DataSize::DoublePrecision,
            is_signed: true,
        },
        Register::FS11,
        Register::Sp,
        (-stack_offset_2_s0 - 200) as u64,
    ));

    let mut backup_ra_and_init_sp = mk_itype(
        asm::IType::Addi(DataSize::Double),
        Register::Sp,
        Register::Sp,
        stack_offset_2_s0 as u64,
    );
    backup_ra_and_init_sp.extend(backup_ra);
    backup_ra_and_init_sp.extend(backup_sx);
    backup_ra_and_init_sp.extend(mk_itype(
        asm::IType::Addi(DataSize::Double),
        Register::S0,
        Register::Sp,
        (-stack_offset_2_s0) as u64,
    ));
    backup_ra_and_init_sp.extend(alloc_arg);
    backup_ra_and_init_sp.extend(init_allocation);

    let mut before_ret_instructions = restore_ra;
    before_ret_instructions.extend(restore_sx);
    before_ret_instructions.extend(mk_itype(
        asm::IType::Addi(DataSize::Double),
        Register::Sp,
        Register::Sp,
        (-stack_offset_2_s0) as u64,
    ));
    before_ret_instructions.push(asm::Instruction::Pseudo(Pseudo::Ret));

    let mut temp_block: Vec<asm::Block> = vec![];

    for (&bid, block) in definition.blocks.iter() {
        let instructions = translate_block(
            func_name,
            bid,
            block,
            &mut temp_block,
            &register_mp,
            source,
            function_abi_mp,
            abi,
            before_ret_instructions.clone(),
            float_mp,
        );
        function.blocks.push(asm::Block {
            label: Some(Label::new(func_name, bid)),
            instructions,
        });
    }

    let init_block = function.blocks.get_mut(0).unwrap();
    assert_eq!(
        init_block.label,
        Some(Label::new(func_name, definition.bid_init))
    );
    backup_ra_and_init_sp.extend(init_block.instructions.clone());
    init_block.instructions = backup_ra_and_init_sp;
    init_block.label = Some(Label(func_name.to_owned()));

    for b in temp_block.into_iter() {
        function.blocks.push(b);
    }

    function
}

fn alloc_register(
    definition: &ir::FunctionDefinition,
    register_mp: &mut HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
    stack_offset_2_s0: &mut i64,
) {
    let mut int_ig = int_interference_graph(definition, register_mp);
    spills(&mut int_ig, &INT_REGISTERS, stack_offset_2_s0, register_mp);
    color(int_ig, &INT_REGISTERS, register_mp);

    let mut float_ig = float_interference_graph(definition, register_mp);
    spills(
        &mut float_ig,
        &FLOAT_REGISTERS,
        stack_offset_2_s0,
        register_mp,
    );

    color(float_ig, &FLOAT_REGISTERS, register_mp);
}

fn color(
    mut ig: GraphWrapper,
    colors: &[Register],
    register_mp: &mut HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
) {
    let Some(peo) = petgraph::algo::peo::peo(&ig.graph) else {unreachable!()};
    for node_index in peo.into_iter().rev() {
        let mut colors: HashSet<Register> = colors.iter().copied().collect();
        for neighbour in ig.graph.neighbors(node_index) {
            if let Some(Some(color)) = ig.graph.node_weight(neighbour) {
                let true = colors.remove(color) else {unreachable!()};
            }
        }
        let Some(x)  = ig.graph.node_weight_mut(node_index) else {unreachable!()};
        *x = Some(colors.into_iter().next().unwrap());
    }

    for node_index in ig.graph.node_identifiers() {
        let register_id = ig.node_index_2_register_id.get(&node_index).unwrap();
        let Some(Some(color))= ig.graph.node_weight(node_index) else {unreachable!()};
        match register_mp.get(register_id).unwrap() {
            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure) => {
                let Some(DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)) = register_mp.insert(
                    *register_id,
                    DirectOrInDirect::Direct(RegOrStack::Reg(*color)),
                ) else {unreachable!()};
            }
            DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => {
                let Some(DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure)) = register_mp.insert(
                    *register_id,
                    DirectOrInDirect::Direct(RegOrStack::Reg(*color)),
                ) else {unreachable!()};
            }
            DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure) => {
                let Some(DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure)) = register_mp.insert(
                    *register_id,
                    DirectOrInDirect::InDirect(RegOrStack::Reg(*color)),
                ) else {unreachable!()};
            }
            DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure) => {
                let Some(DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure)) = register_mp.insert(
                    *register_id,
                    DirectOrInDirect::InDirect(RegOrStack::Reg(*color)),
                ) else {unreachable!()};
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Default)]
struct GraphWrapper {
    graph: StableGraph<Option<Register>, (), petgraph::Undirected>,
    register_id_2_node_index: HashMap<RegisterId, NodeIndex>,
    node_index_2_register_id: HashMap<NodeIndex, RegisterId>,
}

impl GraphWrapper {
    fn add_node(&mut self, register_id: RegisterId) -> NodeIndex {
        match self.register_id_2_node_index.entry(register_id) {
            std::collections::hash_map::Entry::Occupied(_) => unreachable!(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let node = self.graph.add_node(None);
                let _ = v.insert(node);
                let None = self.node_index_2_register_id.insert(node, register_id) else {unreachable!()};
                node
            }
        }
    }

    fn add_edge(
        &mut self,
        a: RegisterId,
        b: RegisterId,
        register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
    ) {
        if a == b {
            return;
        }
        if matches!(a, RegisterId::Local { .. }) || matches!(b, RegisterId::Local { .. }) {
            return;
        }
        if matches!(
            register_mp.get(&a).unwrap(),
            DirectOrInDirect::Direct(RegOrStack::Stack { .. })
                | DirectOrInDirect::InDirect(RegOrStack::Stack { .. })
        ) || matches!(
            register_mp.get(&b).unwrap(),
            DirectOrInDirect::Direct(RegOrStack::Stack { .. })
                | DirectOrInDirect::InDirect(RegOrStack::Stack { .. })
        ) {
            return;
        }
        let a = *self
            .register_id_2_node_index
            .get(&a)
            .expect(&format!("can't find {a}"));
        let b = *self.register_id_2_node_index.get(&b).unwrap();
        let _ = self.graph.update_edge(a, b, ());
    }
}

fn int_inter_block_liveness_graph(
    definition: &ir::FunctionDefinition,
    _register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
) -> LivenessRes {
    // def can be aggressive
    let mut def: HashMap<BlockId, Vec<RegisterId>> = HashMap::new();
    for (&bid, block) in &definition.blocks {
        let mut v = Vec::new();
        for aid in 0..block.phinodes.len() {
            v.push(RegisterId::Arg { bid, aid });
        }
        for iid in 0..block.instructions.len() {
            v.push(RegisterId::Temp { bid, iid });
        }
        let None = def.insert(bid, v) else {unreachable!()};
    }

    let mut usee: HashMap<BlockId, Vec<RegisterId>> = HashMap::new();
    for (&curr_bid, block) in &definition.blocks {
        let v: Vec<RegisterId> = block
            .walk_int_register()
            .filter_map(|rid| match &rid {
                RegisterId::Local { .. } => None,
                RegisterId::Arg { bid, .. } | RegisterId::Temp { bid, .. } => {
                    if *bid == curr_bid {
                        None
                    } else {
                        Some(rid)
                    }
                }
            })
            .collect();
        let None = usee.insert(curr_bid, v) else {unreachable!()};
    }
    gen_kill(&def, &usee, definition)
}

fn float_inter_block_liveness_graph(
    definition: &ir::FunctionDefinition,
    _register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
) -> LivenessRes {
    // def can be aggressive
    let mut def: HashMap<BlockId, Vec<RegisterId>> = HashMap::new();
    for (&bid, block) in &definition.blocks {
        let mut v = Vec::new();
        for aid in 0..block.phinodes.len() {
            v.push(RegisterId::Arg { bid, aid });
        }
        for iid in 0..block.instructions.len() {
            v.push(RegisterId::Temp { bid, iid });
        }
        let None = def.insert(bid, v) else {unreachable!()};
    }

    let mut usee: HashMap<BlockId, Vec<RegisterId>> = HashMap::new();
    for (&curr_bid, block) in &definition.blocks {
        let v: Vec<RegisterId> = block
            .walk_float_register()
            .filter_map(|rid| match &rid {
                RegisterId::Local { .. } => None,
                RegisterId::Arg { bid, .. } | RegisterId::Temp { bid, .. } => {
                    if *bid == curr_bid {
                        None
                    } else {
                        Some(rid)
                    }
                }
            })
            .collect();
        let None = usee.insert(curr_bid, v) else {unreachable!()};
    }
    gen_kill(&def, &usee, definition)
}

fn int_interference_graph(
    definition: &ir::FunctionDefinition,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
) -> GraphWrapper {
    let f = |register_id: RegisterId| {
        // TODO: maybe unnecessary
        if matches!(register_id, RegisterId::Local { .. })
            || matches!(
                register_mp.get(&register_id).unwrap(),
                DirectOrInDirect::Direct(RegOrStack::Stack { .. })
                    | DirectOrInDirect::InDirect(RegOrStack::Stack { .. })
            )
        {
            None
        } else {
            Some(register_id)
        }
    };

    let mut int_ig: GraphWrapper = GraphWrapper::default();
    for (rid, v) in register_mp {
        match v {
            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
            | DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure) => {
                let _ = int_ig.add_node(*rid);
            }
            _ => {}
        }
    }

    let liveness_sets = int_inter_block_liveness_graph(definition, register_mp).out;

    for (bid, mut live_set) in liveness_sets {
        let block = definition.blocks.get(&bid).unwrap();

        // first analyze block.exit
        for rid in block.exit.walk_int_register().filter_map(f) {
            let _ = live_set.insert(rid);
        }
        for rid in block.exit.walk_int_register().filter_map(f) {
            for a in &live_set {
                int_ig.add_edge(*a, rid, register_mp);
            }
        }

        // then visit instruction in reverse order
        for (iid, instr) in block.instructions.iter().enumerate().rev() {
            let _ = live_set.remove(&RegisterId::Temp { bid, iid });

            match instr.dtype() {
                ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => {
                    for &a in &live_set {
                        int_ig.add_edge(a, RegisterId::Temp { bid, iid }, register_mp);
                    }
                }
                _ => {}
            }

            for rid_1 in instr.walk_int_register().filter_map(f) {
                let _ = live_set.insert(rid_1);
            }

            for rid_1 in instr.walk_int_register().filter_map(f) {
                for &a in &live_set {
                    int_ig.add_edge(a, rid_1, register_mp);
                }
            }

            // TODO: check Call T5 T6 ...
        }

        // TODO: not a good idea to define closure in loop
        let f = |(aid, dtype): (usize, &ir::Dtype)| match dtype {
            ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => Some(aid),
            _ => None,
        };

        for aid in block
            .phinodes
            .iter()
            .enumerate()
            .filter_map(|(aid, dtype)| f((aid, &**dtype)))
        {
            let _ = live_set.insert(RegisterId::Arg { bid, aid });
        }

        for aid in block
            .phinodes
            .iter()
            .enumerate()
            .filter_map(|(aid, dtype)| f((aid, &**dtype)))
        {
            let arg = RegisterId::Arg { bid, aid };
            for &rid in &live_set {
                int_ig.add_edge(rid, arg, register_mp);
            }
        }
    }

    int_ig
}

fn float_interference_graph(
    definition: &ir::FunctionDefinition,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
) -> GraphWrapper {
    let f = |register_id: RegisterId| {
        // TODO: maybe unnecessary
        if matches!(register_id, RegisterId::Local { .. })
            || matches!(
                register_mp.get(&register_id).unwrap(),
                DirectOrInDirect::Direct(RegOrStack::Stack { .. })
                    | DirectOrInDirect::InDirect(RegOrStack::Stack { .. })
            )
        {
            None
        } else {
            Some(register_id)
        }
    };

    let mut float_ig: GraphWrapper = GraphWrapper::default();
    for (rid, v) in register_mp {
        match v {
            DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure)
            | DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure) => {
                let _ = float_ig.add_node(*rid);
            }
            _ => {}
        }
    }

    let liveness_sets = float_inter_block_liveness_graph(definition, register_mp).out;

    for (bid, mut live_set) in liveness_sets {
        let block = definition.blocks.get(&bid).unwrap();

        // first analyze block.exit
        for rid in block.exit.walk_float_register().filter_map(f) {
            let _ = live_set.insert(rid);
        }
        for rid in block.exit.walk_float_register().filter_map(f) {
            for a in &live_set {
                float_ig.add_edge(*a, rid, register_mp);
            }
        }

        // then visit instruction in reverse order
        for (iid, instr) in block.instructions.iter().enumerate().rev() {
            let _ = live_set.remove(&RegisterId::Temp { bid, iid });

            match instr.dtype() {
                ir::Dtype::Float { .. } => {
                    for &a in &live_set {
                        float_ig.add_edge(a, RegisterId::Temp { bid, iid }, register_mp);
                    }
                }
                _ => {}
            }

            for rid_1 in instr.walk_float_register().filter_map(f) {
                let _ = live_set.insert(rid_1);
            }

            for rid_1 in instr.walk_float_register().filter_map(f) {
                for &a in &live_set {
                    float_ig.add_edge(a, rid_1, register_mp);
                }
            }

            // TODO: check Call T5 T6 ...
        }

        // TODO: not a good idea to define closure in loop
        let f = |(aid, dtype): (usize, &ir::Dtype)| match dtype {
            ir::Dtype::Float { .. } => Some(aid),
            _ => None,
        };

        for aid in block
            .phinodes
            .iter()
            .enumerate()
            .filter_map(|(aid, dtype)| f((aid, &**dtype)))
        {
            let _ = live_set.insert(RegisterId::Arg { bid, aid });
        }

        for aid in block
            .phinodes
            .iter()
            .enumerate()
            .filter_map(|(aid, dtype)| f((aid, &**dtype)))
        {
            let arg = RegisterId::Arg { bid, aid };
            for &rid in &live_set {
                float_ig.add_edge(rid, arg, register_mp);
            }
        }
    }

    float_ig
}

fn spills(
    ig: &mut GraphWrapper,
    colors: &[Register],
    stack_offset_2_s0: &mut i64,
    register_mp: &mut HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
) {
    // TODO: find all clique larger than threhold ?
    loop {
        let Some(max_cliques) = petgraph::algo::peo::max_cliques(&ig.graph) else {
            dbg!(&ig.node_index_2_register_id);
            println!("{:?}", petgraph::dot::Dot::with_config(&ig.graph, &[petgraph::dot::Config::EdgeNoLabel,petgraph::dot::Config::NodeIndexLabel]));
            panic!("not a chordal graph")
        };
        if max_cliques.is_empty() {
            return;
        }
        if max_cliques
            .iter()
            .all(|clique| clique.len() <= colors.len())
        {
            return;
        }
        let k = max_cliques[0].len() - colors.len();
        let mut hs: HashSet<NodeIndex> = HashSet::new();
        for clique in max_cliques {
            for node in clique.iter().take(k) {
                if hs.insert(*node) {
                    spill(
                        *ig.node_index_2_register_id.get(node).unwrap(),
                        stack_offset_2_s0,
                        register_mp,
                    );
                    let Some(None) = ig.graph.remove_node(*node) else {unreachable!()};
                }
            }
        }
    }
}

/// no dtype here
/// always allocate 8 bytes
fn spill(
    register_id: RegisterId,
    stack_offset_2_s0: &mut i64,
    register_mp: &mut HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
) {
    let align: i64 = 8;
    let size: i64 = 8;
    while *stack_offset_2_s0 % align != 0 {
        *stack_offset_2_s0 -= 1;
    }
    *stack_offset_2_s0 -= size;

    match register_mp.get(&register_id).unwrap() {
        DirectOrInDirect::Direct(RegOrStack::IntRegNotSure) => {
            let Some(DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)) = register_mp.insert(
                register_id,
                DirectOrInDirect::Direct(RegOrStack::Stack {
                    offset_to_s0: *stack_offset_2_s0,
                }),
            ) else {unreachable!()};
        }
        DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => {
            let Some(DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure)) = register_mp.insert(
                register_id,
                DirectOrInDirect::Direct(RegOrStack::Stack {
                    offset_to_s0: *stack_offset_2_s0,
                }),
            ) else {unreachable!()};
        }
        DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure) => {
            let Some(DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure)) = register_mp.insert(
                register_id,
                DirectOrInDirect::InDirect(RegOrStack::Stack {
                    offset_to_s0: *stack_offset_2_s0,
                }),
            ) else {unreachable!()};
        }
        DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure) => {
            let Some(DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure)) = register_mp.insert(
                register_id,
                DirectOrInDirect::InDirect(RegOrStack::Stack {
                    offset_to_s0: *stack_offset_2_s0,
                }),
            ) else {unreachable!()};
        }
        _ => unreachable!(),
    }
}

fn translate_block(
    func_name: &str,
    bid: BlockId,
    block: &ir::Block,
    temp_block: &mut Vec<asm::Block>,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
    source: &ir::TranslationUnit,
    function_abi_mp: &HashMap<String, FunctionAbi>,
    abi: &FunctionAbi,
    before_ret_instructions: Vec<asm::Instruction>,
    float_mp: &mut FloatMp,
) -> Vec<asm::Instruction> {
    let mut res = vec![];

    for (iid, instr) in block.instructions.iter().enumerate() {
        match &**instr {
            ir::Instruction::UnaryOp {
                op,
                operand: ir::Operand::Constant(c),
                dtype,
            } => {
                let v = crate::ir::interp::calculator::calculate_unary_operator_expression(
                    op,
                    Value::try_from(c.clone()).unwrap(),
                )
                .unwrap();
                let data_size = DataSize::try_from(dtype.clone()).unwrap();
                match v {
                    Value::Int { value, .. } => {
                        match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                            DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                                res.extend(mk_itype(
                                    IType::Addi(data_size),
                                    *dest_reg,
                                    Register::Zero,
                                    value as u64 & data_size.mask(),
                                ));
                            }
                            DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                                let data_size = DataSize::try_from(dtype.clone()).unwrap();
                                res.extend(mk_itype(
                                    IType::Addi(data_size),
                                    Register::T0,
                                    Register::Zero,
                                    value as u64 & data_size.mask(),
                                ));
                                res.extend(mk_stype(
                                    SType::store(dtype.clone()),
                                    Register::S0,
                                    Register::T0,
                                    *offset_to_s0 as u64,
                                ));
                            }
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => {
                                unreachable!()
                            }
                            DirectOrInDirect::InDirect(_) => unreachable!(),
                        }
                    }
                    Value::Float { value, width } => {
                        let label = float_mp.get_label(Float { value, width });
                        res.push(asm::Instruction::Pseudo(Pseudo::La {
                            rd: Register::T0,
                            symbol: label,
                        }));
                        match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                            DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                                res.push(asm::Instruction::IType {
                                    instr: IType::load(dtype.clone()),
                                    rd: *dest_reg,
                                    rs1: Register::T0,
                                    imm: Immediate::Value(0),
                                });
                            }
                            DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                                res.push(asm::Instruction::IType {
                                    instr: IType::load(dtype.clone()),
                                    rd: Register::FT0,
                                    rs1: Register::T0,
                                    imm: Immediate::Value(0),
                                });
                                res.extend(mk_stype(
                                    SType::store(dtype.clone()),
                                    Register::S0,
                                    Register::FT0,
                                    *offset_to_s0 as u64,
                                ));
                            }
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => {
                                unreachable!()
                            }
                            DirectOrInDirect::InDirect(_) => unreachable!(),
                        }
                    }
                    _ => unreachable!(),
                }
            }
            ir::Instruction::UnaryOp {
                op: UnaryOperator::Minus,
                operand: operand @ ir::Operand::Register { .. },
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                let reg = load_operand_to_reg(
                    operand.clone(),
                    Register::T0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: RType::sub(dtype.clone()),
                            rd: *dest_reg,
                            rs1: Register::Zero,
                            rs2: Some(reg),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: RType::sub(dtype.clone()),
                            rd: Register::T1,
                            rs1: Register::Zero,
                            rs2: Some(reg),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T1,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::UnaryOp {
                op: UnaryOperator::Minus,
                operand: operand @ ir::Operand::Register { .. },
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                let reg = load_operand_to_reg(
                    operand.clone(),
                    Register::FT0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::fneg(
                            dtype.clone(),
                            *dest_reg,
                            reg,
                        )));
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::fneg(
                            dtype.clone(),
                            Register::FT1,
                            reg,
                        )));
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT1,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::UnaryOp {
                op: UnaryOperator::Negate,
                operand: operand @ ir::Operand::Register { .. },
                dtype: ir::Dtype::Int { .. },
            } => {
                let reg = load_operand_to_reg(
                    operand.clone(),
                    Register::T0,
                    &mut res,
                    register_mp,
                    float_mp,
                );

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::Seqz {
                            rd: *dest_reg,
                            rs: reg,
                        }));
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::Seqz {
                            rd: Register::T0,
                            rs: reg,
                        }));
                        res.extend(mk_stype(
                            SType::SW,
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::UnaryOp {
                op: UnaryOperator::Plus,
                operand: operand @ ir::Operand::Register { .. },
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                let reg = load_operand_to_reg(
                    operand.clone(),
                    Register::T0,
                    &mut res,
                    register_mp,
                    float_mp,
                );

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                            rd: *dest_reg,
                            rs: reg,
                        }));
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            reg,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::UnaryOp {
                op: UnaryOperator::Plus,
                operand: operand @ ir::Operand::Register { .. },
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                let reg = load_operand_to_reg(
                    operand.clone(),
                    Register::FT0,
                    &mut res,
                    register_mp,
                    float_mp,
                );

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::Fmv {
                            rd: *dest_reg,
                            rs: reg,
                            data_size: DataSize::try_from(dtype.clone()).unwrap(),
                        }));
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            reg,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op,
                lhs: ir::Operand::Constant(c1),
                rhs: ir::Operand::Constant(c2),
                dtype,
            } => {
                let v = crate::ir::interp::calculator::calculate_binary_operator_expression(
                    op,
                    Value::try_from(c1.clone()).unwrap(),
                    Value::try_from(c2.clone()).unwrap(),
                )
                .unwrap();
                match (v, register_mp.get(&RegisterId::Temp { bid, iid }).unwrap()) {
                    (
                        Value::Int { value, .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let data_size = DataSize::try_from(dtype.clone()).unwrap();
                        res.extend(mk_itype(
                            IType::Addi(data_size),
                            *dest_reg,
                            Register::Zero,
                            value as u64 & data_size.mask(),
                        ));
                    }
                    (
                        Value::Float { value, width },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let label = float_mp.get_label(Float { value, width });
                        res.push(asm::Instruction::Pseudo(Pseudo::La {
                            rd: Register::T0,
                            symbol: label,
                        }));
                        res.push(asm::Instruction::IType {
                            instr: IType::load(dtype.clone()),
                            rd: *dest_reg,
                            rs1: Register::T0,
                            imm: Immediate::Value(0),
                        });
                    }
                    (
                        Value::Int { value, .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let data_size = DataSize::try_from(dtype.clone()).unwrap();
                        res.extend(mk_itype(
                            IType::Addi(data_size),
                            Register::T0,
                            Register::Zero,
                            value as u64 & data_size.mask(),
                        ));
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    (
                        Value::Float { value, width },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let label = float_mp.get_label(Float { value, width });
                        res.push(asm::Instruction::Pseudo(Pseudo::La {
                            rd: Register::T0,
                            symbol: label,
                        }));
                        res.push(asm::Instruction::IType {
                            instr: IType::load(dtype.clone()),
                            rd: Register::FT0,
                            rs1: Register::T0,
                            imm: Immediate::Value(0),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    _ => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Plus,
                lhs: x,
                rhs: ir::Operand::Constant(ir::Constant::Int { value: 0, .. }),
                dtype: dtype @ ir::Dtype::Int { .. },
            }
            | ir::Instruction::BinOp {
                op: BinaryOperator::Plus,
                lhs: ir::Operand::Constant(ir::Constant::Int { value: 0, .. }),
                rhs: x,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                let reg =
                    load_operand_to_reg(x.clone(), Register::T0, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                            rd: *dest_reg,
                            rs: reg,
                        }));
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            reg,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Plus,
                lhs: x,
                rhs: ir::Operand::Constant(ir::Constant::Int { value, .. }),
                dtype: dtype @ ir::Dtype::Int { .. },
            }
            | ir::Instruction::BinOp {
                op: BinaryOperator::Plus,
                lhs: ir::Operand::Constant(ir::Constant::Int { value, .. }),
                rhs: x,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                let reg =
                    load_operand_to_reg(x.clone(), Register::T0, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        let data_size = DataSize::try_from(dtype.clone()).unwrap();
                        res.extend(mk_itype(
                            IType::Addi(data_size),
                            *dest_reg,
                            reg,
                            *value as u64 & data_size.mask(),
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        let data_size = DataSize::try_from(dtype.clone()).unwrap();
                        res.extend(mk_itype(
                            IType::Addi(data_size),
                            Register::T0,
                            reg,
                            *value as u64 & data_size.mask(),
                        ));
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Plus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                let reg1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::add(dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::add(dtype.clone()),
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Plus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                let reg1 = load_operand_to_reg(
                    lhs.clone(),
                    Register::FT0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                let reg2 = load_operand_to_reg(
                    rhs.clone(),
                    Register::FT1,
                    &mut res,
                    register_mp,
                    float_mp,
                );

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fadd(dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fadd(dtype.clone()),
                            rd: Register::FT0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Minus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                let reg1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sub(dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sub(dtype.clone()),
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Minus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                let reg1 = load_operand_to_reg(
                    lhs.clone(),
                    Register::FT0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                let reg2 = load_operand_to_reg(
                    rhs.clone(),
                    Register::FT1,
                    &mut res,
                    register_mp,
                    float_mp,
                );

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fsub(dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fsub(dtype.clone()),
                            rd: Register::FT0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::Multiply,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                let reg1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::mul(dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::mul(dtype.clone()),
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Multiply,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                let reg1 = load_operand_to_reg(
                    lhs.clone(),
                    Register::FT0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                let reg2 = load_operand_to_reg(
                    rhs.clone(),
                    Register::FT1,
                    &mut res,
                    register_mp,
                    float_mp,
                );

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fmul(dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fmul(dtype.clone()),
                            rd: Register::FT0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::Divide,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { is_signed, .. },
            } => {
                let reg1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::div(dtype.clone(), *is_signed),
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::div(dtype.clone(), *is_signed),
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Divide,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                let reg1 = load_operand_to_reg(
                    lhs.clone(),
                    Register::FT0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                let reg2 = load_operand_to_reg(
                    rhs.clone(),
                    Register::FT1,
                    &mut res,
                    register_mp,
                    float_mp,
                );

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fdiv(dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fdiv(dtype.clone()),
                            rd: Register::FT0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::Equals,
                lhs: x,
                rhs:
                    ir::Operand::Constant(
                        c @ ir::Constant::Int {
                            is_signed: true, ..
                        },
                    ),
                dtype: target_dtype @ ir::Dtype::Int { .. },
            }
            | ir::Instruction::BinOp {
                op: BinaryOperator::Equals,
                rhs: x,
                lhs:
                    ir::Operand::Constant(
                        c @ ir::Constant::Int {
                            is_signed: true, ..
                        },
                    ),
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let c = c.clone().minus();
                let ir::Constant::Int { value, .. } = c else {unreachable!()};
                let reg1 =
                    load_operand_to_reg(x.clone(), Register::T0, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        let data_size = DataSize::try_from(x.dtype()).unwrap();
                        res.extend(mk_itype(
                            IType::Addi(data_size),
                            *dest_reg,
                            reg1,
                            value as u64 & data_size.mask(),
                        ));
                        res.push(asm::Instruction::Pseudo(Pseudo::Seqz {
                            rd: *dest_reg,
                            rs: *dest_reg,
                        }));
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        let data_size = DataSize::try_from(x.dtype()).unwrap();
                        res.extend(mk_itype(
                            IType::Addi(data_size),
                            Register::T0,
                            reg1,
                            value as u64 & data_size.mask(),
                        ));
                        res.push(asm::Instruction::Pseudo(Pseudo::Seqz {
                            rd: Register::T0,
                            rs: Register::T0,
                        }));
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Equals,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let dtype = lhs.dtype();
                match (
                    &dtype,
                    register_mp.get(&RegisterId::Temp { bid, iid }).unwrap(),
                ) {
                    (
                        ir::Dtype::Int { .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg1 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg2 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Xor,
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.push(asm::Instruction::Pseudo(Pseudo::Seqz {
                            rd: *dest_reg,
                            rs: *dest_reg,
                        }));
                    }
                    (
                        ir::Dtype::Int { .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg1 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg2 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Xor,
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.push(asm::Instruction::Pseudo(Pseudo::Seqz {
                            rd: Register::T0,
                            rs: Register::T0,
                        }));
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    _ => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::NotEquals,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let dtype = lhs.dtype();
                match (
                    &dtype,
                    register_mp.get(&RegisterId::Temp { bid, iid }).unwrap(),
                ) {
                    (
                        ir::Dtype::Int { .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg1 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg2 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Xor,
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.push(asm::Instruction::Pseudo(Pseudo::Snez {
                            rd: Register::T0,
                            rs: Register::T0,
                        }));
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    (
                        ir::Dtype::Int { .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg1 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg2 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Xor,
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.push(asm::Instruction::Pseudo(Pseudo::Snez {
                            rd: *dest_reg,
                            rs: *dest_reg,
                        }));
                    }
                    _ => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::Less,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let dtype = lhs.dtype();

                match (
                    &dtype,
                    register_mp.get(&RegisterId::Temp { bid, iid }).unwrap(),
                ) {
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg1 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg2 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                    }
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg1 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg2 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    (
                        ir::Dtype::Float { .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg1 = load_operand_to_reg(
                            lhs.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg2 = load_operand_to_reg(
                            rhs.clone(),
                            Register::FT1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                    }
                    (
                        ir::Dtype::Float { .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg1 = load_operand_to_reg(
                            lhs.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg2 = load_operand_to_reg(
                            rhs.clone(),
                            Register::FT1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    _ => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::LessOrEqual,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let dtype = lhs.dtype();
                match (
                    &dtype,
                    register_mp.get(&RegisterId::Temp { bid, iid }).unwrap(),
                ) {
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg0),
                        });
                        res.push(asm::Instruction::IType {
                            instr: IType::Xori,
                            rd: *dest_reg,
                            rs1: *dest_reg,
                            imm: Immediate::Value(1),
                        });
                    }
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg0),
                        });
                        res.push(asm::Instruction::IType {
                            instr: IType::Xori,
                            rd: Register::T0,
                            rs1: Register::T0,
                            imm: Immediate::Value(1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    (
                        ir::Dtype::Float { .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::FT1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg0),
                        });
                        res.push(asm::Instruction::IType {
                            instr: IType::Xori,
                            rd: *dest_reg,
                            rs1: *dest_reg,
                            imm: Immediate::Value(1),
                        });
                    }
                    (
                        ir::Dtype::Float { .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::FT1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg0),
                        });
                        res.push(asm::Instruction::IType {
                            instr: IType::Xori,
                            rd: Register::T0,
                            rs1: Register::T0,
                            imm: Immediate::Value(1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    _ => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::Greater,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let dtype = lhs.dtype();
                match (
                    &dtype,
                    register_mp.get(&RegisterId::Temp { bid, iid }).unwrap(),
                ) {
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg0),
                        });
                    }
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg0),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    (
                        ir::Dtype::Float { .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::FT1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg1,
                            rs2: Some(reg0),
                        });
                    }
                    (
                        ir::Dtype::Float { .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::FT1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: Register::T0,
                            rs1: reg1,
                            rs2: Some(reg0),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    _ => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::GreaterOrEqual,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let dtype = lhs.dtype();
                match (
                    &dtype,
                    register_mp.get(&RegisterId::Temp { bid, iid }).unwrap(),
                ) {
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: *dest_reg,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                        res.push(asm::Instruction::IType {
                            instr: IType::Xori,
                            rd: *dest_reg,
                            rs1: *dest_reg,
                            imm: Immediate::Value(1),
                        });
                    }
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: Register::T0,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                        res.push(asm::Instruction::IType {
                            instr: IType::Xori,
                            rd: Register::T0,
                            rs1: Register::T0,
                            imm: Immediate::Value(1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    _ => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::Modulo,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let dtype = lhs.dtype();
                match (
                    &dtype,
                    register_mp.get(&RegisterId::Temp { bid, iid }).unwrap(),
                ) {
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::rem(dtype.clone(), *is_signed),
                            rd: *dest_reg,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                    }
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg0 = load_operand_to_reg(
                            lhs.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        let reg1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::T1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::rem(dtype.clone(), *is_signed),
                            rd: Register::T0,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    _ => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::ShiftLeft,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let reg0 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg1 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sll(target_dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sll(target_dtype.clone()),
                            rd: Register::T0,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            // all shr is i32
            ir::Instruction::BinOp {
                op: BinaryOperator::ShiftRight,
                lhs,
                rhs: ir::Operand::Constant(ir::Constant::Int { value, .. }),
                dtype:
                    target_dtype @ ir::Dtype::Int {
                        width: 32,
                        is_signed: true,
                        ..
                    },
            } => {
                let reg0 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);

                let instr = match lhs.dtype() {
                    ir::Dtype::Int {
                        width: 32,
                        is_signed: true,
                        ..
                    } => IType::Srai(DataSize::Double),
                    ir::Dtype::Int {
                        width: 32,
                        is_signed: false,
                        ..
                    } => IType::Srli(DataSize::Word),
                    _ => unreachable!(),
                };
                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::IType {
                            instr,
                            rd: *dest_reg,
                            rs1: reg0,
                            imm: Immediate::Value(*value as u64),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::IType {
                            instr,
                            rd: Register::T0,
                            rs1: reg0,
                            imm: Immediate::Value(*value as u64),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::ShiftRight,
                rhs: ir::Operand::Constant(ir::Constant::Int { .. }),
                dtype: ir::Dtype::Int { .. },
                ..
            } => {
                unreachable!()
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::ShiftRight,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let reg0 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg1 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sra(target_dtype.clone()),
                            rd: *dest_reg,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sra(target_dtype.clone()),
                            rd: Register::T0,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::BitwiseXor,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let reg0 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg1 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Xor,
                            rd: *dest_reg,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Xor,
                            rd: Register::T0,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::BitwiseAnd,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let reg0 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg1 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::And,
                            rd: *dest_reg,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::And,
                            rd: Register::T0,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::BitwiseOr,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let reg0 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg1 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Or,
                            rd: *dest_reg,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Or,
                            rd: Register::T0,
                            rs1: reg0,
                            rs2: Some(reg1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::Store {
                ptr:
                    ir::Operand::Register {
                        rid,
                        dtype: ir::Dtype::Pointer { inner, .. },
                    },
                value: operand @ ir::Operand::Constant(ir::Constant::Int { .. }),
            } => {
                let reg1 = load_operand_to_reg(
                    operand.clone(),
                    Register::T1,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                let DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) = *register_mp.get(rid).unwrap() else {unreachable!()};
                res.extend(mk_itype(
                    IType::LD,
                    Register::T0,
                    Register::S0,
                    offset_to_s0 as u64,
                ));
                res.push(asm::Instruction::SType {
                    instr: SType::store((**inner).clone()),
                    rs1: Register::T0,
                    rs2: reg1,
                    imm: Immediate::Value(0),
                });
            }
            ir::Instruction::Store {
                ptr:
                    ir::Operand::Register {
                        rid,
                        dtype: ir::Dtype::Pointer { inner, .. },
                    },
                value: operand @ ir::Operand::Constant(ir::Constant::Float { .. }),
            } => {
                let reg0 = load_operand_to_reg(
                    operand.clone(),
                    Register::FT0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                let DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0  }) = *register_mp.get(rid).unwrap() else {unreachable!()};
                res.extend(mk_itype(
                    IType::LD,
                    Register::T0,
                    Register::S0,
                    offset_to_s0 as u64,
                ));
                res.push(asm::Instruction::SType {
                    instr: SType::store((**inner).clone()),
                    rs1: Register::T0,
                    rs2: reg0,
                    imm: Immediate::Value(0),
                });
            }
            ir::Instruction::Store {
                ptr:
                    ir::Operand::Register {
                        dtype: ir::Dtype::Pointer { .. },
                        ..
                    },
                value: ir::Operand::Constant(ir::Constant::GlobalVariable { .. }),
            } => {
                todo!()
            }
            ir::Instruction::Store {
                ptr:
                    ir::Operand::Register {
                        rid: ptr_rid,
                        dtype: ir::Dtype::Pointer { .. },
                    },
                value:
                    ir::Operand::Register {
                        rid: value_rid,
                        dtype,
                    },
            } => {
                let DirectOrInDirect::Direct(reg_or_stack) = register_mp.get(ptr_rid).unwrap() else {unreachable!()};
                let dest_location = match reg_or_stack {
                    RegOrStack::Reg(reg) => *reg,
                    RegOrStack::Stack { offset_to_s0 } => {
                        res.extend(mk_itype(
                            IType::LD,
                            Register::T3,
                            Register::S0,
                            *offset_to_s0 as u64,
                        ));
                        Register::T3
                    }
                    RegOrStack::IntRegNotSure | RegOrStack::FloatRegNotSure => unreachable!(),
                };
                match register_mp.get(value_rid).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0: src }) => {
                        cp_to_indirect_target(
                            (Register::S0, *src),
                            dest_location,
                            0,
                            dtype.clone(),
                            &mut res,
                            source,
                        );
                    }
                    DirectOrInDirect::Direct(RegOrStack::Reg(value_reg)) => {
                        // store reg to location
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            dest_location,
                            *value_reg,
                            0,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(src) => {
                        let source_location = match src {
                            RegOrStack::Reg(reg) => *reg,
                            RegOrStack::Stack { offset_to_s0 } => {
                                res.extend(mk_itype(
                                    IType::LD,
                                    Register::T2,
                                    Register::S0,
                                    *offset_to_s0 as u64,
                                ));
                                Register::T2
                            }
                            RegOrStack::IntRegNotSure | RegOrStack::FloatRegNotSure => {
                                unreachable!()
                            }
                        };
                        cp_from_indirect_to_indirect(
                            source_location,
                            dest_location,
                            0,
                            dtype.clone(),
                            &mut res,
                            source,
                        );
                    }
                }
            }
            ir::Instruction::Store {
                ptr:
                    ir::Operand::Constant(ir::Constant::GlobalVariable {
                        name,
                        dtype: dtype @ ir::Dtype::Int { .. },
                    }),
                value,
            } => {
                res.push(asm::Instruction::Pseudo(Pseudo::La {
                    rd: Register::T1,
                    symbol: Label(name.clone()),
                }));
                let reg0 = load_operand_to_reg(
                    value.clone(),
                    Register::T0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                res.push(asm::Instruction::SType {
                    instr: SType::store(dtype.clone()),
                    rs1: Register::T1,
                    rs2: reg0,
                    imm: Immediate::Value(0),
                });
            }
            ir::Instruction::Load {
                ptr:
                    ir::Operand::Register {
                        rid,
                        dtype: ir::Dtype::Pointer { inner, .. },
                    },
            } => {
                let DirectOrInDirect::Direct(src) = register_mp.get(rid).unwrap() else {unreachable!()};
                let source_location = match src {
                    RegOrStack::Reg(reg) => *reg,
                    RegOrStack::Stack { offset_to_s0 } => {
                        res.extend(mk_itype(
                            IType::LD,
                            Register::T2,
                            Register::S0,
                            *offset_to_s0 as u64,
                        ));
                        Register::T2
                    }
                    RegOrStack::IntRegNotSure | RegOrStack::FloatRegNotSure => unreachable!(),
                };

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.extend(mk_itype(
                            IType::load((**inner).clone()),
                            *dest_reg,
                            source_location,
                            0,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        cp_from_indirect_source(
                            source_location,
                            *offset_to_s0,
                            0,
                            (**inner).clone(),
                            &mut res,
                            source,
                        );
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::Load {
                ptr:
                    ir::Operand::Constant(ir::Constant::GlobalVariable {
                        name,
                        dtype: dtype @ ir::Dtype::Int { .. },
                    }),
            } => match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                    res.push(asm::Instruction::Pseudo(Pseudo::La {
                        rd: *dest_reg,
                        symbol: Label(name.clone()),
                    }));
                    res.push(asm::Instruction::IType {
                        instr: IType::load(dtype.clone()),
                        rd: *dest_reg,
                        rs1: *dest_reg,
                        imm: Immediate::Value(0),
                    });
                }
                DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                    res.push(asm::Instruction::Pseudo(Pseudo::La {
                        rd: Register::T0,
                        symbol: Label(name.clone()),
                    }));
                    res.push(asm::Instruction::IType {
                        instr: IType::load(dtype.clone()),
                        rd: Register::T0,
                        rs1: Register::T0,
                        imm: Immediate::Value(0),
                    });
                    res.extend(mk_stype(
                        SType::store(dtype.clone()),
                        Register::S0,
                        Register::T0,
                        *offset_to_s0 as u64,
                    ));
                }
                DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                DirectOrInDirect::InDirect(_) => unreachable!(),
            },
            ir::Instruction::Load {
                ptr:
                    ir::Operand::Constant(ir::Constant::GlobalVariable {
                        name,
                        dtype: dtype @ ir::Dtype::Float { .. },
                    }),
            } => {
                res.push(asm::Instruction::Pseudo(Pseudo::La {
                    rd: Register::T0,
                    symbol: Label(name.clone()),
                }));
                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::IType {
                            instr: IType::load(dtype.clone()),
                            rd: *dest_reg,
                            rs1: Register::T0,
                            imm: Immediate::Value(0),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::IType {
                            instr: IType::load(dtype.clone()),
                            rd: Register::FT0,
                            rs1: Register::T0,
                            imm: Immediate::Value(0),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::Call { callee, args, .. } => {
                let (
                    params_dtype,
                    ret_dtype,
                    FunctionAbi {
                        params_alloc,
                        caller_alloc,
                        ret_alloc,
                    },
                ) = match callee {
                    ir::Operand::Constant(ir::Constant::GlobalVariable {
                        dtype: ir::Dtype::Function { ret, params },
                        name,
                    }) => (
                        params,
                        ret.as_ref(),
                        function_abi_mp.get(name).unwrap().clone(),
                    ),
                    ir::Operand::Register {
                        dtype: ir::Dtype::Pointer { inner, .. },
                        ..
                    } => {
                        let ir::Dtype::Function { ret, params } = &**inner else {unreachable!()};
                        let function_signature = FunctionSignature {
                            ret: (**ret).clone(),
                            params: params.clone(),
                        };
                        (params, ret.as_ref(), function_signature.try_alloc(source))
                    }
                    _ => unreachable!(),
                };
                if caller_alloc != 0 {
                    res.extend(mk_itype(
                        IType::Addi(DataSize::Double),
                        Register::Sp,
                        Register::Sp,
                        (-(<usize as TryInto<i64>>::try_into(caller_alloc).unwrap())) as u64,
                    ));
                }
                for (operand, alloc, dtype) in izip!(args, params_alloc, params_dtype) {
                    match (alloc, operand) {
                        (
                            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(
                                reg,
                            ))),
                            _,
                        ) => {
                            store_operand_to_reg(
                                operand.clone(),
                                reg,
                                &mut res,
                                register_mp,
                                float_mp,
                            );
                        }
                        (
                            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(
                                RegOrStack::Stack { offset_to_s0 },
                            )),
                            ir::Operand::Constant(..),
                        ) => {
                            assert!(offset_to_s0 > 0);
                            operand_to_stack(
                                operand.clone(),
                                (Register::Sp, offset_to_s0 as u64),
                                &mut res,
                                register_mp,
                                float_mp,
                            );
                        }
                        (
                            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(
                                RegOrStack::Stack { .. },
                            )),
                            ir::Operand::Register { .. },
                        ) => {
                            unreachable!("{dtype}")
                        }
                        (
                            ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Reg(
                                target_reg,
                            ))),
                            ir::Operand::Register { rid, .. },
                        ) => match register_mp.get(rid).unwrap() {
                            DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                                res.extend(mk_itype(
                                    IType::Addi(DataSize::Double),
                                    target_reg,
                                    Register::S0,
                                    *offset_to_s0 as u64,
                                ));
                            }
                            DirectOrInDirect::Direct(RegOrStack::Reg(_)) => unreachable!(),
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => {
                                unreachable!()
                            }
                            DirectOrInDirect::InDirect(RegOrStack::Stack { offset_to_s0 }) => {
                                res.extend(mk_itype(
                                    IType::LD,
                                    target_reg,
                                    Register::S0,
                                    *offset_to_s0 as u64,
                                ));
                            }
                            DirectOrInDirect::InDirect(RegOrStack::Reg(reg)) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: target_reg,
                                    rs: *reg,
                                }));
                            }
                            DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure)
                            | DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure) => {
                                unreachable!()
                            }
                        },
                        (
                            ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(
                                RegOrStack::Stack {
                                    offset_to_s0: target_offset,
                                },
                            )),
                            ir::Operand::Register { rid, .. },
                        ) => match register_mp.get(rid).unwrap() {
                            DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                                res.extend(mk_itype(
                                    IType::Addi(DataSize::Double),
                                    Register::T0,
                                    Register::S0,
                                    *offset_to_s0 as u64,
                                ));
                                res.extend(mk_stype(
                                    SType::Store(DataSize::Double),
                                    Register::Sp,
                                    Register::T0,
                                    target_offset as u64,
                                ));
                            }
                            DirectOrInDirect::Direct(RegOrStack::Reg(_)) => {
                                unreachable!()
                            }
                            DirectOrInDirect::InDirect(RegOrStack::Stack { offset_to_s0 }) => {
                                res.extend(mk_itype(
                                    IType::LD,
                                    Register::T0,
                                    Register::S0,
                                    *offset_to_s0 as u64,
                                ));
                                res.extend(mk_stype(
                                    SType::Store(DataSize::Double),
                                    Register::Sp,
                                    Register::T0,
                                    target_offset as u64,
                                ));
                            }
                            DirectOrInDirect::InDirect(RegOrStack::Reg(reg)) => {
                                res.extend(mk_stype(
                                    SType::Store(DataSize::Double),
                                    Register::Sp,
                                    *reg,
                                    target_offset as u64,
                                ));
                            }
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure)
                            | DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure)
                            | DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure) => {
                                unreachable!()
                            }
                        },
                        (
                            ParamAlloc::StructInRegister(v),
                            ir::Operand::Register {
                                rid,
                                dtype:
                                    ir::Dtype::Struct {
                                        name,
                                        fields,
                                        size_align_offsets,
                                        ..
                                    },
                            },
                        ) => {
                            let DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 : base_offset}) = *register_mp.get(rid).unwrap() else {unreachable!()};

                            let Some((_, _, offsets)) = (if size_align_offsets.is_some() {
                                size_align_offsets.clone()
                            } else {
                                source
                                    .structs
                                    .get(name.as_ref().unwrap())
                                    .and_then(|x| x.as_ref())
                                    .and_then(|x| x.get_struct_size_align_offsets())
                                    .and_then(|x| x.as_ref()).cloned()
                            } ) else {unreachable!()};

                            let Some(fields) = (if fields.is_some() {
                                fields.clone()
                            } else {
                                source
                                    .structs
                                    .get(name.as_ref().unwrap())
                                    .and_then(|x| x.as_ref())
                                    .and_then(|x| x.get_struct_fields())
                                    .and_then(|x| x.as_ref()).cloned()
                            } ) else {unreachable!()};

                            for (register_couple, offset, dtype) in izip!(v, offsets, fields) {
                                match register_couple {
                                    RegisterCouple::Single(register) => {
                                        res.extend(mk_itype(
                                            IType::load((*dtype).clone()),
                                            register,
                                            Register::S0,
                                            (base_offset
                                                + <usize as TryInto<i64>>::try_into(offset)
                                                    .unwrap())
                                                as u64,
                                        ));
                                    }
                                    RegisterCouple::Double(register) => {
                                        res.extend(mk_itype(
                                            IType::LD,
                                            register,
                                            Register::S0,
                                            (base_offset
                                                + <usize as TryInto<i64>>::try_into(offset)
                                                    .unwrap())
                                                as u64,
                                        ));
                                    }
                                    RegisterCouple::MergedToPrevious => {}
                                }
                            }
                            todo!()
                        }
                        _ => unreachable!(),
                    }
                }

                match ret_alloc {
                    RetLocation::OnStack => {
                        let DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) = register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() else {unreachable!()} ;
                        res.extend(mk_itype(
                            IType::Addi(DataSize::Double),
                            Register::T0,
                            Register::S0,
                            *offset_to_s0 as u64,
                        ));
                        res.extend(mk_stype(SType::SD, Register::Sp, Register::T0, 0));
                    }
                    RetLocation::InRegister => {}
                };

                match callee {
                    ir::Operand::Constant(ir::Constant::GlobalVariable {
                        name,
                        dtype: ir::Dtype::Function { .. },
                    }) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::Call {
                            offset: Label(name.clone()),
                        }));
                    }
                    ir::Operand::Register {
                        rid,
                        dtype: ir::Dtype::Pointer { .. },
                    } => {
                        let rs = match register_mp.get(rid).unwrap() {
                            DirectOrInDirect::Direct(RegOrStack::Reg(reg)) => *reg,
                            DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                                res.extend(mk_itype(
                                    IType::LD,
                                    Register::T0,
                                    Register::S0,
                                    *offset_to_s0 as u64,
                                ));
                                Register::T0
                            }
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => {
                                unreachable!()
                            }
                            DirectOrInDirect::InDirect(_) => unreachable!(),
                        };
                        res.push(asm::Instruction::Pseudo(Pseudo::Jalr { rs }));
                    }
                    _ => unreachable!(),
                }
                match ret_alloc {
                    RetLocation::OnStack => {}
                    RetLocation::InRegister => match ret_dtype {
                        ir::Dtype::Unit { .. } => {}
                        ir::Dtype::Pointer { .. } | ir::Dtype::Int { .. } => {
                            match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                                DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                                    res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                        rd: *dest_reg,
                                        rs: Register::A0,
                                    }));
                                }
                                DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                                    res.extend(mk_stype(
                                        SType::store(ret_dtype.clone()),
                                        Register::S0,
                                        Register::A0,
                                        *offset_to_s0 as u64,
                                    ));
                                }
                                DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                                | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => {
                                    unreachable!()
                                }
                                DirectOrInDirect::InDirect(_) => unreachable!(),
                            }
                        }
                        ir::Dtype::Float { .. } => {
                            match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                                DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                                    res.push(asm::Instruction::Pseudo(Pseudo::Fmv {
                                        rd: *dest_reg,
                                        rs: Register::FA0,
                                        data_size: DataSize::try_from(ret_dtype.clone()).unwrap(),
                                    }));
                                }
                                DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                                    res.extend(mk_stype(
                                        SType::store(ret_dtype.clone()),
                                        Register::S0,
                                        Register::FA0,
                                        *offset_to_s0 as u64,
                                    ));
                                }
                                DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                                | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => {
                                    unreachable!()
                                }
                                DirectOrInDirect::InDirect(_) => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    },
                }
                if caller_alloc != 0 {
                    res.extend(mk_itype(
                        IType::Addi(DataSize::Double),
                        Register::Sp,
                        Register::Sp,
                        (caller_alloc).try_into().unwrap(),
                    ));
                }
            }
            ir::Instruction::TypeCast {
                value,
                target_dtype:
                    to @ ir::Dtype::Int {
                        width: width_target,
                        is_signed: is_signed_target,
                        ..
                    },
            } => {
                let from = value.dtype();

                match (
                    &from,
                    register_mp.get(&RegisterId::Temp { bid, iid }).unwrap(),
                ) {
                    (
                        ir::Dtype::Int {
                            width: 1,
                            is_signed: true,
                            ..
                        },
                        DirectOrInDirect::Direct(RegOrStack::Reg(_dest_reg)),
                    ) => match (width_target, is_signed_target) {
                        (8, true) => unimplemented!(),
                        (8, false) => unimplemented!(),
                        (16, true) => unimplemented!(),
                        (16, false) => unimplemented!(),
                        (32, true) => unimplemented!(),
                        (32, false) => unimplemented!(),
                        (64, true) => unimplemented!(),
                        (64, false) => unimplemented!(),
                        _ => unreachable!(),
                    },
                    (
                        ir::Dtype::Int {
                            width: 1,
                            is_signed: false,
                            ..
                        },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                            rd: *dest_reg,
                            rs: reg,
                        }));
                    }
                    (
                        ir::Dtype::Int {
                            width: 8,
                            is_signed: true,
                            ..
                        },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        match (width_target, is_signed_target) {
                            (8, true) => unreachable!(),
                            (8, false) => {
                                res.push(asm::Instruction::IType {
                                    instr: IType::Andi,
                                    rd: *dest_reg,
                                    rs1: reg,
                                    imm: Immediate::Value(255),
                                });
                            }
                            (16, true) => unimplemented!(),
                            (16, false) => unimplemented!(),
                            (32, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                            }
                            (32, false) => unimplemented!(),
                            (64, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                            }
                            (64, false) => unimplemented!(),
                            _ => unreachable!(),
                        }
                    }
                    (
                        ir::Dtype::Int {
                            width: 8,
                            is_signed: false,
                            ..
                        },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        match (width_target, is_signed_target) {
                            (8, true) => unimplemented!(),
                            (8, false) => unimplemented!(),
                            (16, true) => unimplemented!(),
                            (16, false) => unimplemented!(),
                            (32, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                            }
                            (32, false) => unimplemented!(),
                            (64, true) => unimplemented!(),
                            (64, false) => unimplemented!(),
                            _ => unreachable!(),
                        }
                    }
                    (
                        ir::Dtype::Int {
                            width: 16,
                            is_signed: true,
                            ..
                        },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        match (width_target, is_signed_target) {
                            (8, true) => unimplemented!(),
                            (8, false) => unimplemented!(),
                            (16, true) => unimplemented!(),
                            (16, false) => unimplemented!(),
                            (32, true) => unimplemented!(),
                            (32, false) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                            }
                            (64, true) => unimplemented!(),
                            (64, false) => unimplemented!(),
                            _ => unreachable!(),
                        }
                    }
                    (
                        ir::Dtype::Int {
                            width: 16,
                            is_signed: false,
                            ..
                        },
                        DirectOrInDirect::Direct(RegOrStack::Reg(_dest_reg)),
                    ) => match (width_target, is_signed_target) {
                        (8, true) => unimplemented!(),
                        (8, false) => unimplemented!(),
                        (16, true) => unimplemented!(),
                        (16, false) => unimplemented!(),
                        (32, true) => unimplemented!(),
                        (32, false) => unimplemented!(),
                        (64, true) => unimplemented!(),
                        (64, false) => unimplemented!(),
                        _ => unreachable!(),
                    },
                    (
                        ir::Dtype::Int {
                            width: 32,
                            is_signed: true,
                            ..
                        },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        match (width_target, is_signed_target) {
                            (8, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                                res.push(asm::Instruction::IType {
                                    instr: IType::Slli(DataSize::Double),
                                    rd: *dest_reg,
                                    rs1: *dest_reg,
                                    imm: Immediate::Value(56),
                                });
                                res.push(asm::Instruction::IType {
                                    instr: IType::Srai(DataSize::Double),
                                    rd: *dest_reg,
                                    rs1: *dest_reg,
                                    imm: Immediate::Value(56),
                                });
                            }
                            (8, false) => {
                                res.push(asm::Instruction::IType {
                                    instr: IType::Andi,
                                    rd: *dest_reg,
                                    rs1: reg,
                                    imm: Immediate::Value(255),
                                });
                            }
                            (16, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                                res.push(asm::Instruction::IType {
                                    instr: IType::Slli(DataSize::Double),
                                    rd: *dest_reg,
                                    rs1: *dest_reg,
                                    imm: Immediate::Value(48),
                                });
                                res.push(asm::Instruction::IType {
                                    instr: IType::Srai(DataSize::Double),
                                    rd: *dest_reg,
                                    rs1: *dest_reg,
                                    imm: Immediate::Value(48),
                                });
                            }
                            (16, false) => unimplemented!(),
                            (32, true) => unimplemented!(),
                            (32, false) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                            }
                            (64, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                            }
                            (64, false) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                            }
                            _ => unreachable!(),
                        }
                    }
                    (
                        ir::Dtype::Int {
                            width: 32,
                            is_signed: false,
                            ..
                        },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        match (width_target, is_signed_target) {
                            (8, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                                res.push(asm::Instruction::IType {
                                    instr: IType::Slli(DataSize::Double),
                                    rd: *dest_reg,
                                    rs1: *dest_reg,
                                    imm: Immediate::Value(56),
                                });
                                res.push(asm::Instruction::IType {
                                    instr: IType::Srai(DataSize::Double),
                                    rd: *dest_reg,
                                    rs1: *dest_reg,
                                    imm: Immediate::Value(56),
                                });
                            }
                            (8, false) => unimplemented!(),
                            (16, true) => unimplemented!(),
                            (16, false) => unimplemented!(),
                            (32, true) => unimplemented!(),
                            (32, false) => unimplemented!(),
                            (64, true) => unimplemented!(),
                            (64, false) => unimplemented!(),
                            _ => unreachable!(),
                        }
                    }
                    (
                        ir::Dtype::Int {
                            width: 64,
                            is_signed: true,
                            ..
                        },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        match (width_target, is_signed_target) {
                            (8, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                                res.push(asm::Instruction::IType {
                                    instr: IType::Slli(DataSize::Double),
                                    rd: *dest_reg,
                                    rs1: *dest_reg,
                                    imm: Immediate::Value(56),
                                });
                                res.push(asm::Instruction::IType {
                                    instr: IType::Srai(DataSize::Double),
                                    rd: *dest_reg,
                                    rs1: *dest_reg,
                                    imm: Immediate::Value(56),
                                });
                            }
                            (8, false) => {
                                res.push(asm::Instruction::IType {
                                    instr: IType::Andi,
                                    rd: *dest_reg,
                                    rs1: reg,
                                    imm: Immediate::Value(255),
                                });
                            }
                            (16, true) => unimplemented!(),
                            (16, false) => unimplemented!(),
                            (32, true) => unimplemented!(),
                            (32, false) => unimplemented!(),
                            (64, true) => unimplemented!(),
                            (64, false) => unimplemented!(),
                            _ => unreachable!(),
                        }
                    }
                    (
                        ir::Dtype::Int {
                            width: 64,
                            is_signed: false,
                            ..
                        },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        match (width_target, is_signed_target) {
                            (8, true) => unimplemented!(),
                            (8, false) => unimplemented!(),
                            (16, true) => unimplemented!(),
                            (16, false) => unimplemented!(),
                            (32, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::SextW {
                                    rd: *dest_reg,
                                    rs: reg,
                                }));
                            }
                            (32, false) => unimplemented!(),
                            (64, true) => unimplemented!(),
                            (64, false) => unimplemented!(),
                            _ => unreachable!(),
                        }
                    }
                    (
                        ir::Dtype::Int { .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.extend(mk_stype(
                            SType::store(to.clone()),
                            Register::S0,
                            reg,
                            *offset_to_s0 as u64,
                        ));
                    }
                    (
                        ir::Dtype::Float { .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let reg = load_operand_to_reg(
                            value.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: RType::fcvt_float_to_int(from, to.clone()),
                            rd: *dest_reg,
                            rs1: reg,
                            rs2: None,
                        });
                    }
                    (
                        ir::Dtype::Float { .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let reg = load_operand_to_reg(
                            value.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: RType::fcvt_float_to_int(from, to.clone()),
                            rd: Register::T0,
                            rs1: reg,
                            rs2: None,
                        });
                        res.extend(mk_stype(
                            SType::store(to.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    _ => unreachable!("{from}"),
                }
            }
            ir::Instruction::TypeCast {
                value,
                target_dtype: to @ ir::Dtype::Float { .. },
            } => {
                let from = value.dtype();

                match (
                    &from,
                    register_mp.get(&RegisterId::Temp { bid, iid }).unwrap(),
                ) {
                    (
                        ir::Dtype::Int { .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let rs1 = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: RType::fcvt_int_to_float(from, to.clone()),
                            rd: *dest_reg,
                            rs1,
                            rs2: None,
                        });
                    }
                    (
                        ir::Dtype::Int { .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let rs1 = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: RType::fcvt_int_to_float(from, to.clone()),
                            rd: Register::FT1,
                            rs1,
                            rs2: None,
                        });
                        res.extend(mk_stype(
                            SType::store(to.clone()),
                            Register::S0,
                            Register::FT1,
                            *offset_to_s0 as u64,
                        ));
                    }
                    (
                        ir::Dtype::Float { .. },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let rs1 = load_operand_to_reg(
                            value.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: RType::FcvtFloatToFloat {
                                from: DataSize::try_from(from).unwrap(),
                                to: DataSize::try_from(to.clone()).unwrap(),
                            },
                            rd: *dest_reg,
                            rs1,
                            rs2: None,
                        });
                    }
                    (
                        ir::Dtype::Float { .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let rs1 = load_operand_to_reg(
                            value.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: RType::FcvtFloatToFloat {
                                from: DataSize::try_from(from).unwrap(),
                                to: DataSize::try_from(to.clone()).unwrap(),
                            },
                            rd: Register::FT1,
                            rs1,
                            rs2: None,
                        });
                        res.extend(mk_stype(
                            SType::store(to.clone()),
                            Register::S0,
                            Register::FT1,
                            *offset_to_s0 as u64,
                        ));
                    }
                    _ => unreachable!(),
                }
            }
            ir::Instruction::GetElementPtr {
                ptr,
                offset,
                dtype: dtype @ ir::Dtype::Pointer { .. },
            } => {
                let rs1 = match ptr {
                    ir::Operand::Constant(ir::Constant::GlobalVariable { name, .. }) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::La {
                            rd: Register::T0,
                            symbol: Label(name.clone()),
                        }));
                        Register::T0
                    }
                    ptr @ ir::Operand::Register { .. } => load_operand_to_reg(
                        ptr.clone(),
                        Register::T0,
                        &mut res,
                        register_mp,
                        float_mp,
                    ),
                    _ => unreachable!(),
                };
                let rs2 = load_operand_to_reg(
                    offset.clone(),
                    Register::T1,
                    &mut res,
                    register_mp,
                    float_mp,
                );

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: RType::add(dtype.clone()),
                            rd: *dest_reg,
                            rs1,
                            rs2: Some(rs2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: RType::add(dtype.clone()),
                            rd: Register::T0,
                            rs1,
                            rs2: Some(rs2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::Nop => {}
            _ => unreachable!("{:?}", &**instr),
        }
    }

    match &block.exit {
        ir::BlockExit::Jump { arg } => {
            gen_jump_arg(func_name, arg, &mut res, register_mp, float_mp);
        }

        ir::BlockExit::ConditionalJump {
            condition,
            arg_then,
            arg_else,
        } => {
            let rs1 = load_operand_to_reg(
                condition.clone(),
                Register::T0,
                &mut res,
                register_mp,
                float_mp,
            );
            let else_label = gen_jump_arg_or_new_block(
                func_name,
                bid,
                arg_else,
                register_mp,
                float_mp,
                temp_block,
            );
            res.push(asm::Instruction::BType {
                instr: asm::BType::Beq,
                rs1,
                rs2: Register::Zero,
                imm: else_label,
            });
            gen_jump_arg(func_name, arg_then, &mut res, register_mp, float_mp);
        }

        ir::BlockExit::Switch {
            value,
            default,
            cases,
        } => {
            let dtype = value.dtype();
            let rs1 =
                load_operand_to_reg(value.clone(), Register::T0, &mut res, register_mp, float_mp);
            for (c, jump_arg) in cases {
                let ir::Constant::Int { value, .. } = c else {unreachable!()};
                let data_size = DataSize::try_from(dtype.clone()).unwrap();
                res.extend(mk_itype(
                    IType::Addi(data_size),
                    Register::T1,
                    Register::Zero,
                    *value as u64 & data_size.mask(),
                ));
                let then_label = gen_jump_arg_or_new_block(
                    func_name,
                    bid,
                    jump_arg,
                    register_mp,
                    float_mp,
                    temp_block,
                );
                res.push(asm::Instruction::BType {
                    instr: asm::BType::Beq,
                    rs1,
                    rs2: Register::T1,
                    imm: then_label,
                })
            }
            gen_jump_arg(func_name, default, &mut res, register_mp, float_mp);
        }

        ir::BlockExit::Return { value } => {
            match (&abi.ret_alloc, value.dtype()) {
                (_, ir::Dtype::Unit { .. }) => {}
                (RetLocation::InRegister, ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. }) => {
                    store_operand_to_reg(
                        value.clone(),
                        Register::A0,
                        &mut res,
                        register_mp,
                        float_mp,
                    );
                }
                (RetLocation::InRegister, ir::Dtype::Float { .. }) => {
                    store_operand_to_reg(
                        value.clone(),
                        Register::FA0,
                        &mut res,
                        register_mp,
                        float_mp,
                    );
                }
                (RetLocation::InRegister, ir::Dtype::Array { .. }) => unimplemented!(),
                (RetLocation::InRegister, ir::Dtype::Struct { .. }) => unreachable!(),
                (RetLocation::InRegister, ir::Dtype::Function { .. }) => unimplemented!(),
                (RetLocation::OnStack, ir::Dtype::Int { .. }) => unreachable!(),
                (RetLocation::OnStack, ir::Dtype::Float { .. }) => unreachable!(),
                (RetLocation::OnStack, ir::Dtype::Pointer { .. }) => unreachable!(),
                (RetLocation::OnStack, ir::Dtype::Array { .. }) => unimplemented!(),
                (RetLocation::OnStack, ir::Dtype::Struct { .. }) => match value {
                    ir::Operand::Constant(ir::Constant::Undef { .. }) => {}
                    ir::Operand::Register { rid, dtype } => match register_mp.get(rid).unwrap() {
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0: dest }) => {
                            res.extend(mk_itype(IType::LD, Register::T3, Register::S0, 0));
                            cp_to_indirect_target(
                                (Register::S0, *dest),
                                Register::T3,
                                0,
                                dtype.clone(),
                                &mut res,
                                source,
                            );
                        }
                        DirectOrInDirect::InDirect(_) => unimplemented!(),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                },
                (RetLocation::OnStack, ir::Dtype::Function { .. }) => unreachable!(),
                (_, ir::Dtype::Typedef { .. }) => unreachable!(),
            }
            res.extend(before_ret_instructions);
        }
        ir::BlockExit::Unreachable => unreachable!(),
    }

    res
}

// prepare args to jump block
fn gen_jump_arg(
    func_name: &str,
    jump_arg: &ir::JumpArg,
    res: &mut Vec<asm::Instruction>,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
    float_mp: &mut FloatMp,
) {
    let mut v: Vec<(Register, Register, ir::Dtype)> = Vec::new();
    for (aid, operand) in izip!(&jump_arg.args).enumerate() {
        match register_mp
            .get(&RegisterId::Arg {
                bid: jump_arg.bid,
                aid,
            })
            .unwrap()
        {
            DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => match operand {
                ir::Operand::Constant(_) => {
                    store_operand_to_reg(operand.clone(), *dest_reg, res, register_mp, float_mp);
                }
                ir::Operand::Register { rid, dtype } => match register_mp.get(rid).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(src_reg)) => {
                        if *src_reg != *dest_reg {
                            v.push((*src_reg, *dest_reg, dtype.clone()));
                        }
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.extend(mk_itype(
                            IType::load(dtype.clone()),
                            *dest_reg,
                            Register::S0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
                    DirectOrInDirect::InDirect(_) => todo!(),
                },
            },
            DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                operand_to_stack(
                    operand.clone(),
                    (Register::S0, *offset_to_s0 as u64),
                    res,
                    register_mp,
                    float_mp,
                );
            }
            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
            DirectOrInDirect::InDirect(_) => unreachable!(),
        };
    }
    res.extend(cp_parallel(v));

    res.push(asm::Instruction::Pseudo(Pseudo::J {
        offset: Label::new(func_name, jump_arg.bid),
    }));
}

fn gen_jump_arg_or_new_block(
    func_name: &str,
    from: BlockId,
    jump_arg: &ir::JumpArg,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
    float_mp: &mut FloatMp,
    temp_block: &mut Vec<asm::Block>,
) -> Label {
    if jump_arg.args.is_empty() {
        Label::new(func_name, jump_arg.bid)
    } else {
        let label = Label(format!(".{func_name}_{from}_{}", jump_arg.bid));
        temp_block.push(asm::Block {
            label: Some(label.clone()),
            instructions: vec![],
        });
        let res: &mut Vec<asm::Instruction> = &mut temp_block.last_mut().unwrap().instructions;

        gen_jump_arg(func_name, jump_arg, res, register_mp, float_mp);

        label
    }
}

/// may use T0
/// is operand is constant: store in or_register
fn load_operand_to_reg(
    operand: ir::Operand,
    or_register: Register,
    res: &mut Vec<asm::Instruction>,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
    float_mp: &mut FloatMp,
) -> Register {
    match operand {
        ir::Operand::Constant(ir::Constant::Int { value, .. }) => {
            let data_size = DataSize::try_from(operand.dtype()).unwrap();
            res.extend(mk_itype(
                IType::Addi(data_size),
                or_register,
                Register::Zero,
                value as u64 & data_size.mask(),
            ));
            or_register
        }
        ir::Operand::Constant(ref c @ ir::Constant::Float { value, width }) => {
            let label = float_mp.get_label(Float { value, width });
            res.push(asm::Instruction::Pseudo(Pseudo::La {
                rd: Register::T0,
                symbol: label,
            }));
            res.push(asm::Instruction::IType {
                instr: IType::load(c.dtype()),
                rd: or_register,
                rs1: Register::T0,
                imm: Immediate::Value(0),
            });
            or_register
        }
        ir::Operand::Constant(ir::Constant::GlobalVariable {
            name,
            dtype: ir::Dtype::Function { .. },
        }) => {
            res.push(asm::Instruction::Pseudo(Pseudo::La {
                rd: or_register,
                symbol: Label(name),
            }));
            or_register
        }
        ir::Operand::Constant(ir::Constant::Undef {
            dtype:
                dtype @ (ir::Dtype::Int { .. } | ir::Dtype::Float { .. } | ir::Dtype::Pointer { .. }),
        }) => {
            res.push(asm::Instruction::IType {
                instr: IType::load(dtype),
                rd: or_register,
                rs1: Register::Zero,
                imm: asm::Immediate::Value(0),
            });
            or_register
        }
        ir::Operand::Register { rid, dtype } => match register_mp.get(&rid).unwrap() {
            DirectOrInDirect::Direct(RegOrStack::Reg(reg)) => *reg,
            DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                res.extend(mk_itype(
                    IType::load(dtype),
                    or_register,
                    Register::S0,
                    *offset_to_s0 as u64,
                ));
                or_register
            }
            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure)
            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure) => unreachable!(),
            DirectOrInDirect::InDirect(_) => unreachable!(),
        },
        _ => unreachable!("{:?}", operand),
    }
}

fn store_operand_to_reg(
    operand: ir::Operand,
    target_register: Register,
    res: &mut Vec<asm::Instruction>,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
    float_mp: &mut FloatMp,
) {
    let dtype = operand.dtype();
    let x = load_operand_to_reg(operand, target_register, res, register_mp, float_mp);
    if x != target_register {
        match dtype {
            ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => {
                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                    rd: target_register,
                    rs: x,
                }));
            }
            ir::Dtype::Float { .. } => {
                res.push(asm::Instruction::Pseudo(Pseudo::Fmv {
                    rd: target_register,
                    rs: x,
                    data_size: DataSize::try_from(dtype).unwrap(),
                }));
            }
            _ => unreachable!(),
        }
    }
}

fn operand_to_stack(
    operand: ir::Operand,
    (target_register, target_base): (Register, u64),
    res: &mut Vec<asm::Instruction>,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
    float_mp: &mut FloatMp,
) {
    assert!(target_register == Register::Sp || target_register == Register::S0);
    match operand.dtype() {
        ir::Dtype::Unit { .. } => unreachable!(),
        dtype @ (ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. }) => {
            let reg = load_operand_to_reg(operand, Register::T0, res, register_mp, float_mp);
            res.extend(mk_stype(
                SType::store(dtype),
                target_register,
                reg,
                target_base,
            ));
        }
        dtype @ ir::Dtype::Float { .. } => {
            let reg = load_operand_to_reg(operand, Register::FT0, res, register_mp, float_mp);
            res.extend(mk_stype(
                SType::store(dtype),
                target_register,
                reg,
                target_base,
            ));
        }
        ir::Dtype::Array { .. } => unreachable!(),
        ir::Dtype::Struct { .. } => unreachable!(),
        ir::Dtype::Function { .. } => unreachable!(),
        ir::Dtype::Typedef { .. } => unreachable!(),
    }
}

/// must not write into `source_pointer`
/// must not modify A0
fn cp_from_indirect_source(
    src_location: Register, // the address of src
    target_base_to_s0: i64,
    offset: i64,
    dtype: ir::Dtype,
    res: &mut Vec<asm::Instruction>,
    source: &ir::TranslationUnit,
) {
    match &dtype {
        ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => {
            res.extend(mk_itype(
                asm::IType::load(dtype.clone()),
                Register::T0,
                src_location,
                offset as u64,
            ));
            res.extend(mk_stype(
                SType::store(dtype.clone()),
                Register::S0,
                Register::T0,
                (target_base_to_s0 + offset) as u64,
            ));
        }
        ir::Dtype::Float { .. } => {
            res.extend(mk_itype(
                asm::IType::load(dtype.clone()),
                Register::FT0,
                src_location,
                offset as u64,
            ));
            res.push(asm::Instruction::SType {
                instr: SType::store(dtype.clone()),
                rs1: Register::S0,
                rs2: Register::FT0,
                imm: asm::Immediate::Value((target_base_to_s0 + offset) as u64),
            });
        }
        ir::Dtype::Array { inner, size } => {
            let (size_of_inner_type, _) = inner.size_align_of(&source.structs).unwrap();
            for i in 0..*size {
                cp_from_indirect_source(
                    src_location,
                    target_base_to_s0,
                    offset + <usize as TryInto<i64>>::try_into(i * size_of_inner_type).unwrap(),
                    *inner.clone(),
                    res,
                    source,
                );
            }
        }
        ir::Dtype::Struct {
            fields,
            size_align_offsets,
            name,
            ..
        } => {
            let Some((_, _, offsets)) = (if size_align_offsets.is_some() {
                size_align_offsets.clone()
            } else {
                source
                    .structs
                    .get(name.as_ref().unwrap())
                    .and_then(|x| x.as_ref())
                    .and_then(|x| x.get_struct_size_align_offsets())
                    .and_then(|x| x.as_ref()).cloned()
            } ) else {unreachable!()};
            let Some(fields) = (if fields.is_some() {
                fields.clone()
            } else {
                source
                    .structs
                    .get(name.as_ref().unwrap())
                    .and_then(|x| x.as_ref())
                    .and_then(|x| x.get_struct_fields())
                    .and_then(|x| x.as_ref()).cloned()
            } ) else {unreachable!()};

            for (dtype, field_offset) in izip!(fields, offsets) {
                cp_from_indirect_source(
                    src_location,
                    target_base_to_s0,
                    offset + <usize as TryInto<i64>>::try_into(field_offset).unwrap(),
                    dtype.into_inner(),
                    res,
                    source,
                );
            }
        }
        _ => unreachable!(),
    }
}

fn cp_to_indirect_target(
    (source_reg, source_base): (Register, i64), // direct
    dest_location: Register,
    offset: i64,
    dtype: ir::Dtype,
    res: &mut Vec<asm::Instruction>,
    source: &ir::TranslationUnit,
) {
    assert!(offset >= 0);
    match &dtype {
        ir::Dtype::Pointer { .. } | ir::Dtype::Int { .. } => {
            res.extend(mk_itype(
                IType::load(dtype.clone()),
                Register::T0,
                source_reg,
                (source_base + offset) as u64,
            ));
            res.extend(mk_stype(
                SType::store(dtype),
                dest_location,
                Register::T0,
                offset as u64,
            ));
        }
        ir::Dtype::Float { .. } => {
            res.extend(mk_itype(
                IType::load(dtype.clone()),
                Register::FT0,
                source_reg,
                (source_base + offset) as u64,
            ));
            res.extend(mk_stype(
                SType::store(dtype),
                dest_location,
                Register::FT0,
                offset as u64,
            ));
        }
        ir::Dtype::Array { inner, size } => {
            let (size_of_inner_type, _) = inner.size_align_of(&source.structs).unwrap();
            for i in 0..*size {
                cp_to_indirect_target(
                    (source_reg, source_base),
                    dest_location,
                    offset + <usize as TryInto<i64>>::try_into(i * size_of_inner_type).unwrap(),
                    (**inner).clone(),
                    res,
                    source,
                );
            }
        }
        ir::Dtype::Struct {
            fields,
            size_align_offsets,
            name,
            ..
        } => {
            let Some((_, _, offsets)) = (if size_align_offsets.is_some() {
                size_align_offsets.clone()
            } else {
                source
                    .structs
                    .get(name.as_ref().unwrap())
                    .and_then(|x| x.as_ref())
                    .and_then(|x| x.get_struct_size_align_offsets())
                    .and_then(|x| x.as_ref()).cloned()
            } ) else {unreachable!()};
            let Some(fields) = (if fields.is_some() {
                fields.clone()
            } else {
                source
                    .structs
                    .get(name.as_ref().unwrap())
                    .and_then(|x| x.as_ref())
                    .and_then(|x| x.get_struct_fields())
                    .and_then(|x| x.as_ref()).cloned()
            } ) else {unreachable!()};

            for (dtype, field_offset) in izip!(fields, offsets) {
                cp_to_indirect_target(
                    (source_reg, source_base),
                    dest_location,
                    offset + <usize as TryInto<i64>>::try_into(field_offset).unwrap(),
                    dtype.into_inner(),
                    res,
                    source,
                );
            }
        }
        _ => unreachable!(),
    }
}

fn cp_from_indirect_to_indirect(
    source_location: Register,
    dest_location: Register,
    offset: i64,
    dtype: ir::Dtype,
    res: &mut Vec<asm::Instruction>,
    source: &ir::TranslationUnit,
) {
    assert!(offset >= 0);
    match &dtype {
        ir::Dtype::Pointer { .. } | ir::Dtype::Int { .. } => {
            res.extend(mk_itype(
                IType::load(dtype.clone()),
                Register::T0,
                source_location,
                offset as u64,
            ));
            res.extend(mk_stype(
                SType::store(dtype),
                dest_location,
                Register::T0,
                offset as u64,
            ));
        }
        ir::Dtype::Float { .. } => {
            res.extend(mk_itype(
                IType::load(dtype.clone()),
                Register::FT0,
                source_location,
                offset as u64,
            ));
            res.extend(mk_stype(
                SType::store(dtype),
                dest_location,
                Register::FT0,
                offset as u64,
            ));
        }
        ir::Dtype::Array { inner, size } => {
            let (size_of_inner_type, _) = inner.size_align_of(&source.structs).unwrap();
            for i in 0..*size {
                cp_from_indirect_to_indirect(
                    source_location,
                    dest_location,
                    offset + <usize as TryInto<i64>>::try_into(i * size_of_inner_type).unwrap(),
                    (**inner).clone(),
                    res,
                    source,
                );
            }
        }
        ir::Dtype::Struct {
            fields,
            size_align_offsets,
            name,
            ..
        } => {
            let Some((_, _, offsets)) = (if size_align_offsets.is_some() {
                size_align_offsets.clone()
            } else {
                source
                    .structs
                    .get(name.as_ref().unwrap())
                    .and_then(|x| x.as_ref())
                    .and_then(|x| x.get_struct_size_align_offsets())
                    .and_then(|x| x.as_ref()).cloned()
            } ) else {unreachable!()};
            let Some(fields) = (if fields.is_some() {
                fields.clone()
            } else {
                source
                    .structs
                    .get(name.as_ref().unwrap())
                    .and_then(|x| x.as_ref())
                    .and_then(|x| x.get_struct_fields())
                    .and_then(|x| x.as_ref()).cloned()
            } ) else {unreachable!()};

            for (dtype, field_offset) in izip!(fields, offsets) {
                cp_from_indirect_to_indirect(
                    source_location,
                    dest_location,
                    offset + <usize as TryInto<i64>>::try_into(field_offset).unwrap(),
                    dtype.into_inner(),
                    res,
                    source,
                );
            }
        }
        _ => unreachable!(),
    }
}

fn translate_const_expression(e: &Expression) -> Result<Value, ()> {
    match e {
        Expression::Constant(c) => {
            let c: ir::Constant = ir::Constant::try_from(&c.node).unwrap();
            Value::try_from(c)
        }
        Expression::UnaryOperator(u) => {
            let operand = translate_const_expression(&u.node.operand.node)?;
            crate::ir::interp::calculator::calculate_unary_operator_expression(
                &u.node.operator.node,
                operand,
            )
        }
        Expression::BinaryOperator(b) => {
            let lhs = translate_const_expression(&b.node.lhs.node)?;
            let rhs = translate_const_expression(&b.node.rhs.node)?;
            crate::ir::interp::calculator::calculate_binary_operator_expression(
                &b.node.operator.node,
                lhs,
                rhs,
            )
        }
        _ => Err(()),
    }
}

fn translate_const_expression_2_type(e: &Expression, dtype: ir::Dtype) -> Result<Value, ()> {
    let value = translate_const_expression(e)?;
    crate::ir::interp::calculator::calculate_typecast(value, dtype)
}

fn translate_int_float(initializer: &Option<Initializer>, dtype: ir::Dtype) -> Directive {
    let value: Value = match initializer {
        Some(Initializer::Expression(e)) => {
            translate_const_expression_2_type(&e.node, dtype.clone()).unwrap()
        }
        None => Value::default_from_dtype(&dtype, &HashMap::new()).unwrap(),
        _ => panic!(),
    };
    match (&dtype, value) {
        (ir::Dtype::Int { .. }, Value::Int { value, .. }) => {
            Directive::try_from_data_size(DataSize::try_from(dtype).unwrap(), value as u64)
        }
        (ir::Dtype::Float { width: 64, .. }, Value::Float { value, .. }) => {
            Directive::try_from_data_size(
                DataSize::try_from(dtype).unwrap(),
                value.into_inner().to_bits(),
            )
        }
        (ir::Dtype::Float { width: 32, .. }, Value::Float { value, .. }) => {
            let f: f32 = value.into_inner() as f32;
            Directive::Word(f.to_bits())
        }
        _ => unreachable!(),
    }
}

fn initializer_2_directive(
    dtype: ir::Dtype,
    initializer: Option<Initializer>,
    source: &ir::TranslationUnit,
) -> Vec<Directive> {
    match &dtype {
        ir::Dtype::Int { .. } | ir::Dtype::Float { .. } => {
            vec![translate_int_float(&initializer, dtype)]
        }
        ir::Dtype::Array { inner, size } => {
            let (size_of_inner, _) = inner.size_align_of(&source.structs).unwrap();
            match initializer {
                Some(Initializer::List(initializer)) => {
                    let mut v = vec![];
                    for x in &initializer {
                        let y = x.node.initializer.node.clone();
                        v.extend(initializer_2_directive(*inner.clone(), Some(y), source));
                    }
                    let b = size - initializer.len();
                    if b > 0 {
                        v.push(Directive::Zero(size_of_inner * b));
                    }
                    v
                }
                None => {
                    vec![Directive::Zero(size_of_inner * size)]
                }
                _ => unreachable!(),
            }
        }
        ir::Dtype::Struct {
            fields,
            size_align_offsets,
            ..
        } => {
            let Some((size, _, offsets)) = size_align_offsets else {unreachable!()};
            if initializer.is_none() {
                return vec![Directive::Zero(*size)];
            }

            let Some(Initializer::List(l)) =  initializer else {unreachable!()};
            let Some(fields) = fields else {unreachable!()};

            let mut end = 0;
            let mut v = vec![];
            for (field_dtype, initializer, offset) in izip!(fields, l, offsets) {
                if offset - end > 0 {
                    v.push(Directive::Zero(offset - end));
                }
                v.extend(initializer_2_directive(
                    field_dtype.clone().into_inner(),
                    Some(initializer.node.initializer.node),
                    source,
                ));
                let (size, _) = field_dtype.size_align_of(&source.structs).unwrap();
                end += size;
            }
            v
        }
        ir::Dtype::Typedef { .. } => unreachable!(),
        ir::Dtype::Function { .. } => unreachable!(),
        ir::Dtype::Pointer { .. } => unreachable!(),
        ir::Dtype::Unit { .. } => unreachable!(),
    }
}

#[derive(Debug, Clone, Copy)]
enum RegisterCouple {
    Single(Register),
    Double(Register),
    MergedToPrevious,
}

#[derive(Debug, Clone, Copy)]
enum RegOrStack {
    /// to be allocated
    IntRegNotSure,
    FloatRegNotSure,
    Reg(Register),
    Stack {
        offset_to_s0: i64,
    },
}

#[derive(Debug, Clone)]
enum ParamAlloc {
    PrimitiveType(DirectOrInDirect<RegOrStack>),
    StructInRegister(Vec<RegisterCouple>),
}

#[derive(Debug, Clone, Copy)]
enum RetLocation {
    OnStack,
    /// in A0 or FA0
    InRegister,
}

#[derive(Debug, Clone, Copy)]
enum DirectOrInDirect<T: Clone + Copy> {
    Direct(T),
    InDirect(T),
}

#[derive(Clone)]
struct FunctionAbi {
    params_alloc: Vec<ParamAlloc>,
    /// if on stack, offset must be zero
    ret_alloc: RetLocation,
    /// contain the ret_alloc
    caller_alloc: usize,
}

impl FunctionSignature {
    fn try_alloc(&self, source: &ir::TranslationUnit) -> FunctionAbi {
        let mut params: Vec<ParamAlloc> =
            vec![
                ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(Register::A0)));
                self.params.len()
            ];

        let mut next_int_reg: usize = 0;
        let mut next_float_reg: usize = 0;
        let mut caller_alloc: usize = 0;

        for (i, param) in self.params.iter().enumerate() {
            let (size, align) = param.size_align_of(&source.structs).unwrap();
            let align = align.max(4);
            match param {
                ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => {
                    if next_int_reg > 7 {
                        while caller_alloc % align != 0 {
                            caller_alloc += 1;
                        }
                        caller_alloc += size;
                        params[i] = ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(
                            RegOrStack::Stack {
                                offset_to_s0: caller_alloc.try_into().unwrap(),
                            },
                        ));
                    } else {
                        params[i] =
                            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(
                                Register::arg(asm::RegisterType::Integer, next_int_reg),
                            )));
                        next_int_reg += 1;
                    }
                }
                ir::Dtype::Float { .. } => {
                    if next_float_reg > 7 {
                        while caller_alloc % align != 0 {
                            caller_alloc += 1;
                        }
                        caller_alloc += size;
                        params[i] = ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(
                            RegOrStack::Stack {
                                offset_to_s0: caller_alloc.try_into().unwrap(),
                            },
                        ));
                    } else {
                        params[i] =
                            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(
                                Register::arg(asm::RegisterType::FloatingPoint, next_float_reg),
                            )));
                        next_float_reg += 1;
                    }
                }
                ir::Dtype::Struct {
                    name,
                    fields,
                    size_align_offsets,
                    ..
                } => {
                    if size <= 2 * 8 {
                        let Some((_, _, offsets)) = (if size_align_offsets.is_some() {
                            size_align_offsets.clone()
                        } else {
                            source
                                .structs
                                .get(name.as_ref().unwrap())
                                .and_then(|x| x.as_ref())
                                .and_then(|x| x.get_struct_size_align_offsets())
                                .and_then(|x| x.as_ref()).cloned()
                        } ) else {unreachable!()};

                        let Some(fields) = (if fields.is_some() {
                            fields.clone()
                        } else {
                            source
                                .structs
                                .get(name.as_ref().unwrap())
                                .and_then(|x| x.as_ref())
                                .and_then(|x| x.get_struct_fields())
                                .and_then(|x| x.as_ref()).cloned()
                        } ) else {unreachable!()};

                        let mut j = 0;
                        let mut x: Vec<RegisterCouple> =
                            vec![RegisterCouple::MergedToPrevious; offsets.len()];
                        while j < offsets.len() {
                            match &*fields[j] {
                                ir::Dtype::Int { width: 32, .. } => {
                                    if j < offsets.len() - 1 {
                                        // check the next one
                                        match &*fields[j + 1] {
                                            ir::Dtype::Int { width: 32, .. }
                                            | ir::Dtype::Float { width: 32, .. } => {
                                                x[j] = RegisterCouple::Double(Register::arg(
                                                    asm::RegisterType::Integer,
                                                    next_int_reg,
                                                ));
                                                next_int_reg += 1;
                                                x[j + 1] = RegisterCouple::MergedToPrevious;
                                                j += 2;
                                            }
                                            ir::Dtype::Int { width: 64, .. }
                                            | ir::Dtype::Float { width: 64, .. } => {}
                                            _ => unreachable!(),
                                        }
                                    } else {
                                        x[j] = RegisterCouple::Single(Register::arg(
                                            asm::RegisterType::Integer,
                                            next_int_reg,
                                        ));
                                        next_int_reg += 1;
                                        j += 1;
                                    }
                                }
                                ir::Dtype::Int { width: 64, .. } => {
                                    x[j] = RegisterCouple::Single(Register::arg(
                                        asm::RegisterType::Integer,
                                        next_int_reg,
                                    ));
                                    next_int_reg += 1;
                                    j += 1;
                                }
                                ir::Dtype::Float { width: 32, .. } => {
                                    if j == offsets.len() - 1 {
                                        x[j] = RegisterCouple::Single(Register::arg(
                                            asm::RegisterType::FloatingPoint,
                                            next_float_reg,
                                        ));
                                        next_float_reg += 1;
                                        j += 1;
                                    } else {
                                        match &*fields[j + 1] {
                                            ir::Dtype::Int { width: 32, .. } => {
                                                x[j] = RegisterCouple::Double(Register::arg(
                                                    asm::RegisterType::Integer,
                                                    next_int_reg,
                                                ));
                                                next_int_reg += 1;
                                                x[j + 1] = RegisterCouple::MergedToPrevious;
                                                j += 2;
                                            }
                                            ir::Dtype::Float { width: 32, .. } => {
                                                x[j] = RegisterCouple::Single(Register::arg(
                                                    asm::RegisterType::FloatingPoint,
                                                    next_float_reg,
                                                ));
                                                next_float_reg += 1;
                                                x[j + 1] = RegisterCouple::Single(Register::arg(
                                                    asm::RegisterType::FloatingPoint,
                                                    next_float_reg,
                                                ));
                                                next_float_reg += 1;
                                                j += 2;
                                            }
                                            ir::Dtype::Int { width: 64, .. }
                                            | ir::Dtype::Float { width: 64, .. } => {
                                                x[j] = RegisterCouple::Single(Register::arg(
                                                    asm::RegisterType::FloatingPoint,
                                                    next_float_reg,
                                                ));
                                                next_float_reg += 1;
                                                j += 1;
                                            }
                                            _ => unreachable!(),
                                        }
                                    }
                                }
                                ir::Dtype::Float { width: 64, .. } => {
                                    x[j] = RegisterCouple::Single(Register::arg(
                                        asm::RegisterType::FloatingPoint,
                                        next_float_reg,
                                    ));
                                    next_float_reg += 1;
                                    j += 1;
                                }
                                _ => unreachable!(),
                            }
                        }
                        params[i] = ParamAlloc::StructInRegister(x);
                    } else if next_int_reg > 7 {
                        while caller_alloc % 8 != 0 {
                            caller_alloc += 1;
                        }
                        caller_alloc += 8;
                        params[i] = ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(
                            RegOrStack::Stack {
                                offset_to_s0: caller_alloc.try_into().unwrap(),
                            },
                        ));
                    } else {
                        params[i] =
                            ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Reg(
                                Register::arg(asm::RegisterType::Integer, next_int_reg),
                            )));
                        next_int_reg += 1;
                    }
                }
                ir::Dtype::Array { .. } => unimplemented!(),
                ir::Dtype::Function { .. } => unimplemented!(),
                ir::Dtype::Unit { .. } => unreachable!(),
                ir::Dtype::Typedef { .. } => unreachable!(),
            }
        }

        while caller_alloc % 16 != 0 {
            caller_alloc += 1;
        }

        let ret_alloc = match &self.ret {
            ir::Dtype::Array { .. } => unimplemented!(),
            ir::Dtype::Struct { .. } => {
                caller_alloc += 16;
                RetLocation::OnStack
            }
            _ => RetLocation::InRegister,
        };

        let caller_alloc = caller_alloc;

        for x in params.iter_mut() {
            match x {
                ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Stack {
                    offset_to_s0,
                }))
                | ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Stack {
                    offset_to_s0,
                })) => {
                    *offset_to_s0 = (caller_alloc
                        - <i64 as TryInto<usize>>::try_into(*offset_to_s0).unwrap())
                    .try_into()
                    .unwrap();
                }
                _ => {}
            }
        }

        FunctionAbi {
            ret_alloc,
            params_alloc: params,
            caller_alloc,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
struct Float {
    value: OrderedFloat<f64>,
    width: usize,
}

impl Float {
    fn to_directive(self) -> Directive {
        match self.width {
            32 => {
                let f: f32 = self.value.into_inner() as f32;
                Directive::Word(f.to_bits())
            }
            64 => asm::Directive::Quad(self.value.into_inner().to_bits()),
            _ => unreachable!(),
        }
    }
}

struct FloatMp(HashMap<Float, usize>);

impl FloatMp {
    fn get_index(&mut self, f: Float) -> usize {
        let length = self.0.len();
        match self.0.entry(f) {
            std::collections::hash_map::Entry::Occupied(o) => *o.get(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let _ = v.insert(length);
                length
            }
        }
    }
    fn get_label(&mut self, f: Float) -> Label {
        let index = self.get_index(f);
        Label(format!(".LCPI1_{index}"))
    }
}

fn mk_itype(instr: IType, rd: Register, rs1: Register, imm: u64) -> Vec<asm::Instruction> {
    if (-2048..=2047).contains(&(imm as i64)) {
        vec![asm::Instruction::IType {
            instr,
            rd,
            rs1,
            imm: Immediate::Value(imm),
        }]
    } else {
        vec![
            asm::Instruction::Pseudo(Pseudo::Li {
                rd: Register::T4,
                imm: imm,
            }),
            asm::Instruction::RType {
                instr: RType::Add(DataSize::Double),
                rd: Register::T4,
                rs1,
                rs2: Some(Register::T4),
            },
            asm::Instruction::IType {
                instr,
                rd,
                rs1: Register::T4,
                imm: Immediate::Value(0),
            },
        ]
    }
}

fn mk_stype(instr: SType, rs1: Register, rs2: Register, imm: u64) -> Vec<asm::Instruction> {
    if (-2048..=2047).contains(&(imm as i64)) {
        vec![asm::Instruction::SType {
            instr,
            rs1,
            rs2,
            imm: Immediate::Value(imm),
        }]
    } else {
        vec![
            asm::Instruction::Pseudo(Pseudo::Li {
                rd: Register::T4,
                imm: imm,
            }),
            asm::Instruction::RType {
                instr: RType::Add(DataSize::Double),
                rd: Register::T4,
                rs1,
                rs2: Some(Register::T4),
            },
            asm::Instruction::SType {
                instr,
                rs1: Register::T4,
                rs2,
                imm: Immediate::Value(0),
            },
        ]
    }
}

// v : (origin, target)
fn cp_parallel(v: Vec<(Register, Register, ir::Dtype)>) -> Vec<asm::Instruction> {
    for (src, target, _) in &v {
        assert_ne!(src, target);
    }
    let sources: HashSet<Register> = v.iter().map(|(src, _, _)| *src).collect();
    let targets: HashSet<Register> = v.iter().map(|(_, dest, _)| *dest).collect();
    let registers: HashSet<Register> = sources.union(&targets).copied().collect();
    assert!(sources.len() <= targets.len());
    assert_eq!(targets.len(), v.len());

    let mut graph: StableGraph<(), ir::Dtype, petgraph::Directed> = Default::default();
    let mut register_2_node_index: HashMap<Register, NodeIndex> = HashMap::new();
    let mut node_index_2_register: HashMap<NodeIndex, Register> = HashMap::new();

    for register in registers {
        let node_index = graph.add_node(());
        let None = register_2_node_index.insert(register, node_index) else {unreachable!()};
        let None = node_index_2_register.insert(node_index, register) else {unreachable!()};
    }

    let mut edges = Vec::new();
    for (src, target, dtype) in &v {
        let edge = graph.add_edge(
            *register_2_node_index.get(src).unwrap(),
            *register_2_node_index.get(target).unwrap(),
            dtype.clone(),
        );
        edges.push(edge);
    }

    let loops: Vec<Vec<_>> = petgraph::algo::tarjan_scc(&graph)
        .into_iter()
        .filter(|x| x.len() > 1)
        .collect();
    let nodes_in_loop: HashSet<NodeIndex> = loops.iter().flatten().copied().collect();

    let mut instructions: Vec<asm::Instruction> = Vec::new();
    for (src, target, dtype) in &v {
        match (
            nodes_in_loop.get(register_2_node_index.get(src).unwrap()),
            nodes_in_loop.get(register_2_node_index.get(target).unwrap()),
        ) {
            (None, None) | (Some(_), None) => {
                instructions.extend(mv_register(*src, *target, dtype.clone()));
            }
            (Some(_), Some(_)) => {
                // deal with this in loop
            }
            (None, Some(_)) => unreachable!(),
        }
    }
    for loop_in_graph in loops {
        instructions.extend(cp_parallel_inner(
            loop_in_graph,
            &graph,
            &node_index_2_register,
        ));
    }

    instructions
}

/// the loop_in_graph is not is order
fn cp_parallel_inner(
    loop_in_graph: Vec<NodeIndex>,
    graph: &StableGraph<(), ir::Dtype>,
    node_index_2_register: &HashMap<NodeIndex, Register>,
) -> Vec<asm::Instruction> {
    assert!(loop_in_graph.len() > 1);

    let mut instructions: Vec<asm::Instruction> = Vec::new();

    let loop_in_graph = order_loop(loop_in_graph, graph);

    let get_dtype = |src: NodeIndex, dest: NodeIndex| {
        let Some(e) = graph.find_edge(src, dest) else {unreachable!()};
        graph.edge_weight(e).unwrap()
    };

    // backup the last register
    let (backup_register, backup_dtype, backup_instructions) = backup_register(
        *node_index_2_register
            .get(loop_in_graph.last().unwrap())
            .unwrap(),
        get_dtype(*loop_in_graph.last().unwrap(), loop_in_graph[0]),
    );

    instructions.extend(backup_instructions);

    for i in (0..=loop_in_graph.len() - 2).rev() {
        let dtype = get_dtype(loop_in_graph[i], loop_in_graph[i + 1]);
        let x = mv_register(
            *node_index_2_register.get(&loop_in_graph[i]).unwrap(),
            *node_index_2_register.get(&loop_in_graph[i + 1]).unwrap(),
            dtype.clone(),
        );
        instructions.extend(x);
    }

    instructions.extend(mv_register(
        backup_register,
        *node_index_2_register.get(&loop_in_graph[0]).unwrap(),
        backup_dtype,
    ));

    instructions
}

fn order_loop(loop_in_graph: Vec<NodeIndex>, graph: &StableGraph<(), ir::Dtype>) -> Vec<NodeIndex> {
    let mut nodes_in_loop: HashSet<NodeIndex> = loop_in_graph.iter().copied().collect();
    let mut ordered_loop: Vec<NodeIndex> = vec![loop_in_graph[0]];
    let true = nodes_in_loop.remove(&loop_in_graph[0]) else {unreachable!()};

    for _ in 0..loop_in_graph.len() - 1 {
        let src = *ordered_loop.last().unwrap();
        let mut iter = graph
            .neighbors_directed(src, petgraph::Direction::Outgoing)
            .filter(|x| nodes_in_loop.contains(x));
        let dest: NodeIndex = iter.next().unwrap();
        assert!(iter.next().is_none());
        ordered_loop.push(dest);
        let true = nodes_in_loop.remove(&dest) else {unreachable!()};
    }

    ordered_loop
}

fn backup_register(
    src: Register,
    dtype: &ir::Dtype,
) -> (Register, ir::Dtype, Vec<asm::Instruction>) {
    match &dtype {
        ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => (
            Register::T0,
            dtype.clone(),
            mv_register(src, Register::T0, dtype.clone()),
        ),
        ir::Dtype::Float { .. } => (
            Register::FT0,
            dtype.clone(),
            mv_register(src, Register::FT0, dtype.clone()),
        ),
        _ => unreachable!(),
    }
}

fn mv_register(src: Register, target: Register, dtype: ir::Dtype) -> Vec<asm::Instruction> {
    match &dtype {
        ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => {
            vec![asm::Instruction::Pseudo(Pseudo::Mv {
                rd: target,
                rs: src,
            })]
        }
        ir::Dtype::Float { .. } => {
            vec![asm::Instruction::Pseudo(Pseudo::Fmv {
                rd: target,
                rs: src,
                data_size: DataSize::try_from(dtype).unwrap(),
            })]
        }
        _ => unreachable!(),
    }
}

/*
fn pre_order(dom_tree: &HashMap<BlockId, Vec<BlockId>>, bid_init: BlockId) -> Vec<BlockId> {
    let mut res = vec![bid_init];

    match dom_tree.get(&bid_init) {
        Some(v) => {
            for x in v {
                res.extend(pre_order(dom_tree, *x));
            }
        }
        None => {}
    }

    res
}

fn post_order(dom_tree: &HashMap<BlockId, Vec<BlockId>>, bid_init: BlockId) -> Vec<BlockId> {
    let mut res = vec![];

    if let Some(v) = dom_tree.get(&bid_init) {
        for x in v {
            res.extend(post_order(dom_tree, *x));
        }
    }

    res.push(bid_init);
    res
}
 */

#[derive(Debug, Default)]
struct LivenessRes {
    inn: HashMap<BlockId, HashSet<RegisterId>>,
    out: HashMap<BlockId, HashSet<RegisterId>>,
}

// https://groups.seas.harvard.edu/courses/cs153/2019fa/lectures/Lec20-Dataflow-analysis.pdf
fn gen_kill(
    def: &HashMap<BlockId, Vec<RegisterId>>,
    usee: &HashMap<BlockId, Vec<RegisterId>>,
    definition: &ir::FunctionDefinition,
) -> LivenessRes {
    let mut liveness_res = LivenessRes::default();

    let mut modified = true;
    while modified {
        modified = false;
        for (bid, block) in &definition.blocks {
            let old_out_len = liveness_res
                .out
                .get(bid)
                .map(|x| x.len())
                .unwrap_or_default();
            let old_in_len = liveness_res
                .inn
                .get(bid)
                .map(|x| x.len())
                .unwrap_or_default();

            let new_out: HashSet<RegisterId> = block.exit.walk_jump_bid().fold(
                HashSet::new(),
                |value, jump_bid| match liveness_res.inn.get(&jump_bid) {
                    Some(x) => value.union(x).copied().collect(),
                    None => value,
                },
            );

            match new_out.len().cmp(&old_out_len) {
                std::cmp::Ordering::Less => unreachable!(),
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => modified = true,
            }

            let new_in: HashSet<RegisterId> = new_out
                .iter()
                .filter(|&rid| !def.get(bid).unwrap().contains(rid))
                .chain(usee.get(bid).unwrap())
                .copied()
                .collect();

            match new_in.len().cmp(&old_in_len) {
                std::cmp::Ordering::Less => unreachable!(),
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => modified = true,
            }

            let _x = liveness_res.inn.insert(*bid, new_in);
            let _x = liveness_res.out.insert(*bid, new_out);
        }
    }

    liveness_res
}

impl DataSize {
    fn mask(self) -> u64 {
        match self {
            DataSize::Byte => 0xff,
            DataSize::Half => 0xffff,
            DataSize::Word => 0xffffffff,
            DataSize::Double => 0xffffffffffffffff,
            DataSize::SinglePrecision => unreachable!(),
            DataSize::DoublePrecision => unreachable!(),
        }
    }
}
