#![allow(clippy::too_many_arguments)]

mod mesh;

use crate::asm::{
    self, DataSize, Directive, IType, Immediate, Label, Pseudo, RType, Register, SType, Section,
    TranslationUnit,
};
use crate::ir::{
    self, BlockId, Declaration, FunctionDefinition, FunctionSignature, HasDtype, RegisterId, Value,
};
use crate::Translate;
use bimap::BiMap;
use itertools::{iproduct, izip};
use lang_c::ast::{BinaryOperator, Expression, Initializer, UnaryOperator};
use ordered_float::OrderedFloat;

use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::visit::IntoNodeIdentifiers;
use std::collections::{HashMap, HashSet, LinkedList};
use std::iter::once;
use std::ops::Deref;

static INT_OFFSETS: [(Register, i64); 12] = [
    (Register::S0, 16),
    (Register::S1, 24),
    (Register::S2, 32),
    (Register::S3, 40),
    (Register::S4, 48),
    (Register::S5, 56),
    (Register::S6, 64),
    (Register::S7, 72),
    (Register::S8, 80),
    (Register::S9, 88),
    (Register::S10, 96),
    (Register::S11, 104),
];

static FLOAT_OFFSETS: [(Register, i64); 12] = [
    (Register::FS0, 112),
    (Register::FS1, 120),
    (Register::FS2, 128),
    (Register::FS3, 136),
    (Register::FS4, 144),
    (Register::FS5, 152),
    (Register::FS6, 160),
    (Register::FS7, 168),
    (Register::FS8, 176),
    (Register::FS9, 184),
    (Register::FS10, 192),
    (Register::FS11, 200),
];

static INT_REGISTERS: [Register; 21] = [
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
    Register::T5,
    Register::T6,
    Register::A0,
    Register::A1,
    Register::A2,
    Register::A3,
    Register::A4,
    Register::A5,
    Register::A6,
    Register::A7,
];

static FLOAT_REGISTERS: [Register; 26] = [
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
    Register::FT2,
    Register::FT3,
    Register::FT4,
    Register::FT5,
    Register::FT6,
    Register::FT7,
    Register::FA0,
    Register::FA1,
    Register::FA2,
    Register::FA3,
    Register::FA4,
    Register::FA5,
    Register::FA6,
    Register::FA7,
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
            let Declaration::Variable { dtype, initializer } = decl else {
                continue;
            };
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
            let Declaration::Function {
                signature,
                definition,
            } = decl
            else {
                continue;
            };
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

        asm.unit.rm_needless_mv();
        Ok(asm)
    }
}

fn translate_function(
    func_name: &str,
    signature: &FunctionSignature,
    definition: &FunctionDefinition,
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
            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(reg))) => {
                match dtype {
                    ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => {
                        let None = register_mp.insert(
                            register_id,
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { src: Some(*reg) }),
                        ) else {
                            unreachable!()
                        };
                    }
                    ir::Dtype::Float { .. } => {
                        let None = register_mp.insert(
                            register_id,
                            DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure {
                                src: Some(*reg),
                            }),
                        ) else {
                            unreachable!()
                        };
                    }
                    _ => unreachable!(),
                }
            }
            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Stack {
                offset_to_s0,
            })) => {
                let None = register_mp.insert(
                    register_id,
                    DirectOrInDirect::Direct(RegOrStack::Stack {
                        offset_to_s0: *offset_to_s0,
                    }),
                ) else {
                    unreachable!()
                };
            }
            ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Reg(reg))) => {
                // must be a struct
                let None = register_mp.insert(
                    register_id,
                    DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure { src: Some(*reg) }),
                ) else {
                    unreachable!()
                };
            }
            ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Stack {
                offset_to_s0,
            })) => {
                let None = register_mp.insert(
                    register_id,
                    DirectOrInDirect::InDirect(RegOrStack::Stack {
                        offset_to_s0: *offset_to_s0,
                    }),
                ) else {
                    unreachable!()
                };
            }
            ParamAlloc::StructInRegister(v) => {
                let ir::Dtype::Struct {
                    name,
                    size_align_offsets,
                    fields,
                    ..
                } = dtype
                else {
                    unreachable!()
                };
                let Some((size, align, offsets)) = (if size_align_offsets.is_some() {
                    size_align_offsets.clone()
                } else {
                    source
                        .structs
                        .get(name.as_ref().unwrap())
                        .and_then(|x| x.as_ref())
                        .and_then(|x| x.get_struct_size_align_offsets())
                        .and_then(|x| x.as_ref())
                        .cloned()
                }) else {
                    unreachable!()
                };

                let Some(fields) = (if fields.is_some() {
                    fields.clone()
                } else {
                    source
                        .structs
                        .get(name.as_ref().unwrap())
                        .and_then(|x| x.as_ref())
                        .and_then(|x| x.get_struct_fields())
                        .and_then(|x| x.as_ref())
                        .cloned()
                }) else {
                    unreachable!()
                };

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
                let None = register_mp.insert(
                    register_id,
                    DirectOrInDirect::Direct(RegOrStack::Stack {
                        offset_to_s0: stack_offset_2_s0,
                    }),
                ) else {
                    unreachable!()
                };
            }
            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::IntRegNotSure {
                ..
            }))
            | ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure {
                ..
            }))
            | ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure {
                ..
            }))
            | ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(
                RegOrStack::FloatRegNotSure { .. },
            )) => {
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
        let None = register_mp.insert(
            RegisterId::Local { aid },
            DirectOrInDirect::Direct(RegOrStack::Stack {
                offset_to_s0: stack_offset_2_s0,
            }),
        ) else {
            unreachable!()
        };
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
                    let None = register_mp.insert(
                        RegisterId::Arg { bid, aid },
                        DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { src: None }),
                    ) else {
                        unreachable!()
                    };
                }
                ir::Dtype::Float { .. } => {
                    let None = register_mp.insert(
                        RegisterId::Arg { bid, aid },
                        DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { src: None }),
                    ) else {
                        unreachable!()
                    };
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
                    let None = register_mp.insert(
                        RegisterId::Temp { bid, iid },
                        DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { src: None }),
                    ) else {
                        unreachable!()
                    };
                }
                ir::Dtype::Float { .. } => {
                    let None = register_mp.insert(
                        RegisterId::Temp { bid, iid },
                        DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { src: None }),
                    ) else {
                        unreachable!()
                    };
                }
                ir::Dtype::Array { .. } => unreachable!(),
                ir::Dtype::Struct { .. } => {
                    let (size, align) = dtype.size_align_of(&source.structs).unwrap();
                    let align: i64 = align.max(4).try_into().unwrap();
                    while stack_offset_2_s0 % align != 0 {
                        stack_offset_2_s0 -= 1;
                    }
                    stack_offset_2_s0 -= size.max(4) as i64;
                    let None = register_mp.insert(
                        RegisterId::Temp { bid, iid },
                        DirectOrInDirect::Direct(RegOrStack::Stack {
                            offset_to_s0: stack_offset_2_s0,
                        }),
                    ) else {
                        unreachable!()
                    };
                }
                ir::Dtype::Function { .. } => unreachable!(),
                ir::Dtype::Typedef { .. } => unreachable!(),
            }
        }
    }

    // before gen detailed asm::Instruction
    // we need to allocate register first
    alloc_register(definition, abi, &mut register_mp, &mut stack_offset_2_s0);

    #[allow(clippy::all)]
    for (_, v) in &register_mp {
        match v {
            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. })
            | DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure { .. })
            | DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure { .. }) => unreachable!(),
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
            float_mp,
        );
        function.blocks.push(asm::Block {
            label: Some(Label::new(func_name, bid)),
            instructions,
        });
    }

    function.blocks.extend(temp_block);

    let used_registers: HashSet<Register> = function
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .flat_map(|instruction| instruction.walk_register())
        .chain(once(Register::S0)) // always backup S0
        .collect();
    let is_used = |x: &&(Register, i64)| -> bool { used_registers.contains(&x.0) };

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

    let backup_sx: Vec<crate::asm::Instruction> = INT_OFFSETS
        .iter()
        .filter(is_used)
        .flat_map(|(rs2, offset)| {
            mk_stype(
                asm::SType::SD,
                Register::Sp,
                *rs2,
                (-stack_offset_2_s0 - *offset) as u64,
            )
        })
        .chain(
            FLOAT_OFFSETS
                .iter()
                .filter(is_used)
                .flat_map(|(rs2, offset)| {
                    mk_stype(
                        asm::SType::Store(DataSize::DoublePrecision),
                        Register::Sp,
                        *rs2,
                        (-stack_offset_2_s0 - *offset) as u64,
                    )
                }),
        )
        .collect();

    let restore_sx: Vec<crate::asm::Instruction> = INT_OFFSETS
        .iter()
        .filter(is_used)
        .flat_map(|(rd, offset)| {
            mk_itype(
                asm::IType::LD,
                *rd,
                Register::Sp,
                (-stack_offset_2_s0 - *offset) as u64,
            )
        })
        .chain(
            FLOAT_OFFSETS
                .iter()
                .filter(is_used)
                .flat_map(|(rd, offset)| {
                    mk_itype(
                        asm::IType::Load {
                            data_size: DataSize::DoublePrecision,
                            is_signed: true,
                        },
                        *rd,
                        Register::Sp,
                        (-stack_offset_2_s0 - *offset) as u64,
                    )
                }),
        )
        .collect();

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

    for b in function.blocks.iter_mut() {
        match b.instructions.last().unwrap() {
            asm::Instruction::Pseudo(Pseudo::Ret) => {
                let Some(asm::Instruction::Pseudo(Pseudo::Ret)) = b.instructions.pop() else {
                    unreachable!()
                };
                b.instructions.extend(before_ret_instructions.clone());
            }
            _ => {}
        }
    }

    let init_block = function.blocks.first_mut().unwrap();
    assert_eq!(
        init_block.label,
        Some(Label::new(func_name, definition.bid_init))
    );
    backup_ra_and_init_sp.extend(std::mem::replace(&mut init_block.instructions, Vec::new()));
    *init_block = asm::Block {
        label: Some(Label(func_name.to_owned())),
        instructions: backup_ra_and_init_sp,
    };

    function
}

fn alloc_register(
    definition: &FunctionDefinition,
    abi: &FunctionAbi,
    register_mp: &mut HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
    stack_offset_2_s0: &mut i64,
) {
    let reserved_rids = definition.get_rid_in_loop();

    let mut int_ig = int_interference_graph(definition, abi, register_mp);
    spills(
        &mut int_ig,
        &INT_REGISTERS,
        stack_offset_2_s0,
        &reserved_rids,
        register_mp,
    );
    color(&mut int_ig, &INT_REGISTERS, register_mp);
    int_ig.dump_register_mp(register_mp);

    let mut float_ig = float_interference_graph(definition, abi, register_mp);
    spills(
        &mut float_ig,
        &FLOAT_REGISTERS,
        stack_offset_2_s0,
        &reserved_rids,
        register_mp,
    );
    color(&mut float_ig, &FLOAT_REGISTERS, register_mp);
    float_ig.dump_register_mp(register_mp);
}

fn color(
    ig: &mut GraphWrapper,
    colors: &[Register],
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
) {
    let mut used_color: HashSet<Register> = HashSet::new();
    for node_index in lbfs(&ig.graph).into_iter() {
        if let Some(Some(color)) = ig.graph.node_weight(node_index) {
            // already colored
            let _ = used_color.insert(*color);
            continue;
        }
        let conflict_colors: HashSet<Register> = ig
            .graph
            .neighbors(node_index)
            .map(|n| ig.graph.node_weight(n).unwrap())
            .flat_map(|x| *x)
            .collect();

        let Some(x) = ig.graph.node_weight_mut(node_index) else {
            unreachable!()
        };
        assert_eq!(*x, None);

        let mut available_color_iter = used_color
            .iter()
            .chain(colors.iter())
            .filter(|c| !conflict_colors.contains(c))
            .copied();

        let color = match register_mp
            .get(
                ig.register_id_2_node_index
                    .get_by_right(&node_index)
                    .unwrap(),
            )
            .unwrap()
        {
            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { src: Some(src) })
            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { src: Some(src) })
            | DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure { src: Some(src) })
            | DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure { src: Some(src) }) => {
                // don't use hashset !
                if available_color_iter.clone().any(|x| x == *src) {
                    *src
                } else {
                    available_color_iter.next().unwrap()
                }
            }
            _ => available_color_iter.next().unwrap(),
        };
        *x = Some(color);
        let _ = used_color.insert(color);
    }
}

#[derive(Debug, Default)]
struct GraphWrapper {
    graph: StableGraph<Option<Register>, (), petgraph::Undirected>,
    register_id_2_node_index: BiMap<RegisterId, NodeIndex>,
}

impl GraphWrapper {
    fn add_node(&mut self, register_id: RegisterId) {
        let node = self.graph.add_node(None);
        let Ok(()) = self
            .register_id_2_node_index
            .insert_no_overwrite(register_id, node)
        else {
            unreachable!()
        };
    }

    fn add_edge(&mut self, a: RegisterId, b: RegisterId) {
        if a == b {
            return;
        }
        let Some(&a) = self.register_id_2_node_index.get_by_left(&a) else {
            return;
        };
        let Some(&b) = self.register_id_2_node_index.get_by_left(&b) else {
            return;
        };
        let _ = self.graph.update_edge(a, b, ());
    }

    fn dump_register_mp(
        &self,
        register_mp: &mut HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
    ) {
        for node_index in self.graph.node_identifiers() {
            let Some(register_id) = self.register_id_2_node_index.get_by_right(&node_index) else {
                // precolored
                continue;
            };
            let Some(Some(color)) = self.graph.node_weight(node_index) else {
                unreachable!()
            };
            match register_mp.get(register_id).unwrap() {
                DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. }) => {
                    let Some(DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })) =
                        register_mp.insert(
                            *register_id,
                            DirectOrInDirect::Direct(RegOrStack::Reg(*color)),
                        )
                    else {
                        unreachable!()
                    };
                }
                DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                    let Some(DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. })) =
                        register_mp.insert(
                            *register_id,
                            DirectOrInDirect::Direct(RegOrStack::Reg(*color)),
                        )
                    else {
                        unreachable!()
                    };
                }
                DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure { .. }) => {
                    let Some(DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure { .. })) =
                        register_mp.insert(
                            *register_id,
                            DirectOrInDirect::InDirect(RegOrStack::Reg(*color)),
                        )
                    else {
                        unreachable!()
                    };
                }
                DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure { .. }) => {
                    let Some(DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure { .. })) =
                        register_mp.insert(
                            *register_id,
                            DirectOrInDirect::InDirect(RegOrStack::Reg(*color)),
                        )
                    else {
                        unreachable!()
                    };
                }
                _ => unreachable!(),
            }
        }
    }
}

fn int_inter_block_liveness_graph(
    definition: &FunctionDefinition,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
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
        let None = def.insert(bid, v) else {
            unreachable!()
        };
    }

    let mut usee: HashMap<BlockId, Vec<RegisterId>> = HashMap::new();
    for (&curr_bid, block) in &definition.blocks {
        let v: Vec<RegisterId> = block
            .walk_register()
            .filter_map(|(rid, _)| {
                if let RegisterId::Arg { bid, .. } | RegisterId::Temp { bid, .. } = rid
                    && bid == curr_bid
                {
                    None
                } else {
                    match register_mp.get(&rid) {
                        Some(DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. }))
                        | Some(DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure { .. })) => {
                            Some(rid)
                        }
                        _ => None,
                    }
                }
            })
            .collect();
        let None = usee.insert(curr_bid, v) else {
            unreachable!()
        };
    }
    gen_kill(&def, &usee, definition)
}

fn float_inter_block_liveness_graph(
    definition: &FunctionDefinition,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
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
        let None = def.insert(bid, v) else {
            unreachable!()
        };
    }

    let mut usee: HashMap<BlockId, Vec<RegisterId>> = HashMap::new();
    for (&curr_bid, block) in &definition.blocks {
        let v: Vec<RegisterId> = block
            .walk_register()
            .filter_map(|(rid, _)| {
                if let RegisterId::Arg { bid, .. } | RegisterId::Temp { bid, .. } = rid
                    && bid == curr_bid
                {
                    None
                } else {
                    match register_mp.get(&rid) {
                        Some(DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }))
                        | Some(DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure {
                            ..
                        })) => Some(rid),
                        _ => None,
                    }
                }
            })
            .collect();
        let None = usee.insert(curr_bid, v) else {
            unreachable!()
        };
    }
    gen_kill(&def, &usee, definition)
}

fn int_interference_graph(
    definition: &FunctionDefinition,
    abi: &FunctionAbi,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
) -> GraphWrapper {
    let mut int_ig: GraphWrapper = GraphWrapper::default();
    for (rid, v) in register_mp {
        match v {
            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
            | DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure { .. }) => {
                int_ig.add_node(*rid);
            }
            _ => {}
        }
    }
    let caller_saved_register = [
        int_ig.graph.add_node(Some(Register::A0)),
        int_ig.graph.add_node(Some(Register::A1)),
        int_ig.graph.add_node(Some(Register::A2)),
        int_ig.graph.add_node(Some(Register::A3)),
        int_ig.graph.add_node(Some(Register::A4)),
        int_ig.graph.add_node(Some(Register::A5)),
        int_ig.graph.add_node(Some(Register::A6)),
        int_ig.graph.add_node(Some(Register::A7)),
        int_ig.graph.add_node(Some(Register::T5)),
        int_ig.graph.add_node(Some(Register::T6)),
    ];

    for (a, b) in iproduct!(caller_saved_register, caller_saved_register) {
        if a != b {
            let _ = int_ig.graph.add_edge(a, b, ());
        }
    }

    for aid_1 in 0..definition.allocations.len() {
        for aid_2 in 0..definition.allocations.len() {
            int_ig.add_edge(
                RegisterId::Local { aid: aid_1 },
                RegisterId::Local { aid: aid_2 },
            );
        }
    }

    for aid in 0..definition.allocations.len() {
        for (x, _) in definition.get_init_block().phinodes.iter().enumerate() {
            int_ig.add_edge(
                RegisterId::Local { aid },
                RegisterId::Arg {
                    bid: definition.bid_init,
                    aid: x,
                },
            );
        }
    }

    let liveness_sets = int_inter_block_liveness_graph(definition, register_mp).out;

    for (bid, mut live_set) in liveness_sets {
        let block = definition.blocks.get(&bid).unwrap();

        // first analyze block.exit
        for (rid, _) in block.exit.walk_register() {
            let _ = live_set.insert(rid);
        }
        for (rid, _) in block.exit.walk_register() {
            for a in &live_set {
                int_ig.add_edge(*a, rid);
            }
        }

        // then visit instruction in reverse order
        for (iid, instr) in block.instructions.iter().enumerate().rev() {
            let _ = live_set.remove(&RegisterId::Temp { bid, iid });

            for &a in &live_set {
                int_ig.add_edge(a, RegisterId::Temp { bid, iid });
            }

            if let ir::Instruction::Call { callee, .. } = &**instr {
                // live set conflict with caller_saved_register
                // callee conflict with caller_saved_register
                for a in live_set
                    .iter()
                    .chain(once(callee).filter_map(|operand| match operand {
                        ir::Operand::Constant(_) => None,
                        ir::Operand::Register { rid, .. } => Some(rid),
                    }))
                {
                    let Some(&a) = int_ig.register_id_2_node_index.get_by_left(a) else {
                        continue;
                    };
                    for &b in &caller_saved_register {
                        let _ = int_ig.graph.add_edge(a, b, ());
                    }
                }
            }

            for (rid, _) in instr.walk_register() {
                let _ = live_set.insert(rid);
            }

            for (rid_1, _) in instr.walk_register() {
                for &a in &live_set {
                    int_ig.add_edge(a, rid_1);
                }
            }
        }

        for (aid, _) in block.phinodes.iter().enumerate() {
            let _ = live_set.insert(RegisterId::Arg { bid, aid });
        }

        for (aid, _) in block.phinodes.iter().enumerate() {
            let arg = RegisterId::Arg { bid, aid };
            for &rid in &live_set {
                int_ig.add_edge(rid, arg);
            }
        }
    }

    for (aid, param) in abi.params_alloc.iter().enumerate() {
        let reg = match param {
            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(reg))) => reg,
            ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Reg(reg))) => reg,
            _ => continue,
        };
        let Some(node_index) = int_ig
            .register_id_2_node_index
            .get_by_left(&RegisterId::Arg {
                bid: definition.bid_init,
                aid,
            })
        else {
            continue;
        };

        if int_ig
            .graph
            .neighbors(*node_index)
            .all(|neighbour| neighbour != caller_saved_register[aid])
        {
            let x = int_ig.graph.node_weight_mut(*node_index).unwrap();
            assert_eq!(*x, None);
            *x = Some(*reg);
        }
    }

    int_ig
}

fn float_interference_graph(
    definition: &FunctionDefinition,
    abi: &FunctionAbi,
    register_mp: &HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
) -> GraphWrapper {
    let mut float_ig: GraphWrapper = GraphWrapper::default();
    for (rid, v) in register_mp {
        match v {
            DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. })
            | DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure { .. }) => {
                float_ig.add_node(*rid);
            }
            _ => {}
        }
    }
    let caller_saved_register = [
        float_ig.graph.add_node(Some(Register::FA0)),
        float_ig.graph.add_node(Some(Register::FA1)),
        float_ig.graph.add_node(Some(Register::FA2)),
        float_ig.graph.add_node(Some(Register::FA3)),
        float_ig.graph.add_node(Some(Register::FA4)),
        float_ig.graph.add_node(Some(Register::FA5)),
        float_ig.graph.add_node(Some(Register::FA6)),
        float_ig.graph.add_node(Some(Register::FA7)),
        float_ig.graph.add_node(Some(Register::FT2)),
        float_ig.graph.add_node(Some(Register::FT3)),
        float_ig.graph.add_node(Some(Register::FT4)),
        float_ig.graph.add_node(Some(Register::FT5)),
        float_ig.graph.add_node(Some(Register::FT6)),
        float_ig.graph.add_node(Some(Register::FT7)),
    ];

    for (a, b) in iproduct!(caller_saved_register, caller_saved_register) {
        if a != b {
            let _ = float_ig.graph.add_edge(a, b, ());
        }
    }

    let liveness_sets = float_inter_block_liveness_graph(definition, register_mp).out;

    for (bid, mut live_set) in liveness_sets {
        let block = definition.blocks.get(&bid).unwrap();

        // first analyze block.exit
        for (rid, _) in block.exit.walk_register() {
            let _ = live_set.insert(rid);
        }
        for (rid, _) in block.exit.walk_register() {
            for a in &live_set {
                float_ig.add_edge(*a, rid);
            }
        }

        // then visit instruction in reverse order
        for (iid, instr) in block.instructions.iter().enumerate().rev() {
            let _ = live_set.remove(&RegisterId::Temp { bid, iid });

            for &a in &live_set {
                float_ig.add_edge(a, RegisterId::Temp { bid, iid });
            }

            if let ir::Instruction::Call { callee, .. } = &**instr {
                // live set conflict with caller_saved_register
                // callee conflict with caller_saved_register
                for a in live_set
                    .iter()
                    .chain(once(callee).filter_map(|operand| match operand {
                        ir::Operand::Constant(_) => None,
                        ir::Operand::Register { rid, .. } => Some(rid),
                    }))
                {
                    let Some(&a) = float_ig.register_id_2_node_index.get_by_left(a) else {
                        continue;
                    };
                    for &b in &caller_saved_register {
                        let _ = float_ig.graph.add_edge(a, b, ());
                    }
                }
            }

            for (rid, _) in instr.walk_register() {
                let _ = live_set.insert(rid);
            }

            for (rid_1, _) in instr.walk_register() {
                for &a in &live_set {
                    float_ig.add_edge(a, rid_1);
                }
            }

            // TODO: check Call T5 T6 ...
        }

        for (aid, _) in block.phinodes.iter().enumerate() {
            let _ = live_set.insert(RegisterId::Arg { bid, aid });
        }

        for (aid, _) in block.phinodes.iter().enumerate() {
            let arg = RegisterId::Arg { bid, aid };
            for &rid in &live_set {
                float_ig.add_edge(rid, arg);
            }
        }
    }

    for (aid, param) in abi.params_alloc.iter().enumerate() {
        let reg = match param {
            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(reg))) => reg,
            ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Reg(reg))) => reg,
            _ => continue,
        };
        let Some(node_index) = float_ig
            .register_id_2_node_index
            .get_by_left(&RegisterId::Arg {
                bid: definition.bid_init,
                aid,
            })
        else {
            continue;
        };

        if float_ig
            .graph
            .neighbors(*node_index)
            .all(|neighbour| neighbour != caller_saved_register[aid])
        {
            let x = float_ig.graph.node_weight_mut(*node_index).unwrap();
            assert_eq!(*x, None);
            *x = Some(*reg);
        }
    }

    float_ig
}

fn spills(
    ig: &mut GraphWrapper,
    colors: &[Register],
    stack_offset_2_s0: &mut i64,
    reserved_rids: &HashSet<RegisterId>,
    register_mp: &mut HashMap<RegisterId, DirectOrInDirect<RegOrStack>>,
) {
    let Some(clique_iterator) = petgraph::algo::peo::CliqueIterator::new(&ig.graph) else {
        dbg!(&ig.register_id_2_node_index);
        println!(
            "{:?}",
            petgraph::dot::Dot::with_config(
                &ig.graph,
                &[
                    petgraph::dot::Config::EdgeNoLabel,
                    petgraph::dot::Config::NodeIndexLabel
                ]
            )
        );
        panic!("not a chordal graph")
    };
    let max_cliques: Vec<_> = clique_iterator
        .filter(|clique| clique.len() >= colors.len())
        .collect();
    if max_cliques.is_empty() {
        return;
    }
    if max_cliques
        .iter()
        .all(|clique| clique.len() <= colors.len())
    {
        return;
    }
    for clique in max_cliques {
        let (uncolored, colored): (Vec<_>, Vec<_>) = clique
            .into_iter()
            .partition(|&node_index| matches!(ig.graph.node_weight(node_index), Some(None)));

        let (reserved_node_index, to_be_spilled): (Vec<NodeIndex>, Vec<NodeIndex>) = uncolored
            .into_iter()
            .filter(|&node_index| matches!(ig.graph.node_weight(node_index), Some(None)))
            .partition(|node_index| {
                reserved_rids.contains(
                    ig.register_id_2_node_index
                        .get_by_right(node_index)
                        .unwrap(),
                )
            });
        if let Some(count) = usize::checked_sub(
            reserved_node_index.len() + to_be_spilled.len() + colored.len(),
            colors.len(),
        ) {
            for node in to_be_spilled
                .iter()
                .chain(reserved_node_index.iter())
                .take(count)
            {
                match ig.graph.remove_node(*node) {
                    Some(Some(_color)) => {
                        // never remove precolored node
                        unreachable!()
                    }
                    Some(None) => {
                        spill(
                            *ig.register_id_2_node_index.get_by_right(node).unwrap(),
                            stack_offset_2_s0,
                            register_mp,
                        );
                    }
                    None => {
                        // already removed
                        unreachable!()
                    }
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
    let mut allocate_on_stack = || {
        let align: i64 = 8;
        let size: i64 = 8;
        while *stack_offset_2_s0 % align != 0 {
            *stack_offset_2_s0 -= 1;
        }
        *stack_offset_2_s0 -= size;
    };

    match register_mp.get(&register_id).unwrap() {
        DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. }) => {
            allocate_on_stack();
            let Some(DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })) = register_mp
                .insert(
                    register_id,
                    DirectOrInDirect::Direct(RegOrStack::Stack {
                        offset_to_s0: *stack_offset_2_s0,
                    }),
                )
            else {
                unreachable!()
            };
        }
        DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
            allocate_on_stack();
            let Some(DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. })) = register_mp
                .insert(
                    register_id,
                    DirectOrInDirect::Direct(RegOrStack::Stack {
                        offset_to_s0: *stack_offset_2_s0,
                    }),
                )
            else {
                unreachable!()
            };
        }
        DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure { .. }) => {
            allocate_on_stack();
            let Some(DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure { .. })) = register_mp
                .insert(
                    register_id,
                    DirectOrInDirect::InDirect(RegOrStack::Stack {
                        offset_to_s0: *stack_offset_2_s0,
                    }),
                )
            else {
                unreachable!()
            };
        }
        DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure { .. }) => {
            allocate_on_stack();
            let Some(DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure { .. })) = register_mp
                .insert(
                    register_id,
                    DirectOrInDirect::InDirect(RegOrStack::Stack {
                        offset_to_s0: *stack_offset_2_s0,
                    }),
                )
            else {
                unreachable!()
            };
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
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
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
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
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
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
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
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::UnaryOp {
                op: UnaryOperator::Negate,
                operand: operand @ ir::Operand::Register { .. },
                dtype: ir::Dtype::Int { .. },
            } => {
                let rs = load_operand_to_reg(
                    operand.clone(),
                    Register::T0,
                    &mut res,
                    register_mp,
                    float_mp,
                );

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::Seqz { rd: *dest_reg, rs }));
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::Seqz {
                            rd: Register::T0,
                            rs,
                        }));
                        res.extend(mk_stype(
                            SType::SW,
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
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
                let rs =
                    load_operand_to_reg(x.clone(), Register::T0, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::Mv { rd: *dest_reg, rs }));
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            rs,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
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
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Plus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                let rs1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::add(dtype.clone()),
                            rd: *dest_reg,
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::add(dtype.clone()),
                            rd: Register::T0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Plus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                let rs1 = load_operand_to_reg(
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
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fadd(dtype.clone()),
                            rd: Register::FT0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Minus,
                lhs,
                rhs: ir::Operand::Constant(c),
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                let value: u64 = match c {
                    ir::Constant::Int {
                        value,
                        is_signed: true,
                        ..
                    } => {
                        let value: i128 = *value as i128;
                        let value: i128 = -value;
                        let value: i64 = value.try_into().unwrap();
                        value as u64
                    }
                    ir::Constant::Int {
                        value,
                        is_signed: false,
                        ..
                    } => {
                        let value: i128 = (*value).try_into().unwrap();
                        let value: i128 = -value;
                        let value: i64 = value.try_into().unwrap();
                        value as u64
                    }
                    _ => unreachable!(),
                };
                let reg1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);

                let data_size = DataSize::try_from(dtype.clone()).unwrap();

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.extend(mk_itype(
                            IType::Addi(data_size),
                            *dest_reg,
                            reg1,
                            value & data_size.mask(),
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.extend(mk_itype(
                            IType::Addi(data_size),
                            Register::T0,
                            reg1,
                            value & data_size.mask(),
                        ));
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Minus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                let rs1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sub(dtype.clone()),
                            rd: *dest_reg,
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sub(dtype.clone()),
                            rd: Register::T0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Minus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                let rs1 = load_operand_to_reg(
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
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fsub(dtype.clone()),
                            rd: Register::FT0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::Multiply,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                let rs1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::mul(dtype.clone()),
                            rd: *dest_reg,
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::mul(dtype.clone()),
                            rd: Register::T0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Multiply,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                let rs1 = load_operand_to_reg(
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
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fmul(dtype.clone()),
                            rd: Register::FT0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::Divide,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { is_signed, .. },
            } => {
                let rs1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::div(dtype.clone(), *is_signed),
                            rd: *dest_reg,
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::div(dtype.clone(), *is_signed),
                            rd: Register::T0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Divide,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                let rs1 = load_operand_to_reg(
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
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::fdiv(dtype.clone()),
                            rd: Register::FT0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
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
                let ir::Constant::Int { value, .. } = c else {
                    unreachable!()
                };
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
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let rs1 = load_operand_to_reg(
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
                            rs1,
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    (
                        ir::Dtype::Float { .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let rs1 = load_operand_to_reg(
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
                            rs1,
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
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
                        let rs1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::FT1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: *dest_reg,
                            rs1,
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
                        let rs1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::FT1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: Register::T0,
                            rs1,
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
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
                        let rs1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::FT1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: *dest_reg,
                            rs1,
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
                        let rs1 = load_operand_to_reg(
                            rhs.clone(),
                            Register::FT1,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: Register::T0,
                            rs1,
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
                            rs2: Some(reg2),
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
                        let rs1 = load_operand_to_reg(
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
                            rs1,
                            rs2: Some(reg2),
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
                        let rs1 = load_operand_to_reg(
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
                            instr: asm::RType::rem(dtype.clone(), *is_signed),
                            rd: *dest_reg,
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    (
                        ir::Dtype::Int { is_signed, .. },
                        DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }),
                    ) => {
                        let rs1 = load_operand_to_reg(
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
                            instr: asm::RType::rem(dtype.clone(), *is_signed),
                            rd: Register::T0,
                            rs1,
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
                op: BinaryOperator::ShiftLeft,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let rs1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sll(target_dtype.clone()),
                            rd: *dest_reg,
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sll(target_dtype.clone()),
                            rd: Register::T0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
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
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
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
                let rs1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sra(target_dtype.clone()),
                            rd: *dest_reg,
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::sra(target_dtype.clone()),
                            rd: Register::T0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::BitwiseXor,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let rs1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Xor,
                            rd: *dest_reg,
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Xor,
                            rd: Register::T0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::BitwiseAnd,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let rs1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::And,
                            rd: *dest_reg,
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::And,
                            rd: Register::T0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::BitwiseOr,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let rs1 =
                    load_operand_to_reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                let reg2 =
                    load_operand_to_reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);

                match register_mp.get(&RegisterId::Temp { bid, iid }).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Or,
                            rd: *dest_reg,
                            rs1,
                            rs2: Some(reg2),
                        });
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Or,
                            rd: Register::T0,
                            rs1,
                            rs2: Some(reg2),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }

            ir::Instruction::Store {
                ptr:
                    ptr @ ir::Operand::Register {
                        dtype: ir::Dtype::Pointer { inner, .. },
                        ..
                    },
                value: operand @ ir::Operand::Constant(ir::Constant::Int { .. }),
            } => {
                let rs2 = load_operand_to_reg(
                    operand.clone(),
                    Register::T1,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                let rs1 =
                    load_operand_to_reg(ptr.clone(), Register::T0, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::SType {
                    instr: SType::store((**inner).clone()),
                    rs1,
                    rs2,
                    imm: Immediate::Value(0),
                });
            }
            ir::Instruction::Store {
                ptr:
                    ptr @ ir::Operand::Register {
                        dtype: ir::Dtype::Pointer { inner, .. },
                        ..
                    },
                value: operand @ ir::Operand::Constant(ir::Constant::Float { .. }),
            } => {
                let rs2 = load_operand_to_reg(
                    operand.clone(),
                    Register::FT0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                let rs1 =
                    load_operand_to_reg(ptr.clone(), Register::T0, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::SType {
                    instr: SType::store((**inner).clone()),
                    rs1,
                    rs2,
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
                let DirectOrInDirect::Direct(reg_or_stack) = register_mp.get(ptr_rid).unwrap()
                else {
                    unreachable!()
                };
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
                    RegOrStack::IntRegNotSure { .. } | RegOrStack::FloatRegNotSure { .. } => {
                        unreachable!()
                    }
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
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
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
                            RegOrStack::IntRegNotSure { .. }
                            | RegOrStack::FloatRegNotSure { .. } => {
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
                let rs2 = load_operand_to_reg(
                    value.clone(),
                    Register::T0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                res.push(asm::Instruction::SType {
                    instr: SType::store(dtype.clone()),
                    rs1: Register::T1,
                    rs2,
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
                let DirectOrInDirect::Direct(src) = register_mp.get(rid).unwrap() else {
                    unreachable!()
                };
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
                    RegOrStack::IntRegNotSure { .. } | RegOrStack::FloatRegNotSure { .. } => {
                        unreachable!()
                    }
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
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
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
                DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => unreachable!(),
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
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => unreachable!(),
                }
            }
            ir::Instruction::Call {
                callee,
                args,
                return_type,
            } => {
                let FunctionAbi {
                    params_alloc,
                    caller_alloc,
                } = clown(callee, function_abi_mp, source);
                if caller_alloc != 0 {
                    res.extend(mk_itype(
                        IType::Addi(DataSize::Double),
                        Register::Sp,
                        Register::Sp,
                        (-(<usize as TryInto<i64>>::try_into(caller_alloc).unwrap())) as u64,
                    ));
                }

                let mut to_be_cp_parallel: Vec<(Register, Register, ir::Dtype)> = Vec::new();
                let mut after_cp_parallel: Vec<asm::Instruction> = Vec::new();

                for (operand, alloc) in izip!(args, params_alloc) {
                    match (alloc, operand) {
                        (
                            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(
                                target_reg,
                            ))),
                            ir::Operand::Constant(c),
                        ) => {
                            load_constant_to_reg(
                                c.clone(),
                                target_reg,
                                &mut after_cp_parallel,
                                float_mp,
                            );
                        }
                        (
                            ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(
                                target_reg,
                            ))),
                            ir::Operand::Register { rid, dtype },
                        ) => match foo(register_mp.get(rid).unwrap()) {
                            None => {
                                store_operand_to_reg(
                                    operand.clone(),
                                    target_reg,
                                    &mut after_cp_parallel,
                                    register_mp,
                                    float_mp,
                                );
                            }
                            Some(src_reg) => {
                                if src_reg != target_reg {
                                    to_be_cp_parallel.push((src_reg, target_reg, dtype.clone()))
                                }
                            }
                        },
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
                                &mut after_cp_parallel,
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
                            unreachable!("{}", operand.dtype())
                        }
                        (
                            ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(RegOrStack::Reg(
                                target_reg,
                            ))),
                            ir::Operand::Register { rid, .. },
                        ) => match register_mp.get(rid).unwrap() {
                            DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                                after_cp_parallel.extend(mk_itype(
                                    IType::Addi(DataSize::Double),
                                    target_reg,
                                    Register::S0,
                                    *offset_to_s0 as u64,
                                ));
                            }
                            DirectOrInDirect::Direct(RegOrStack::Reg(_)) => unreachable!(),
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                                unreachable!()
                            }
                            DirectOrInDirect::InDirect(RegOrStack::Stack { offset_to_s0 }) => {
                                after_cp_parallel.extend(mk_itype(
                                    IType::LD,
                                    target_reg,
                                    Register::S0,
                                    *offset_to_s0 as u64,
                                ));
                            }
                            DirectOrInDirect::InDirect(RegOrStack::Reg(src_reg)) => {
                                if *src_reg != target_reg {
                                    to_be_cp_parallel.push((*src_reg, target_reg, operand.dtype()));
                                }
                            }
                            DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure { .. })
                            | DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure { .. }) => {
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
                                after_cp_parallel.extend(mk_itype(
                                    IType::Addi(DataSize::Double),
                                    Register::T0,
                                    Register::S0,
                                    *offset_to_s0 as u64,
                                ));
                                after_cp_parallel.extend(mk_stype(
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
                                after_cp_parallel.extend(mk_itype(
                                    IType::LD,
                                    Register::T0,
                                    Register::S0,
                                    *offset_to_s0 as u64,
                                ));
                                after_cp_parallel.extend(mk_stype(
                                    SType::Store(DataSize::Double),
                                    Register::Sp,
                                    Register::T0,
                                    target_offset as u64,
                                ));
                            }
                            DirectOrInDirect::InDirect(RegOrStack::Reg(reg)) => {
                                after_cp_parallel.extend(mk_stype(
                                    SType::Store(DataSize::Double),
                                    Register::Sp,
                                    *reg,
                                    target_offset as u64,
                                ));
                            }
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. })
                            | DirectOrInDirect::InDirect(RegOrStack::IntRegNotSure { .. })
                            | DirectOrInDirect::InDirect(RegOrStack::FloatRegNotSure { .. }) => {
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
                            let DirectOrInDirect::Direct(RegOrStack::Stack {
                                offset_to_s0: base_offset,
                            }) = *register_mp.get(rid).unwrap()
                            else {
                                unreachable!()
                            };

                            let Some((_, _, offsets)) = (if size_align_offsets.is_some() {
                                size_align_offsets.clone()
                            } else {
                                source
                                    .structs
                                    .get(name.as_ref().unwrap())
                                    .and_then(|x| x.as_ref())
                                    .and_then(|x| x.get_struct_size_align_offsets())
                                    .and_then(|x| x.as_ref())
                                    .cloned()
                            }) else {
                                unreachable!()
                            };

                            let Some(fields) = (if fields.is_some() {
                                fields.clone()
                            } else {
                                source
                                    .structs
                                    .get(name.as_ref().unwrap())
                                    .and_then(|x| x.as_ref())
                                    .and_then(|x| x.get_struct_fields())
                                    .and_then(|x| x.as_ref())
                                    .cloned()
                            }) else {
                                unreachable!()
                            };

                            for (register_couple, offset, dtype) in izip!(v, offsets, fields) {
                                match register_couple {
                                    RegisterCouple::Single(register) => {
                                        after_cp_parallel.extend(mk_itype(
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
                                        after_cp_parallel.extend(mk_itype(
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

                res.extend(cp_parallel(to_be_cp_parallel));
                res.extend(after_cp_parallel);

                match return_type {
                    ir::Dtype::Struct { .. } => {
                        let DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) =
                            register_mp.get(&RegisterId::Temp { bid, iid }).unwrap()
                        else {
                            unreachable!()
                        };
                        res.extend(mk_itype(
                            IType::Addi(DataSize::Double),
                            Register::T0,
                            Register::S0,
                            *offset_to_s0 as u64,
                        ));
                        res.extend(mk_stype(SType::SD, Register::Sp, Register::T0, 0));
                    }
                    ir::Dtype::Array { .. } => unreachable!(),
                    _ => {}
                }

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
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                                unreachable!()
                            }
                            DirectOrInDirect::InDirect(_) => unreachable!(),
                        };
                        res.push(asm::Instruction::Pseudo(Pseudo::Jalr { rs }));
                    }
                    _ => unreachable!(),
                }

                match return_type {
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
                                    SType::store(return_type.clone()),
                                    Register::S0,
                                    Register::A0,
                                    *offset_to_s0 as u64,
                                ));
                            }
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
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
                                    data_size: DataSize::try_from(return_type.clone()).unwrap(),
                                }));
                            }
                            DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                                res.extend(mk_stype(
                                    SType::store(return_type.clone()),
                                    Register::S0,
                                    Register::FA0,
                                    *offset_to_s0 as u64,
                                ));
                            }
                            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                                unreachable!()
                            }
                            DirectOrInDirect::InDirect(_) => unreachable!(),
                        }
                    }
                    _ => {}
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
                        let rs = load_operand_to_reg(
                            value.clone(),
                            Register::T0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::Pseudo(Pseudo::Mv { rd: *dest_reg, rs }));
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
                        let rs = load_operand_to_reg(
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
                                    rs,
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
                        let rs = load_operand_to_reg(
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
                                    rs,
                                }));
                            }
                            (64, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs,
                                }));
                            }
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
                        let rs = load_operand_to_reg(
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
                                    rs,
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
                                    rs1: rs,
                                    imm: Immediate::Value(255),
                                });
                            }
                            (16, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs,
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
                                    rs,
                                }));
                            }
                            (64, true) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs,
                                }));
                            }
                            (64, false) => {
                                res.push(asm::Instruction::Pseudo(Pseudo::Mv {
                                    rd: *dest_reg,
                                    rs,
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
                        let rs = load_operand_to_reg(
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
                                    rs,
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
                        ir::Dtype::Int {
                            width: 64,
                            is_signed: false,
                            ..
                        },
                        DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)),
                    ) => {
                        let rs = load_operand_to_reg(
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
                                    rs,
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
                        let rs1 = load_operand_to_reg(
                            value.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: RType::fcvt_float_to_int(from, to.clone()),
                            rd: Register::T0,
                            rs1,
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
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
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
                let ir::Constant::Int { value, .. } = c else {
                    unreachable!()
                };
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
            match value.dtype() {
                ir::Dtype::Unit { .. } => {}
                ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => {
                    store_operand_to_reg(
                        value.clone(),
                        Register::A0,
                        &mut res,
                        register_mp,
                        float_mp,
                    );
                }
                ir::Dtype::Float { .. } => {
                    store_operand_to_reg(
                        value.clone(),
                        Register::FA0,
                        &mut res,
                        register_mp,
                        float_mp,
                    );
                }
                ir::Dtype::Struct { .. } => match value {
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
                _ => unreachable!(),
            }
            res.push(asm::Instruction::Pseudo(Pseudo::Ret));
        }
        ir::BlockExit::Unreachable => unreachable!(),
    }

    res
}

fn clown<'a>(
    callee: &'a ir::Operand,
    function_abi_mp: &HashMap<String, FunctionAbi>,
    source: &ir::TranslationUnit,
) -> FunctionAbi {
    match callee {
        ir::Operand::Constant(ir::Constant::GlobalVariable {
            dtype: ir::Dtype::Function { .. },
            name,
        }) => function_abi_mp.get(name).unwrap().clone(),
        ir::Operand::Register {
            dtype: ir::Dtype::Pointer { inner, .. },
            ..
        } => {
            let ir::Dtype::Function { ret, params } = &**inner else {
                unreachable!()
            };
            let function_signature = FunctionSignature {
                ret: (**ret).clone(),
                params: params.clone(),
            };
            function_signature.try_alloc(source)
        }
        _ => unreachable!(),
    }
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
    let mut after_cp_parallel: Vec<asm::Instruction> = Vec::new();
    for (aid, operand) in jump_arg.args.iter().enumerate() {
        match register_mp
            .get(&RegisterId::Arg {
                bid: jump_arg.bid,
                aid,
            })
            .unwrap()
        {
            DirectOrInDirect::Direct(RegOrStack::Reg(dest_reg)) => match operand {
                ir::Operand::Constant(_) => {
                    store_operand_to_reg(
                        operand.clone(),
                        *dest_reg,
                        &mut after_cp_parallel,
                        register_mp,
                        float_mp,
                    );
                }
                ir::Operand::Register { rid, dtype } => match register_mp.get(rid).unwrap() {
                    DirectOrInDirect::Direct(RegOrStack::Reg(src_reg)) => {
                        if *src_reg != *dest_reg {
                            v.push((*src_reg, *dest_reg, dtype.clone()));
                        }
                    }
                    DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                        after_cp_parallel.extend(mk_itype(
                            IType::load(dtype.clone()),
                            *dest_reg,
                            Register::S0,
                            *offset_to_s0 as u64,
                        ));
                    }
                    DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
                    | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => {
                        unreachable!()
                    }
                    DirectOrInDirect::InDirect(_) => todo!(),
                },
            },
            DirectOrInDirect::Direct(RegOrStack::Stack { offset_to_s0 }) => {
                operand_to_stack(
                    operand.clone(),
                    (Register::S0, *offset_to_s0 as u64),
                    &mut after_cp_parallel,
                    register_mp,
                    float_mp,
                );
            }
            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => unreachable!(),
            DirectOrInDirect::InDirect(_) => unreachable!(),
        };
    }

    // cp_parallel first
    res.extend(cp_parallel(v));
    res.extend(after_cp_parallel);

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

fn load_int_to_reg(c: ir::Constant, register: Register) -> Vec<asm::Instruction> {
    if let ir::Constant::Int { value, .. } = c {
        let data_size = DataSize::try_from(c.dtype()).unwrap();
        mk_itype(
            IType::Addi(data_size),
            register,
            Register::Zero,
            value as u64 & data_size.mask(),
        )
    } else {
        unreachable!()
    }
}

fn load_float_to_reg(
    c: ir::Constant,
    register: Register,
    float_mp: &mut FloatMp,
) -> Vec<asm::Instruction> {
    if let ir::Constant::Float { value, width } = c {
        let mut res = Vec::new();
        let label = float_mp.get_label(Float { value, width });
        res.push(asm::Instruction::Pseudo(Pseudo::La {
            rd: Register::T0,
            symbol: label,
        }));
        res.push(asm::Instruction::IType {
            instr: IType::load(c.dtype()),
            rd: register,
            rs1: Register::T0,
            imm: Immediate::Value(0),
        });
        res
    } else {
        unreachable!()
    }
}

fn load_constant_to_reg(
    c: ir::Constant,
    rd: Register,
    res: &mut Vec<asm::Instruction>,
    float_mp: &mut FloatMp,
) {
    match c {
        ir::Constant::Int { .. } => {
            res.extend(load_int_to_reg(c, rd));
        }
        ir::Constant::Float { .. } => {
            res.extend(load_float_to_reg(c, rd, float_mp));
        }
        ir::Constant::GlobalVariable {
            name,
            dtype: ir::Dtype::Function { .. },
        } => {
            res.push(asm::Instruction::Pseudo(Pseudo::La {
                rd,
                symbol: Label(name),
            }));
        }
        ir::Constant::Undef {
            dtype:
                dtype @ (ir::Dtype::Int { .. } | ir::Dtype::Float { .. } | ir::Dtype::Pointer { .. }),
        } => {
            res.push(asm::Instruction::IType {
                instr: IType::load(dtype),
                rd,
                rs1: Register::Zero,
                imm: asm::Immediate::Value(0),
            });
        }
        _ => unreachable!(),
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
        ir::Operand::Constant(c) => {
            load_constant_to_reg(c, or_register, res, float_mp);
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
            DirectOrInDirect::Direct(RegOrStack::IntRegNotSure { .. })
            | DirectOrInDirect::Direct(RegOrStack::FloatRegNotSure { .. }) => unreachable!(),
            DirectOrInDirect::InDirect(_) => unreachable!(),
        },
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
                    .and_then(|x| x.as_ref())
                    .cloned()
            }) else {
                unreachable!()
            };
            let Some(fields) = (if fields.is_some() {
                fields.clone()
            } else {
                source
                    .structs
                    .get(name.as_ref().unwrap())
                    .and_then(|x| x.as_ref())
                    .and_then(|x| x.get_struct_fields())
                    .and_then(|x| x.as_ref())
                    .cloned()
            }) else {
                unreachable!()
            };

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
                    .and_then(|x| x.as_ref())
                    .cloned()
            }) else {
                unreachable!()
            };
            let Some(fields) = (if fields.is_some() {
                fields.clone()
            } else {
                source
                    .structs
                    .get(name.as_ref().unwrap())
                    .and_then(|x| x.as_ref())
                    .and_then(|x| x.get_struct_fields())
                    .and_then(|x| x.as_ref())
                    .cloned()
            }) else {
                unreachable!()
            };

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
                    .and_then(|x| x.as_ref())
                    .cloned()
            }) else {
                unreachable!()
            };
            let Some(fields) = (if fields.is_some() {
                fields.clone()
            } else {
                source
                    .structs
                    .get(name.as_ref().unwrap())
                    .and_then(|x| x.as_ref())
                    .and_then(|x| x.get_struct_fields())
                    .and_then(|x| x.as_ref())
                    .cloned()
            }) else {
                unreachable!()
            };

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
            let Some((size, _, offsets)) = size_align_offsets else {
                unreachable!()
            };
            if initializer.is_none() {
                return vec![Directive::Zero(*size)];
            }

            let Some(Initializer::List(l)) = initializer else {
                unreachable!()
            };
            let Some(fields) = fields else { unreachable!() };

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
    IntRegNotSure {
        /// prefer to be allocate as src
        src: Option<Register>,
    },
    FloatRegNotSure {
        /// prefer to be allocate as src
        src: Option<Register>,
    },
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
enum DirectOrInDirect<T: Clone + Copy> {
    Direct(T),
    InDirect(T),
}

#[derive(Debug, Clone)]
struct FunctionAbi {
    params_alloc: Vec<ParamAlloc>,
    /// contain the ret_alloc
    caller_alloc: usize,
}

impl FunctionSignature {
    fn try_alloc(&self, source: &ir::TranslationUnit) -> FunctionAbi {
        let mut params_alloc: Vec<ParamAlloc> =
            vec![
                ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(RegOrStack::Reg(Register::A0)));
                self.params.len()
            ];

        let mut next_int_reg: usize = 0;
        let mut next_float_reg: usize = 0;
        let mut caller_alloc: usize = 0;

        for (param, param_alloc) in self.params.iter().zip(params_alloc.iter_mut()) {
            let (size, align) = param.size_align_of(&source.structs).unwrap();
            let align = align.max(4);
            match param {
                ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => {
                    if next_int_reg > 7 {
                        while caller_alloc % align != 0 {
                            caller_alloc += 1;
                        }
                        caller_alloc += size;
                        *param_alloc = ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(
                            RegOrStack::Stack {
                                offset_to_s0: caller_alloc.try_into().unwrap(),
                            },
                        ));
                    } else {
                        *param_alloc =
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
                        *param_alloc = ParamAlloc::PrimitiveType(DirectOrInDirect::Direct(
                            RegOrStack::Stack {
                                offset_to_s0: caller_alloc.try_into().unwrap(),
                            },
                        ));
                    } else {
                        *param_alloc =
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
                                .and_then(|x| x.as_ref())
                                .cloned()
                        }) else {
                            unreachable!()
                        };

                        let Some(fields) = (if fields.is_some() {
                            fields.clone()
                        } else {
                            source
                                .structs
                                .get(name.as_ref().unwrap())
                                .and_then(|x| x.as_ref())
                                .and_then(|x| x.get_struct_fields())
                                .and_then(|x| x.as_ref())
                                .cloned()
                        }) else {
                            unreachable!()
                        };

                        let mut j = 0;
                        let mut x: Vec<RegisterCouple> =
                            vec![RegisterCouple::MergedToPrevious; offsets.len()];
                        while j < offsets.len() {
                            match &*fields[j] {
                                ir::Dtype::Int { width: 32, .. } => {
                                    match fields.get(j + 1).map(Deref::deref) {
                                        None => {
                                            x[j] = RegisterCouple::Single(Register::arg(
                                                asm::RegisterType::Integer,
                                                next_int_reg,
                                            ));
                                            next_int_reg += 1;
                                            j += 1;
                                        }
                                        Some(ir::Dtype::Int { width: 32, .. })
                                        | Some(ir::Dtype::Float { width: 32, .. }) => {
                                            x[j] = RegisterCouple::Double(Register::arg(
                                                asm::RegisterType::Integer,
                                                next_int_reg,
                                            ));
                                            next_int_reg += 1;
                                            x[j + 1] = RegisterCouple::MergedToPrevious;
                                            j += 2;
                                        }
                                        Some(ir::Dtype::Int { width: 64, .. })
                                        | Some(ir::Dtype::Float { width: 64, .. }) => {
                                            unreachable!()
                                        }
                                        _ => unreachable!(),
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
                                    match fields.get(j + 1).map(Deref::deref) {
                                        None => {
                                            x[j] = RegisterCouple::Single(Register::arg(
                                                asm::RegisterType::FloatingPoint,
                                                next_float_reg,
                                            ));
                                            next_float_reg += 1;
                                            j += 1;
                                        }
                                        Some(ir::Dtype::Int { width: 32, .. }) => {
                                            x[j] = RegisterCouple::Double(Register::arg(
                                                asm::RegisterType::Integer,
                                                next_int_reg,
                                            ));
                                            next_int_reg += 1;
                                            x[j + 1] = RegisterCouple::MergedToPrevious;
                                            j += 2;
                                        }
                                        Some(ir::Dtype::Float { width: 32, .. }) => {
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
                                        Some(ir::Dtype::Int { width: 64, .. })
                                        | Some(ir::Dtype::Float { width: 64, .. }) => {
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
                        *param_alloc = ParamAlloc::StructInRegister(x);
                    } else if next_int_reg > 7 {
                        while caller_alloc % 8 != 0 {
                            caller_alloc += 1;
                        }
                        caller_alloc += 8;
                        *param_alloc = ParamAlloc::PrimitiveType(DirectOrInDirect::InDirect(
                            RegOrStack::Stack {
                                offset_to_s0: caller_alloc.try_into().unwrap(),
                            },
                        ));
                    } else {
                        *param_alloc =
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

        match &self.ret {
            ir::Dtype::Array { .. } => unimplemented!(),
            ir::Dtype::Struct { .. } => {
                caller_alloc += 16;
            }
            _ => {}
        };

        let caller_alloc = caller_alloc;

        for x in params_alloc.iter_mut() {
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
            params_alloc,
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
                imm,
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
                imm,
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
        let None = register_2_node_index.insert(register, node_index) else {
            unreachable!()
        };
        let None = node_index_2_register.insert(node_index, register) else {
            unreachable!()
        };
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

    let mut linear: HashSet<(Register, Register, ir::Dtype)> = HashSet::new();
    for (src, target, dtype) in &v {
        match (
            nodes_in_loop.get(register_2_node_index.get(src).unwrap()),
            nodes_in_loop.get(register_2_node_index.get(target).unwrap()),
        ) {
            (None, None) | (Some(_), None) => {
                let true = linear.insert((*src, *target, dtype.clone())) else {
                    unreachable!()
                };
            }
            (Some(_), Some(_)) => {
                // deal with this in loop
            }
            (None, Some(_)) => unreachable!(),
        }
    }

    let mut instructions: Vec<asm::Instruction> = Vec::new();

    while !linear.is_empty() {
        let ref x @ (src, target, ref dtype) = linear
            .iter()
            .find(|(_, target, _)| !linear.iter().any(|(src, _, _)| src == target))
            .unwrap()
            .clone();
        instructions.extend(mv_register(src, target, dtype.clone()));
        let true = linear.remove(x) else {
            unreachable!()
        };
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
        let Some(e) = graph.find_edge(src, dest) else {
            unreachable!()
        };
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
    let true = nodes_in_loop.remove(&loop_in_graph[0]) else {
        unreachable!()
    };

    for _ in 0..loop_in_graph.len() - 1 {
        let src = *ordered_loop.last().unwrap();
        let mut iter = graph
            .neighbors_directed(src, petgraph::Direction::Outgoing)
            .filter(|x| nodes_in_loop.contains(x));
        let dest: NodeIndex = iter.next().unwrap();
        assert!(iter.next().is_none());
        ordered_loop.push(dest);
        let true = nodes_in_loop.remove(&dest) else {
            unreachable!()
        };
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
        ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } | ir::Dtype::Struct { .. } => {
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
        _ => unreachable!("{dtype}"),
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
    definition: &FunctionDefinition,
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

impl FunctionDefinition {
    fn get_init_block(&self) -> &ir::Block {
        self.blocks.get(&self.bid_init).unwrap()
    }

    fn get_bid_in_loop(&self) -> HashSet<BlockId> {
        let mut res: HashSet<BlockId> = HashSet::new();
        let mut graph: petgraph::Graph<(), (), petgraph::Directed> = Default::default();
        let mut bid_2_node_index: HashMap<BlockId, NodeIndex> = HashMap::new();
        let mut node_index_2_bid: HashMap<NodeIndex, BlockId> = HashMap::new();
        for &bid in self.blocks.keys() {
            let node_index = graph.add_node(());
            let None = bid_2_node_index.insert(bid, node_index) else {
                unreachable!()
            };
            let None = node_index_2_bid.insert(node_index, bid) else {
                unreachable!()
            };
        }

        for (&curr_bid, block) in &self.blocks {
            for next_bid in block.exit.walk_jump_bid() {
                if curr_bid == next_bid {
                    let true = res.insert(curr_bid) else {
                        unreachable!()
                    };
                } else {
                    let _ = graph.add_edge(
                        *bid_2_node_index.get(&curr_bid).unwrap(),
                        *bid_2_node_index.get(&next_bid).unwrap(),
                        (),
                    );
                }
            }
        }

        let loops: Vec<BlockId> = petgraph::algo::tarjan_scc(&graph)
            .into_iter()
            .filter(|x| x.len() > 1)
            .flatten()
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|node_index| *node_index_2_bid.get(&node_index).unwrap())
            .collect();

        res.extend(loops);
        res
    }

    fn get_rid_in_loop(&self) -> HashSet<RegisterId> {
        let bid_in_loop = self.get_bid_in_loop();

        // check used rid
        bid_in_loop
            .into_iter()
            .flat_map(|bid| {
                self.blocks
                    .get(&bid)
                    .unwrap()
                    .walk_register()
                    .map(|(rid, _)| rid)
            })
            .collect()
    }
}

fn foo(x: &DirectOrInDirect<RegOrStack>) -> Option<Register> {
    match x {
        DirectOrInDirect::Direct(RegOrStack::Reg(reg))
        | DirectOrInDirect::InDirect(RegOrStack::Reg(reg)) => Some(*reg),
        DirectOrInDirect::Direct(RegOrStack::Stack { .. })
        | DirectOrInDirect::InDirect(RegOrStack::Stack { .. }) => None,
        _ => unreachable!(),
    }
}

// a modified lbfs: precolored node first
// output of lbfs = reverse of peo
pub fn lbfs(graph: &StableGraph<Option<Register>, (), petgraph::Undirected>) -> Vec<NodeIndex> {
    let mut res: Vec<NodeIndex> = Vec::new();

    let mut l: LinkedList<Vec<NodeIndex>> = LinkedList::from([graph.node_identifiers().collect()]);

    for _ in 0..graph.node_identifiers().count() {
        let v = l.front_mut().unwrap();

        // precolored node first
        let pivot = match v
            .iter()
            .find(|n| matches!(graph.node_weight(**n), Some(Some(_))))
        {
            Some(&x) => {
                v.retain(|y| *y != x);
                x
            }
            None => v.pop().unwrap(),
        };
        res.push(pivot);

        if v.is_empty() {
            let Some(_) = l.pop_front() else {
                unreachable!()
            };
        }

        let mut cursor = l.cursor_front_mut();
        let neighbour: HashSet<NodeIndex> = graph.neighbors(pivot).collect();

        while let Some(x) = cursor.current() {
            let (a, b): (Vec<_>, Vec<_>) = x.iter().partition(|&y| neighbour.contains(y));
            if a.is_empty() || b.is_empty() {
            } else {
                *x = b;
                cursor.insert_before(a);
            }
            cursor.move_next();
        }
    }
    assert!(l.is_empty());

    res
}

impl TranslationUnit {
    // rm `mv a0, a0`
    // rm `fmv fa0, fa0`
    fn rm_needless_mv(&mut self) {
        for function in self.functions.iter_mut() {
            let function = &mut function.body;
            for block in function.blocks.iter_mut() {
                block.instructions.retain(|instr| match instr {
                    asm::Instruction::Pseudo(Pseudo::Fmv { rd, rs, .. }) => {
                        if rd == rs {
                            false
                        } else {
                            true
                        }
                    }
                    asm::Instruction::Pseudo(Pseudo::Mv { rd, rs }) => {
                        if rd == rs {
                            false
                        } else {
                            true
                        }
                    }
                    asm::Instruction::RType {
                        instr: RType::Add(_),
                        rd,
                        rs1: asm::Register::Zero,
                        rs2: Some(x),
                    }
                    | asm::Instruction::RType {
                        instr: RType::Add(_),
                        rd,
                        rs1: x,
                        rs2: Some(asm::Register::Zero),
                    } => {
                        if rd == x {
                            false
                        } else {
                            true
                        }
                    }
                    _ => true,
                });
            }
        }
    }
}
