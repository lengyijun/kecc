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
use std::collections::HashMap;

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

            let directives = initializer_2_directive(dtype.clone(), initializer.clone(), source);

            asm.unit.variables.push(Section {
                header: vec![Directive::Section(asm::SectionType::Data)],
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

    // s0
    stack_offset_2_s0 -= 8;

    let mut register_mp: HashMap<RegisterId, i64> = HashMap::new();

    let mut alloc_arg = vec![];

    for (aid, (alloc, dtype)) in izip!(params, &signature.params).enumerate() {
        let register_id = RegisterId::Arg {
            bid: definition.bid_init,
            aid,
        };
        match alloc {
            Alloc::Reg(reg) => {
                let (size, align) = dtype.size_align_of(&source.structs).unwrap();
                let align: i64 = align.max(4).try_into().unwrap();
                while stack_offset_2_s0 % align != 0 {
                    stack_offset_2_s0 -= 1;
                }
                stack_offset_2_s0 -= <usize as TryInto<i64>>::try_into(size.max(4)).unwrap();
                alloc_arg.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    *reg,
                    stack_offset_2_s0,
                ));
                let None = register_mp.insert(register_id, stack_offset_2_s0 ) else {unreachable!()};
            }
            Alloc::Stack { offset_to_s0 } => {
                let None = register_mp.insert(register_id, *offset_to_s0 ) else {unreachable!()};
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
            stack_offset_2_s0,
        ));
        while stack_offset_2_s0 % 8 != 0 {
            stack_offset_2_s0 -= 1;
        }
        stack_offset_2_s0 -= 8;
        init_allocation.extend(mk_stype(
            SType::SD,
            Register::S0,
            Register::T0,
            stack_offset_2_s0,
        ));
        let None = register_mp.insert(RegisterId::Local { aid }, stack_offset_2_s0 ) else {unreachable!()};
    }

    for (&bid, block) in definition
        .blocks
        .iter()
        .filter(|(&bid, _)| bid != definition.bid_init)
    {
        for (aid, dtype) in block.phinodes.iter().enumerate() {
            let (size, align) = dtype.size_align_of(&source.structs).unwrap();
            let align: i64 = align.max(4).try_into().unwrap();
            while stack_offset_2_s0 % align != 0 {
                stack_offset_2_s0 -= 1;
            }
            stack_offset_2_s0 -= size.max(4) as i64;
            let None = register_mp.insert(RegisterId::Arg { bid, aid }, stack_offset_2_s0 ) else {unreachable!()};
        }
    }

    for (&bid, block) in definition.blocks.iter() {
        for (iid, instr) in block.instructions.iter().enumerate() {
            let dtype = instr.dtype();
            let (size, align) = dtype.size_align_of(&source.structs).unwrap();
            let align: i64 = align.max(4).try_into().unwrap();
            while stack_offset_2_s0 % align != 0 {
                stack_offset_2_s0 -= 1;
            }
            stack_offset_2_s0 -= size.max(4) as i64;
            let None = register_mp.insert(RegisterId::Temp { bid , iid }, stack_offset_2_s0 ) else {unreachable!()};
        }
    }

    // the stack pointer is always kept 16-byte aligned
    while stack_offset_2_s0 % 16 != 0 {
        stack_offset_2_s0 -= 1;
    }
    let stack_offset_2_s0 = stack_offset_2_s0;

    let backup_ra: Vec<crate::asm::Instruction> = mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::Ra,
        -stack_offset_2_s0 - 8,
    );
    let restore_ra: Vec<crate::asm::Instruction> = mk_itype(
        asm::IType::LD,
        Register::Ra,
        Register::Sp,
        -stack_offset_2_s0 - 8,
    );

    let backup_s0: Vec<crate::asm::Instruction> = mk_stype(
        asm::SType::SD,
        Register::Sp,
        Register::S0,
        -stack_offset_2_s0 - 16,
    );
    let restore_s0: Vec<crate::asm::Instruction> = mk_itype(
        asm::IType::LD,
        Register::S0,
        Register::Sp,
        -stack_offset_2_s0 - 16,
    );

    let mut backup_ra_and_init_sp = mk_itype(
        asm::IType::Addi(DataSize::Double),
        Register::Sp,
        Register::Sp,
        stack_offset_2_s0,
    );
    backup_ra_and_init_sp.extend(backup_ra);
    backup_ra_and_init_sp.extend(backup_s0);
    backup_ra_and_init_sp.extend(mk_itype(
        asm::IType::Addi(DataSize::Double),
        Register::S0,
        Register::Sp,
        -stack_offset_2_s0,
    ));
    backup_ra_and_init_sp.extend(alloc_arg);
    backup_ra_and_init_sp.extend(init_allocation);

    let mut before_ret_instructions = restore_ra;
    before_ret_instructions.extend(restore_s0);
    before_ret_instructions.extend(mk_itype(
        asm::IType::Addi(DataSize::Double),
        Register::Sp,
        Register::Sp,
        -stack_offset_2_s0,
    ));
    before_ret_instructions.push(asm::Instruction::Pseudo(Pseudo::Ret));

    let mut temp_block: Vec<asm::Block> = vec![];

    for (&bid, block) in definition.blocks.iter() {
        let instructions = translate_block(
            func_name,
            bid,
            definition,
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

fn translate_block(
    func_name: &str,
    bid: BlockId,
    definition: &ir::FunctionDefinition,
    block: &ir::Block,
    temp_block: &mut Vec<asm::Block>,
    register_mp: &HashMap<RegisterId, i64>,
    source: &ir::TranslationUnit,
    function_abi_mp: &HashMap<String, FunctionAbi>,
    abi: &FunctionAbi,
    before_ret_instructions: Vec<asm::Instruction>,
    float_mp: &mut FloatMp,
) -> Vec<asm::Instruction> {
    let mut res = vec![];

    for (iid, instr) in block.instructions.iter().enumerate() {
        let Some(  destination  ) = register_mp.get(&RegisterId::Temp { bid, iid }) else {unreachable!()};
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
                match v {
                    Value::Int { value, .. } => {
                        res.push(asm::Instruction::Pseudo(Pseudo::Li {
                            rd: Register::T0,
                            imm: value as u64,
                        }));
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
                        ));
                    }
                    Value::Float { value, width } => {
                        let label = float_mp.get_label(Float { value, width });
                        res.push(asm::Instruction::Pseudo(Pseudo::La {
                            rd: Register::T3,
                            symbol: label,
                        }));
                        res.push(asm::Instruction::IType {
                            instr: IType::load(dtype.clone()),
                            rd: Register::FT0,
                            rs1: Register::T3,
                            imm: Immediate::Value(0),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::FT0,
                            *destination,
                        ));
                    }
                    _ => unreachable!(),
                }
            }
            ir::Instruction::UnaryOp {
                op: UnaryOperator::Minus,
                operand,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                operand2reg(
                    operand.clone(),
                    Register::T0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                res.push(asm::Instruction::Pseudo(Pseudo::neg(
                    dtype.clone(),
                    Register::T1,
                    Register::T0,
                )));
                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::T1,
                    *destination,
                ));
            }
            ir::Instruction::UnaryOp {
                op: UnaryOperator::Minus,
                operand,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                operand2reg(
                    operand.clone(),
                    Register::FT0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                res.push(asm::Instruction::Pseudo(Pseudo::fneg(
                    dtype.clone(),
                    Register::FT1,
                    Register::FT0,
                )));
                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::FT1,
                    *destination,
                ));
            }
            ir::Instruction::UnaryOp {
                op: UnaryOperator::Negate,
                operand,
                dtype: ir::Dtype::Int { .. },
            } => {
                operand2reg(
                    operand.clone(),
                    Register::T0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                res.push(asm::Instruction::Pseudo(Pseudo::Seqz {
                    rd: Register::T0,
                    rs: Register::T0,
                }));
                res.extend(mk_stype(
                    SType::SW,
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }
            ir::Instruction::UnaryOp {
                op: UnaryOperator::Plus,
                operand,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                operand2reg(
                    operand.clone(),
                    Register::T0,
                    &mut res,
                    register_mp,
                    float_mp,
                );

                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }
            ir::Instruction::UnaryOp {
                op: UnaryOperator::Plus,
                operand,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                operand2reg(
                    operand.clone(),
                    Register::FT0,
                    &mut res,
                    register_mp,
                    float_mp,
                );

                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::FT0,
                    *destination,
                ));
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
                match v {
                    Value::Int { value, .. } => {
                        res.push(asm::Instruction::Pseudo(Pseudo::Li {
                            rd: Register::T0,
                            imm: value as u64,
                        }));
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
                        ));
                    }
                    Value::Float { value, width } => {
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
                            *destination,
                        ));
                    }
                    _ => unreachable!(),
                }
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Plus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::add(dtype.clone()),
                    rd: Register::T0,
                    rs1: Register::T1,
                    rs2: Some(Register::T0),
                });
                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Plus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                operand2reg(lhs.clone(), Register::FT0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::FT1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::fadd(dtype.clone()),
                    rd: Register::FT0,
                    rs1: Register::FT1,
                    rs2: Some(Register::FT0),
                });
                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::FT0,
                    *destination,
                ));
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Minus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::Sub(DataSize::try_from(dtype.clone()).unwrap()),
                    rd: Register::T0,
                    rs1: Register::T0,
                    rs2: Some(Register::T1),
                });
                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Minus,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                operand2reg(lhs.clone(), Register::FT0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::FT1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::fsub(dtype.clone()),
                    rd: Register::FT0,
                    rs1: Register::FT0,
                    rs2: Some(Register::FT1),
                });
                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::FT0,
                    *destination,
                ));
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::Multiply,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { .. },
            } => {
                operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::mul(dtype.clone()),
                    rd: Register::T0,
                    rs1: Register::T1,
                    rs2: Some(Register::T0),
                });
                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Multiply,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                operand2reg(lhs.clone(), Register::FT0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::FT1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::fmul(dtype.clone()),
                    rd: Register::FT0,
                    rs1: Register::FT1,
                    rs2: Some(Register::FT0),
                });
                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::FT0,
                    *destination,
                ));
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::Divide,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Int { is_signed, .. },
            } => {
                operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::div(dtype.clone(), *is_signed),
                    rd: Register::T0,
                    rs1: Register::T0,
                    rs2: Some(Register::T1),
                });
                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }
            ir::Instruction::BinOp {
                op: BinaryOperator::Divide,
                lhs,
                rhs,
                dtype: dtype @ ir::Dtype::Float { .. },
            } => {
                operand2reg(lhs.clone(), Register::FT0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::FT1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::fdiv(dtype.clone()),
                    rd: Register::FT0,
                    rs1: Register::FT0,
                    rs2: Some(Register::FT1),
                });
                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::FT0,
                    *destination,
                ));
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::Equals,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                let dtype = lhs.dtype();
                match &dtype {
                    ir::Dtype::Int { .. } => {
                        operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Xor,
                            rd: Register::T0,
                            rs1: Register::T1,
                            rs2: Some(Register::T0),
                        });
                        res.push(asm::Instruction::Pseudo(Pseudo::Seqz {
                            rd: Register::T0,
                            rs: Register::T0,
                        }));
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
                        ));
                    }
                    ir::Dtype::Float { .. } => {
                        operand2reg(lhs.clone(), Register::FT0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::FT1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::feq(dtype.clone()),
                            rd: Register::T0,
                            rs1: Register::FT1,
                            rs2: Some(Register::FT0),
                        });
                        res.extend(mk_stype(
                            SType::store(dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
                        ));
                    }
                    ir::Dtype::Pointer { .. } => todo!(),
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
                match &dtype {
                    ir::Dtype::Int { .. } => {
                        operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Xor,
                            rd: Register::T0,
                            rs1: Register::T1,
                            rs2: Some(Register::T0),
                        });
                        res.push(asm::Instruction::Pseudo(Pseudo::Snez {
                            rd: Register::T0,
                            rs: Register::T0,
                        }));
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
                        ));
                    }
                    ir::Dtype::Float { .. } => {
                        operand2reg(lhs.clone(), Register::FT0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::FT1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::feq(dtype.clone()),
                            rd: Register::T0,
                            rs1: Register::FT1,
                            rs2: Some(Register::FT0),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
                        ));
                    }
                    ir::Dtype::Pointer { .. } => todo!(),
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

                match &dtype {
                    ir::Dtype::Int { is_signed, .. } => {
                        operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: Register::T0,
                            rs1: Register::T0,
                            rs2: Some(Register::T1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
                        ));
                    }
                    ir::Dtype::Float { .. } => {
                        operand2reg(lhs.clone(), Register::FT0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::FT1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: Register::T0,
                            rs1: Register::FT0,
                            rs2: Some(Register::FT1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
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
                match &dtype {
                    ir::Dtype::Int { is_signed, .. } => {
                        operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: Register::T0,
                            rs1: Register::T1,
                            rs2: Some(Register::T0),
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
                            *destination,
                        ));
                    }
                    ir::Dtype::Float { .. } => {
                        operand2reg(lhs.clone(), Register::FT0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::FT1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: Register::T0,
                            rs1: Register::FT1,
                            rs2: Some(Register::FT0),
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
                            *destination,
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
                match &dtype {
                    ir::Dtype::Int { is_signed, .. } => {
                        operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: Register::T0,
                            rs1: Register::T1,
                            rs2: Some(Register::T0),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
                        ));
                    }
                    ir::Dtype::Float { .. } => {
                        operand2reg(lhs.clone(), Register::FT0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::FT1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: Register::T0,
                            rs1: Register::FT1,
                            rs2: Some(Register::FT0),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
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
                match &dtype {
                    ir::Dtype::Int { is_signed, .. } => {
                        operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::Slt {
                                is_signed: *is_signed,
                            },
                            rd: Register::T0,
                            rs1: Register::T0,
                            rs2: Some(Register::T1),
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
                            *destination,
                        ));
                    }
                    ir::Dtype::Float { .. } => {
                        operand2reg(lhs.clone(), Register::FT0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::FT1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::flt(dtype.clone()),
                            rd: Register::T0,
                            rs1: Register::FT0,
                            rs2: Some(Register::FT1),
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
                            *destination,
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
                match &dtype {
                    ir::Dtype::Int { is_signed, .. } => {
                        operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                        operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: asm::RType::rem(dtype.clone(), *is_signed),
                            rd: Register::T0,
                            rs1: Register::T0,
                            rs2: Some(Register::T1),
                        });
                        res.extend(mk_stype(
                            SType::store(target_dtype.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
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
                operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::sll(target_dtype.clone()),
                    rd: Register::T0,
                    rs1: Register::T0,
                    rs2: Some(Register::T1),
                });
                res.extend(mk_stype(
                    SType::store(target_dtype.clone()),
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::ShiftRight,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::sra(target_dtype.clone()),
                    rd: Register::T0,
                    rs1: Register::T0,
                    rs2: Some(Register::T1),
                });
                res.extend(mk_stype(
                    SType::store(target_dtype.clone()),
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::BitwiseXor,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::Xor,
                    rd: Register::T0,
                    rs1: Register::T0,
                    rs2: Some(Register::T1),
                });
                res.extend(mk_stype(
                    SType::store(target_dtype.clone()),
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::BitwiseAnd,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::And,
                    rd: Register::T0,
                    rs1: Register::T1,
                    rs2: Some(Register::T0),
                });
                res.extend(mk_stype(
                    SType::store(target_dtype.clone()),
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }

            ir::Instruction::BinOp {
                op: BinaryOperator::BitwiseOr,
                lhs,
                rhs,
                dtype: target_dtype @ ir::Dtype::Int { .. },
            } => {
                operand2reg(lhs.clone(), Register::T0, &mut res, register_mp, float_mp);
                operand2reg(rhs.clone(), Register::T1, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::RType {
                    instr: asm::RType::Or,
                    rd: Register::T0,
                    rs1: Register::T1,
                    rs2: Some(Register::T0),
                });
                res.extend(mk_stype(
                    SType::store(target_dtype.clone()),
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }

            ir::Instruction::Store {
                ptr:
                    ir::Operand::Register {
                        rid,
                        dtype: ir::Dtype::Pointer { inner, .. },
                    },
                value: operand @ ir::Operand::Constant(ir::Constant::Int { .. }),
            } => {
                operand2reg(
                    operand.clone(),
                    Register::T1,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                res.extend(mk_itype(
                    IType::LD,
                    Register::T0,
                    Register::S0,
                    *register_mp.get(rid).unwrap(),
                ));
                res.push(asm::Instruction::SType {
                    instr: SType::store((**inner).clone()),
                    rs1: Register::T0,
                    rs2: Register::T1,
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
                operand2reg(
                    operand.clone(),
                    Register::FT0,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                res.extend(mk_itype(
                    IType::LD,
                    Register::T0,
                    Register::S0,
                    *register_mp.get(rid).unwrap(),
                ));
                res.push(asm::Instruction::SType {
                    instr: SType::store((**inner).clone()),
                    rs1: Register::T0,
                    rs2: Register::FT0,
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
                cp_to_indirect_target(
                    (Register::S0, *register_mp.get(value_rid).unwrap()),
                    (Register::S0, *register_mp.get(ptr_rid).unwrap() as u64),
                    0,
                    dtype.clone(),
                    &mut res,
                    source,
                );
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
                operand2reg(value.clone(), Register::T0, &mut res, register_mp, float_mp);
                res.push(asm::Instruction::SType {
                    instr: SType::store(dtype.clone()),
                    rs1: Register::T1,
                    rs2: Register::T0,
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
                let offset_to_s0 = *register_mp.get(rid).unwrap();
                cp_from_indirect_source(
                    (Register::S0, offset_to_s0 as u64),
                    *destination,
                    0,
                    (**inner).clone(),
                    &mut res,
                    source,
                );
            }
            ir::Instruction::Load {
                ptr:
                    ir::Operand::Constant(ir::Constant::GlobalVariable {
                        name,
                        dtype: dtype @ ir::Dtype::Int { .. },
                    }),
            } => {
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
                    *destination,
                ));
            }
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
                    *destination,
                ));
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
                        -(<usize as TryInto<i64>>::try_into(caller_alloc).unwrap()),
                    ));
                }
                for (param, alloc, dtype) in izip!(args, params_alloc, params_dtype) {
                    match (alloc, param) {
                        (Alloc::Reg(reg), _) => {
                            operand2reg(param.clone(), reg, &mut res, register_mp, float_mp);
                        }
                        (
                            Alloc::Stack { offset_to_s0 },
                            ir::Operand::Constant(ir::Constant::Int { value, .. }),
                        ) => {
                            assert!(offset_to_s0 > 0);
                            res.push(asm::Instruction::Pseudo(Pseudo::Li {
                                rd: Register::T0,
                                imm: *value as u64,
                            }));
                            res.extend(mk_stype(
                                SType::store(dtype.clone()),
                                Register::Sp,
                                Register::T0,
                                offset_to_s0,
                            ));
                        }
                        (
                            Alloc::Stack { offset_to_s0 },
                            operand @ ir::Operand::Constant(ir::Constant::Float { .. }),
                        ) => {
                            assert!(offset_to_s0 > 0);
                            operand2reg(
                                operand.clone(),
                                Register::FT0,
                                &mut res,
                                register_mp,
                                float_mp,
                            );
                            res.extend(mk_stype(
                                SType::store(dtype.clone()),
                                Register::Sp,
                                Register::FT0,
                                offset_to_s0,
                            ));
                        }
                        (Alloc::Stack { offset_to_s0 }, ir::Operand::Register { rid, dtype }) => {
                            assert!(offset_to_s0 > 0);
                            cp(
                                (Register::S0, *register_mp.get(rid).unwrap()),
                                (Register::Sp, offset_to_s0),
                                dtype.clone(),
                                &mut res,
                                source,
                            );
                        }
                        _ => unreachable!(),
                    }
                }

                match ret_alloc {
                    RetLocation::OnStack => {
                        res.extend(mk_itype(
                            IType::Addi(DataSize::Double),
                            Register::T0,
                            Register::S0,
                            *destination,
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
                        res.extend(mk_itype(
                            IType::Load {
                                data_size: DataSize::Word,
                                is_signed: false,
                            },
                            Register::T6,
                            Register::S0,
                            *register_mp.get(rid).unwrap(),
                        ));
                        res.push(asm::Instruction::Pseudo(Pseudo::Jalr { rs: Register::T6 }));
                    }
                    _ => unreachable!(),
                }
                match ret_alloc {
                    RetLocation::OnStack => {}
                    RetLocation::InRegister => match ret_dtype {
                        ir::Dtype::Unit { .. } => {}
                        ir::Dtype::Pointer { .. } | ir::Dtype::Int { .. } => {
                            res.extend(mk_stype(
                                SType::store(ret_dtype.clone()),
                                Register::S0,
                                Register::A0,
                                *destination,
                            ));
                        }
                        ir::Dtype::Float { .. } => {
                            res.extend(mk_stype(
                                SType::store(ret_dtype.clone()),
                                Register::S0,
                                Register::FA0,
                                *destination,
                            ));
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
                target_dtype: to @ ir::Dtype::Int { .. },
            } => {
                let from = value.dtype();

                match &from {
                    ir::Dtype::Int { .. } => {
                        operand2reg(value.clone(), Register::T0, &mut res, register_mp, float_mp);
                        res.extend(mk_stype(
                            SType::store(to.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
                        ));
                    }
                    ir::Dtype::Float { .. } => {
                        operand2reg(
                            value.clone(),
                            Register::FT0,
                            &mut res,
                            register_mp,
                            float_mp,
                        );
                        res.push(asm::Instruction::RType {
                            instr: RType::fcvt_float_to_int(from, to.clone()),
                            rd: Register::T0,
                            rs1: Register::FT0,
                            rs2: None,
                        });
                        res.extend(mk_stype(
                            SType::store(to.clone()),
                            Register::S0,
                            Register::T0,
                            *destination,
                        ));
                    }
                    _ => unreachable!(),
                }
            }
            ir::Instruction::TypeCast {
                value,
                target_dtype: to @ ir::Dtype::Float { .. },
            } => {
                let from = value.dtype();

                match &from {
                    ir::Dtype::Int { .. } => {
                        operand2reg(value.clone(), Register::T0, &mut res, register_mp, float_mp);
                        res.push(asm::Instruction::RType {
                            instr: RType::fcvt_int_to_float(from, to.clone()),
                            rd: Register::FT1,
                            rs1: Register::T0,
                            rs2: None,
                        });
                        res.extend(mk_stype(
                            SType::store(to.clone()),
                            Register::S0,
                            Register::FT1,
                            *destination,
                        ));
                    }
                    ir::Dtype::Float { .. } => {
                        operand2reg(
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
                            rs1: Register::FT0,
                            rs2: None,
                        });
                        res.extend(mk_stype(
                            SType::store(to.clone()),
                            Register::S0,
                            Register::FT1,
                            *destination,
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
                match ptr {
                    ir::Operand::Constant(ir::Constant::GlobalVariable { name, .. }) => {
                        res.push(asm::Instruction::Pseudo(Pseudo::La {
                            rd: Register::T0,
                            symbol: Label(name.clone()),
                        }));
                    }
                    ptr @ ir::Operand::Register { .. } => {
                        operand2reg(ptr.clone(), Register::T0, &mut res, register_mp, float_mp);
                    }
                    _ => unreachable!(),
                }
                operand2reg(
                    offset.clone(),
                    Register::T1,
                    &mut res,
                    register_mp,
                    float_mp,
                );
                res.push(asm::Instruction::RType {
                    instr: RType::add(dtype.clone()),
                    rd: Register::T0,
                    rs1: Register::T0,
                    rs2: Some(Register::T1),
                });
                res.extend(mk_stype(
                    SType::store(dtype.clone()),
                    Register::S0,
                    Register::T0,
                    *destination,
                ));
            }
            ir::Instruction::Nop => {}
            _ => unreachable!("{:?}", &**instr),
        }
    }

    match &block.exit {
        ir::BlockExit::Jump { arg } => {
            gen_jump_arg(func_name, arg, &mut res, register_mp, definition, float_mp);
        }

        ir::BlockExit::ConditionalJump {
            condition,
            arg_then,
            arg_else,
        } => {
            operand2reg(
                condition.clone(),
                Register::T0,
                &mut res,
                register_mp,
                float_mp,
            );
            let (else_label, else_block) = new_tmp_block(func_name, bid, arg_else.bid, temp_block);
            res.push(asm::Instruction::BType {
                instr: asm::BType::Beq,
                rs1: Register::T0,
                rs2: Register::Zero,
                imm: else_label,
            });
            gen_jump_arg(
                func_name,
                arg_then,
                &mut res,
                register_mp,
                definition,
                float_mp,
            );
            gen_jump_arg(
                func_name,
                arg_else,
                else_block,
                register_mp,
                definition,
                float_mp,
            );
        }

        ir::BlockExit::Switch {
            value,
            default,
            cases,
        } => {
            operand2reg(value.clone(), Register::T0, &mut res, register_mp, float_mp);
            for (c, jump_arg) in cases {
                let ir::Constant::Int { value, .. } = c else {unreachable!()};
                res.push(asm::Instruction::Pseudo(Pseudo::Li {
                    rd: Register::T1,
                    imm: *value as u64,
                }));
                let (then_label, then_block) =
                    new_tmp_block(func_name, bid, jump_arg.bid, temp_block);
                gen_jump_arg(
                    func_name,
                    jump_arg,
                    then_block,
                    register_mp,
                    definition,
                    float_mp,
                );
                res.push(asm::Instruction::BType {
                    instr: asm::BType::Beq,
                    rs1: Register::T0,
                    rs2: Register::T1,
                    imm: then_label,
                })
            }
            gen_jump_arg(
                func_name,
                default,
                &mut res,
                register_mp,
                definition,
                float_mp,
            );
        }

        ir::BlockExit::Return { value } => {
            match (&abi.ret_alloc, value.dtype()) {
                (_, ir::Dtype::Unit { .. }) => {}
                (RetLocation::InRegister, ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. }) => {
                    operand2reg(value.clone(), Register::A0, &mut res, register_mp, float_mp)
                }
                (RetLocation::InRegister, ir::Dtype::Float { .. }) => operand2reg(
                    value.clone(),
                    Register::FA0,
                    &mut res,
                    register_mp,
                    float_mp,
                ),
                (RetLocation::InRegister, ir::Dtype::Array { .. }) => unimplemented!(),
                (RetLocation::InRegister, ir::Dtype::Struct { .. }) => unreachable!(),
                (RetLocation::InRegister, ir::Dtype::Function { .. }) => unimplemented!(),
                (RetLocation::OnStack, ir::Dtype::Int { .. }) => unreachable!(),
                (RetLocation::OnStack, ir::Dtype::Float { .. }) => unreachable!(),
                (RetLocation::OnStack, ir::Dtype::Pointer { .. }) => unreachable!(),
                (RetLocation::OnStack, ir::Dtype::Array { .. }) => unimplemented!(),
                (RetLocation::OnStack, ir::Dtype::Struct { .. }) => match value {
                    ir::Operand::Constant(ir::Constant::Undef { .. }) => {}
                    ir::Operand::Register { rid, dtype } => {
                        cp_to_indirect_target(
                            (Register::S0, *register_mp.get(rid).unwrap()),
                            (Register::S0, 0),
                            0,
                            dtype.clone(),
                            &mut res,
                            source,
                        );
                    }
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

fn new_tmp_block<'a>(
    func_name: &str,
    from: BlockId,
    to: BlockId,
    temp_block: &'a mut Vec<asm::Block>,
) -> (Label, &'a mut Vec<asm::Instruction>) {
    let label = Label(format!(".{func_name}_{from}_{to}"));
    temp_block.push(asm::Block {
        label: Some(label.clone()),
        instructions: vec![],
    });

    (label, &mut temp_block.last_mut().unwrap().instructions)
}

// prepare args to jump block
fn gen_jump_arg(
    func_name: &str,
    jump_arg: &ir::JumpArg,
    res: &mut Vec<asm::Instruction>,
    register_mp: &HashMap<RegisterId, i64>,
    definition: &ir::FunctionDefinition,
    float_mp: &mut FloatMp,
) {
    let target_block = definition.blocks.get(&jump_arg.bid).unwrap();
    for (aid, (_dtype, operand)) in izip!(&target_block.phinodes, &jump_arg.args).enumerate() {
        let target_offset = register_mp
            .get(&RegisterId::Arg {
                bid: jump_arg.bid,
                aid,
            })
            .unwrap();
        operand_to_stack(
            operand.clone(),
            *target_offset as u64,
            res,
            register_mp,
            float_mp,
        );
    }
    res.push(asm::Instruction::Pseudo(Pseudo::J {
        offset: Label::new(func_name, jump_arg.bid),
    }));
}

fn operand2reg(
    operand: ir::Operand,
    register: Register,
    res: &mut Vec<asm::Instruction>,
    register_mp: &HashMap<RegisterId, i64>,
    float_mp: &mut FloatMp,
) {
    match operand {
        ir::Operand::Constant(ir::Constant::Int { value, .. }) => {
            res.push(asm::Instruction::Pseudo(Pseudo::Li {
                rd: register,
                imm: value as u64,
            }));
        }
        ir::Operand::Constant(ref c @ ir::Constant::Float { value, width }) => {
            let label = float_mp.get_label(Float { value, width });
            res.push(asm::Instruction::Pseudo(Pseudo::La {
                rd: Register::T3,
                symbol: label,
            }));
            res.push(asm::Instruction::IType {
                instr: IType::load(c.dtype()),
                rd: register,
                rs1: Register::T3,
                imm: Immediate::Value(0),
            });
        }
        ir::Operand::Constant(ir::Constant::GlobalVariable {
            name,
            dtype: ir::Dtype::Function { .. },
        }) => {
            res.push(asm::Instruction::Pseudo(Pseudo::La {
                rd: register,
                symbol: Label(name),
            }));
        }
        ir::Operand::Constant(ir::Constant::Undef {
            dtype:
                dtype @ (ir::Dtype::Int { .. } | ir::Dtype::Float { .. } | ir::Dtype::Pointer { .. }),
        }) => {
            res.push(asm::Instruction::IType {
                instr: IType::load(dtype),
                rd: register,
                rs1: Register::Zero,
                imm: asm::Immediate::Value(0),
            });
        }
        ir::Operand::Register { rid, dtype } => {
            let offset_to_s0 = register_mp.get(&rid).unwrap();
            res.extend(mk_itype(
                IType::load(dtype),
                register,
                Register::S0,
                *offset_to_s0,
            ));
        }
        _ => unreachable!("{:?}", operand),
    }
}

fn operand_to_stack(
    operand: ir::Operand,
    target_base_to_s0: u64,
    res: &mut Vec<asm::Instruction>,
    register_mp: &HashMap<RegisterId, i64>,
    float_mp: &mut FloatMp,
) {
    match operand.dtype() {
        ir::Dtype::Unit { .. } => unreachable!(),
        dtype @ (ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. }) => {
            operand2reg(operand, Register::T0, res, register_mp, float_mp);
            res.extend(mk_stype(
                SType::store(dtype),
                Register::S0,
                Register::T0,
                target_base_to_s0 as i64,
            ));
        }
        dtype @ ir::Dtype::Float { .. } => {
            operand2reg(operand, Register::FT0, res, register_mp, float_mp);
            res.extend(mk_stype(
                SType::store(dtype),
                Register::S0,
                Register::FT0,
                target_base_to_s0 as i64,
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
    (source_pointer, source_offset): (Register, u64), // indirect
    target_base_to_s0: i64,
    offset: i64,
    dtype: ir::Dtype,
    res: &mut Vec<asm::Instruction>,
    source: &ir::TranslationUnit,
) {
    match &dtype {
        ir::Dtype::Int { .. } | ir::Dtype::Pointer { .. } => {
            res.extend(mk_itype(
                asm::IType::LD,
                Register::T0,
                source_pointer,
                source_offset as i64,
            ));
            res.extend(mk_itype(
                asm::IType::load(dtype.clone()),
                Register::T0,
                Register::T0,
                offset,
            ));
            res.extend(mk_stype(
                SType::store(dtype.clone()),
                Register::S0,
                Register::T0,
                target_base_to_s0 + offset,
            ));
        }
        ir::Dtype::Float { .. } => {
            res.extend(mk_itype(
                asm::IType::LD,
                Register::T0,
                source_pointer,
                source_offset as i64,
            ));
            res.extend(mk_itype(
                asm::IType::load(dtype.clone()),
                Register::FT0,
                Register::T0,
                offset,
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
                    (source_pointer, source_offset),
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
                    (source_pointer, source_offset),
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
    (target_reg, target_base): (Register, u64), // indirect
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
                source_base + offset,
            ));
            res.extend(mk_itype(
                IType::LD,
                Register::T1,
                target_reg,
                target_base as i64,
            ));
            res.extend(mk_stype(
                SType::store(dtype),
                Register::T1,
                Register::T0,
                offset,
            ));
        }
        ir::Dtype::Float { .. } => {
            res.extend(mk_itype(
                IType::load(dtype.clone()),
                Register::FT0,
                source_reg,
                source_base + offset,
            ));
            res.extend(mk_itype(
                IType::LD,
                Register::T1,
                target_reg,
                target_base as i64,
            ));
            res.extend(mk_stype(
                SType::store(dtype),
                Register::T1,
                Register::FT0,
                offset,
            ));
        }
        ir::Dtype::Array { inner, size } => {
            let (size_of_inner_type, _) = inner.size_align_of(&source.structs).unwrap();
            for i in 0..*size {
                cp_to_indirect_target(
                    (source_reg, source_base),
                    (target_reg, target_base),
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
                    (target_reg, target_base),
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

fn cp(
    (source_reg, source_base): (Register, i64), // direct
    (target_reg, target_base): (Register, i64), // direct
    dtype: ir::Dtype,
    res: &mut Vec<asm::Instruction>,
    source: &ir::TranslationUnit,
) {
    match &dtype {
        ir::Dtype::Pointer { .. } | ir::Dtype::Int { .. } => {
            res.extend(mk_itype(
                IType::load(dtype.clone()),
                Register::T0,
                source_reg,
                source_base,
            ));
            res.extend(mk_stype(
                SType::store(dtype),
                target_reg,
                Register::T0,
                target_base,
            ));
        }
        ir::Dtype::Float { .. } => {
            res.extend(mk_itype(
                IType::load(dtype.clone()),
                Register::FT0,
                source_reg,
                source_base,
            ));
            res.extend(mk_stype(
                SType::store(dtype),
                target_reg,
                Register::FT0,
                target_base,
            ));
        }
        ir::Dtype::Array { inner, size } => {
            let (size_of_inner_type, _) = inner.size_align_of(&source.structs).unwrap();
            for i in 0..*size {
                cp(
                    (
                        source_reg,
                        source_base
                            + <usize as TryInto<i64>>::try_into(i * size_of_inner_type).unwrap(),
                    ),
                    (
                        target_reg,
                        target_base
                            + <usize as TryInto<i64>>::try_into(i * size_of_inner_type).unwrap(),
                    ),
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
                cp(
                    (
                        source_reg,
                        source_base + <usize as TryInto<i64>>::try_into(field_offset).unwrap(),
                    ),
                    (
                        target_reg,
                        target_base + <usize as TryInto<i64>>::try_into(field_offset).unwrap(),
                    ),
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
            let initializer = initializer.map(|x| {
                let Initializer::List(l) = x else {unreachable!()};
                l
            });
            let mut v = vec![];
            for i in 0..*size {
                let initializer: Option<Initializer> = initializer
                    .as_ref()
                    .and_then(|l| l.get(i).map(|x| x.node.initializer.node.clone()));
                v.push(initializer_2_directive(*inner.clone(), initializer, source));
            }
            v.into_iter().flatten().collect()
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
enum Alloc {
    Reg(Register),
    Stack { offset_to_s0: i64 },
}

#[derive(Debug, Clone, Copy)]
enum RetLocation {
    OnStack,
    InRegister,
}

#[derive(Clone)]
struct FunctionAbi {
    params_alloc: Vec<Alloc>,
    /// if on stack, offset must be zero
    ret_alloc: RetLocation,
    /// contain the ret_alloc
    caller_alloc: usize,
}

impl FunctionSignature {
    fn try_alloc(&self, source: &ir::TranslationUnit) -> FunctionAbi {
        let mut params: Vec<Option<Alloc>> = vec![None; self.params.len()];

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
                        params[i] = Some(Alloc::Stack {
                            offset_to_s0: caller_alloc.try_into().unwrap(),
                        });
                    } else {
                        params[i] = Some(Alloc::Reg(Register::arg(
                            asm::RegisterType::Integer,
                            next_int_reg,
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
                        params[i] = Some(Alloc::Stack {
                            offset_to_s0: caller_alloc.try_into().unwrap(),
                        });
                    } else {
                        params[i] = Some(Alloc::Reg(Register::arg(
                            asm::RegisterType::FloatingPoint,
                            next_float_reg,
                        )));
                        next_float_reg += 1;
                    }
                }
                ir::Dtype::Struct { .. } => {
                    while caller_alloc % align != 0 {
                        caller_alloc += 1;
                    }
                    caller_alloc += size;
                    params[i] = Some(Alloc::Stack {
                        offset_to_s0: caller_alloc.try_into().unwrap(),
                    });
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
                Some(Alloc::Reg(..)) => {}
                Some(Alloc::Stack { offset_to_s0 }) => {
                    let offset_to_s0: usize = (*offset_to_s0).try_into().unwrap();
                    *x = Some(Alloc::Stack {
                        offset_to_s0: (caller_alloc - offset_to_s0).try_into().unwrap(),
                    });
                }
                None => unreachable!(),
            }
        }

        FunctionAbi {
            ret_alloc,
            params_alloc: params.into_iter().map(|x| x.unwrap()).collect(),
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

fn mk_itype(instr: IType, rd: Register, rs1: Register, imm: i64) -> Vec<asm::Instruction> {
    if (-2048..=2047).contains(&imm) {
        vec![asm::Instruction::IType {
            instr,
            rd,
            rs1,
            imm: Immediate::Value(imm as u64),
        }]
    } else {
        vec![
            asm::Instruction::Pseudo(Pseudo::Li {
                rd: Register::T5,
                imm: imm as u64,
            }),
            asm::Instruction::RType {
                instr: RType::Add(DataSize::Double),
                rd: Register::T5,
                rs1,
                rs2: Some(Register::T5),
            },
            asm::Instruction::IType {
                instr,
                rd,
                rs1: Register::T5,
                imm: Immediate::Value(0),
            },
        ]
    }
}

fn mk_stype(instr: SType, rs1: Register, rs2: Register, imm: i64) -> Vec<asm::Instruction> {
    if (-2048..=2047).contains(&imm) {
        vec![asm::Instruction::SType {
            instr,
            rs1,
            rs2,
            imm: Immediate::Value(imm as u64),
        }]
    } else {
        vec![
            asm::Instruction::Pseudo(Pseudo::Li {
                rd: Register::T5,
                imm: imm as u64,
            }),
            asm::Instruction::RType {
                instr: RType::Add(DataSize::Double),
                rd: Register::T5,
                rs1,
                rs2: Some(Register::T5),
            },
            asm::Instruction::SType {
                instr,
                rs1: Register::T5,
                rs2,
                imm: Immediate::Value(0),
            },
        ]
    }
}
