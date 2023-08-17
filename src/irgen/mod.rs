//! # Homework: IR Generation
//!
//! The goal of this homework is to translate the components of a C file into KECC IR. While doing
//! so, you will familarize yourself with the structure of KECC IR, and understand the semantics of
//! C in terms of KECC.
//!
//! We highly recommend checking out the [slides][slides] and [github repo][github-qna-irgen] for
//! useful information.
//!
//! ## Guide
//!
//! ### High Level Guide
//!
//! Please watch the following video from 2020 along the lecture slides.
//! - [Intermediate Representation][ir]
//! - [IRgen (Overview)][irgen-overview]
//!
//! ### Coding Guide
//!
//! We highly recommend you copy-and-paste the code given in the following lecture videos from 2020:
//! - [IRgen (Code, Variable Declaration)][irgen-var-decl]
//! - [IRgen (Code, Function Definition)][irgen-func-def]
//! - [IRgen (Code, Statement 1)][irgen-stmt-1]
//! - [IRgen (Code, Statement 2)][irgen-stmt-2]
//!
//! The skeleton code roughly consists of the code for the first two videos, but you should still
//! watch them to have an idea of what the code is like.
//!
//! [slides]: https://docs.google.com/presentation/d/1SqtU-Cn60Sd1jkbO0OSsRYKPMIkul0eZoYG9KpMugFE/edit?usp=sharing
//! [ir]: https://youtu.be/7CY_lX5ZroI
//! [irgen-overview]: https://youtu.be/YPtnXlKDSYo
//! [irgen-var-decl]: https://youtu.be/HjARCUoK08s
//! [irgen-func-def]: https://youtu.be/Rszt9x0Xu_0
//! [irgen-stmt-1]: https://youtu.be/jFahkyxm994
//! [irgen-stmt-2]: https://youtu.be/UkaXaNw462U
//! [github-qna-irgen]: https://github.com/kaist-cp/cs420/labels/homework%20-%20irgen
#![allow(dead_code)]
use core::convert::TryFrom;
use core::fmt;
use core::mem;
use std::collections::{BTreeMap, HashMap};
use std::ops::Deref;

use lang_c::ast::*;
use lang_c::driver::Parse;
use lang_c::span::Node;
use thiserror::Error;

use crate::ir::{DtypeError, HasDtype, Named};
use crate::write_base::WriteString;
use crate::*;

use itertools::izip;

#[derive(Debug)]
pub struct IrgenError {
    pub code: String,
    pub message: IrgenErrorMessage,
}

impl IrgenError {
    pub fn new(code: String, message: IrgenErrorMessage) -> Self {
        Self { code, message }
    }
}

impl fmt::Display for IrgenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error: {}\r\n\r\ncode: {}", self.message, self.code)
    }
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum IrgenErrorMessage {
    /// For uncommon error
    #[error("{message}")]
    Misc { message: String },
    #[error("called object `{callee:?}` is not a function or function pointer")]
    NeedFunctionOrFunctionPointer { callee: ir::Operand },
    #[error("redefinition, `{name}`")]
    Redefinition { name: String },
    #[error("`{dtype}` conflicts prototype's dtype, `{protorype_dtype}`")]
    ConflictingDtype {
        dtype: ir::Dtype,
        protorype_dtype: ir::Dtype,
    },
    #[error("{dtype_error}")]
    InvalidDtype { dtype_error: DtypeError },
    #[error("l-value required as {message}")]
    RequireLvalue { message: String },
}

#[derive(Default, Debug)]
pub struct Irgen {
    decls: BTreeMap<String, ir::Declaration>,
    typedefs: HashMap<String, ir::Dtype>,
    structs: HashMap<String, Option<ir::Dtype>>,
    struct_tempid_counter: usize,
}

impl Translate<Parse> for Irgen {
    type Target = ir::TranslationUnit;
    type Error = IrgenError;

    fn translate(&mut self, source: &Parse) -> Result<Self::Target, Self::Error> {
        self.translate(&source.unit)
    }
}

impl Translate<TranslationUnit> for Irgen {
    type Target = ir::TranslationUnit;
    type Error = IrgenError;

    fn translate(&mut self, source: &TranslationUnit) -> Result<Self::Target, Self::Error> {
        for ext_decl in &source.0 {
            match ext_decl.node {
                ExternalDeclaration::Declaration(ref var) => {
                    self.add_declaration(&var.node)?;
                }
                ExternalDeclaration::StaticAssert(_) => {
                    panic!("ExternalDeclaration::StaticAssert is unsupported")
                }
                ExternalDeclaration::FunctionDefinition(ref func) => {
                    self.add_function_definition(&func.node)?;
                }
            }
        }

        let decls = mem::take(&mut self.decls);
        let structs = mem::take(&mut self.structs);
        Ok(Self::Target { decls, structs })
    }
}

impl Irgen {
    const BID_INIT: ir::BlockId = ir::BlockId(0);
    // `0` is used to create `BID_INIT`
    const BID_COUNTER_INIT: usize = 1;
    const TEMPID_COUNTER_INIT: usize = 0;

    /// Add a declaration. It can be either a struct, typedef, or a variable.
    fn add_declaration(&mut self, source: &Declaration) -> Result<(), IrgenError> {
        let (base_dtype, is_typedef) =
            ir::Dtype::try_from_ast_declaration_specifiers(&source.specifiers).map_err(|e| {
                IrgenError::new(
                    format!("{source:#?}"),
                    IrgenErrorMessage::InvalidDtype { dtype_error: e },
                )
            })?;
        let base_dtype = base_dtype.resolve_typedefs(&self.typedefs).map_err(|e| {
            IrgenError::new(
                format!("{source:#?}"),
                IrgenErrorMessage::InvalidDtype { dtype_error: e },
            )
        })?;

        let base_dtype = if let ir::Dtype::Struct { name, fields, .. } = &base_dtype {
            if let Some(name) = name {
                let _ = self.structs.entry(name.to_string()).or_insert(None);
            }

            if fields.is_some() {
                base_dtype
                    .resolve_structs(&mut self.structs, &mut self.struct_tempid_counter)
                    .map_err(|e| {
                        IrgenError::new(
                            format!("{source:#?}"),
                            IrgenErrorMessage::InvalidDtype { dtype_error: e },
                        )
                    })?
            } else {
                base_dtype
            }
        } else {
            base_dtype
        };

        for init_decl in &source.declarators {
            let declarator = &init_decl.node.declarator.node;
            let name = name_of_declarator(declarator);
            let dtype = base_dtype
                .clone()
                .with_ast_declarator(declarator)
                .map_err(|e| {
                    IrgenError::new(
                        format!("{source:#?}"),
                        IrgenErrorMessage::InvalidDtype { dtype_error: e },
                    )
                })?
                .deref()
                .clone();
            let dtype = dtype.resolve_typedefs(&self.typedefs).map_err(|e| {
                IrgenError::new(
                    format!("{source:#?}"),
                    IrgenErrorMessage::InvalidDtype { dtype_error: e },
                )
            })?;
            if !is_typedef && is_invalid_structure(&dtype, &self.structs) {
                return Err(IrgenError::new(
                    format!("{source:#?}"),
                    IrgenErrorMessage::Misc {
                        message: "incomplete struct type".to_string(),
                    },
                ));
            }

            if is_typedef {
                // Add new typedef if nothing has been declared before
                let prev_dtype = self
                    .typedefs
                    .entry(name.clone())
                    .or_insert_with(|| dtype.clone());

                if prev_dtype != &dtype {
                    return Err(IrgenError::new(
                        format!("{source:#?}"),
                        IrgenErrorMessage::ConflictingDtype {
                            dtype,
                            protorype_dtype: prev_dtype.clone(),
                        },
                    ));
                }

                continue;
            }

            // Creates a new declaration based on the dtype.
            let mut decl = ir::Declaration::try_from(dtype.clone()).map_err(|e| {
                IrgenError::new(
                    format!("{source:#?}"),
                    IrgenErrorMessage::InvalidDtype { dtype_error: e },
                )
            })?;

            // If `initializer` exists, convert initializer to a constant value
            if let Some(initializer) = init_decl.node.initializer.as_ref() {
                if !is_valid_initializer(&initializer.node, &dtype, &self.structs) {
                    return Err(IrgenError::new(
                        format!("{source:#?}"),
                        IrgenErrorMessage::Misc {
                            message: "initializer is not valid".to_string(),
                        },
                    ));
                }

                match &mut decl {
                    ir::Declaration::Variable {
                        initializer: var_initializer,
                        ..
                    } => {
                        if var_initializer.is_some() {
                            return Err(IrgenError::new(
                                format!("{source:#?}"),
                                IrgenErrorMessage::Redefinition { name },
                            ));
                        }
                        *var_initializer = Some(initializer.node.clone());
                    }
                    ir::Declaration::Function { .. } => {
                        return Err(IrgenError::new(
                            format!("{source:#?}"),
                            IrgenErrorMessage::Misc {
                                message: "illegal initializer (only variables can be initialized)"
                                    .to_string(),
                            },
                        ));
                    }
                }
            }

            self.add_decl(&name, decl)?;
        }

        Ok(())
    }

    /// Add a function definition.
    fn add_function_definition(&mut self, source: &FunctionDefinition) -> Result<(), IrgenError> {
        // Creates name and signature.
        let specifiers = &source.specifiers;
        let declarator = &source.declarator.node;

        let name = name_of_declarator(declarator);
        let name_of_params = name_of_params_from_function_declarator(declarator)
            .expect("declarator is not from function definition");

        let (base_dtype, is_typedef) = ir::Dtype::try_from_ast_declaration_specifiers(specifiers)
            .map_err(|e| {
            IrgenError::new(
                format!("specs: {specifiers:#?}\ndecl: {declarator:#?}"),
                IrgenErrorMessage::InvalidDtype { dtype_error: e },
            )
        })?;

        if is_typedef {
            return Err(IrgenError::new(
                format!("specs: {specifiers:#?}\ndecl: {declarator:#?}"),
                IrgenErrorMessage::Misc {
                    message: "function definition declared typedef".into(),
                },
            ));
        }

        let dtype = base_dtype
            .with_ast_declarator(declarator)
            .map_err(|e| {
                IrgenError::new(
                    format!("specs: {specifiers:#?}\ndecl: {declarator:#?}"),
                    IrgenErrorMessage::InvalidDtype { dtype_error: e },
                )
            })?
            .deref()
            .clone();
        let dtype = dtype.resolve_typedefs(&self.typedefs).map_err(|e| {
            IrgenError::new(
                format!("specs: {specifiers:#?}\ndecl: {declarator:#?}"),
                IrgenErrorMessage::InvalidDtype { dtype_error: e },
            )
        })?;

        let signature = ir::FunctionSignature::new(dtype.clone());

        // Adds new declaration if nothing has been declared before
        let decl = ir::Declaration::try_from(dtype).unwrap();
        self.add_decl(&name, decl)?;

        // Prepare scope for global variable
        let global_scope: HashMap<_, _> = self
            .decls
            .iter()
            .map(|(name, decl)| {
                let dtype = decl.dtype();
                let pointer = ir::Constant::global_variable(name.clone(), dtype);
                let operand = ir::Operand::constant(pointer);
                (name.clone(), operand)
            })
            .collect();

        // Prepares for irgen pass.
        let mut irgen = IrgenFunc {
            return_type: signature.ret.clone(),
            bid_init: Irgen::BID_INIT,
            phinodes_init: Vec::new(),
            allocations: Vec::new(),
            blocks: BTreeMap::new(),
            bid_counter: Irgen::BID_COUNTER_INIT,
            tempid_counter: Irgen::TEMPID_COUNTER_INIT,
            typedefs: &self.typedefs,
            structs: &self.structs,
            // Initial symbol table has scope for global variable already
            symbol_table: vec![global_scope],
        };
        let mut context = Context::new(irgen.bid_init);

        // Enter variable scope for alloc registers matched with function parameters
        irgen.enter_scope();

        // Creates the init block that stores arguments.
        irgen
            .translate_parameter_decl(&signature, irgen.bid_init, &name_of_params, &mut context)
            .map_err(|e| {
                IrgenError::new(format!("specs: {specifiers:#?}\ndecl: {declarator:#?}"), e)
            })?;

        // Translates statement.
        irgen.translate_stmt(&source.statement.node, &mut context, None, None)?;

        // Creates the end block
        let ret = signature.ret.set_const(false);
        let value = if ret == ir::Dtype::unit() {
            ir::Operand::constant(ir::Constant::unit())
        } else if ret == ir::Dtype::INT {
            // If "main" function, default return value is `0` when return type is `int`
            if name == "main" {
                ir::Operand::constant(ir::Constant::int(0, ret))
            } else {
                ir::Operand::constant(ir::Constant::undef(ret))
            }
        } else {
            ir::Operand::constant(ir::Constant::undef(ret))
        };

        // Last Block of the function
        irgen.insert_block(context, ir::BlockExit::Return { value });

        // Exit variable scope created above
        irgen.exit_scope();

        let func_def = ir::FunctionDefinition {
            allocations: irgen.allocations,
            blocks: irgen.blocks,
            bid_init: irgen.bid_init,
        };

        let decl = self
            .decls
            .get_mut(&name)
            .unwrap_or_else(|| panic!("The declaration of `{name}` must exist"));
        if let ir::Declaration::Function { definition, .. } = decl {
            if definition.is_some() {
                return Err(IrgenError::new(
                    format!("specs: {specifiers:#?}\ndecl: {declarator:#?}"),
                    IrgenErrorMessage::Misc {
                        message: format!("the name `{name}` is defined multiple time"),
                    },
                ));
            }

            // Update function definition
            *definition = Some(func_def);
        } else {
            panic!("`{name}` must be function declaration")
        }

        Ok(())
    }

    /// Adds a possibly existing declaration.
    ///
    /// Returns error if the previous declearation is incompatible with `decl`.
    fn add_decl(&mut self, name: &str, decl: ir::Declaration) -> Result<(), IrgenError> {
        let old_decl = some_or!(
            self.decls.insert(name.to_string(), decl.clone()),
            return Ok(())
        );

        // Check if type is conflicting for pre-declared one
        if !old_decl.is_compatible(&decl) {
            return Err(IrgenError::new(
                name.to_string(),
                IrgenErrorMessage::ConflictingDtype {
                    dtype: old_decl.dtype(),
                    protorype_dtype: decl.dtype(),
                },
            ));
        }

        Ok(())
    }
}

/// Storage for instructions up to the insertion of a block
#[derive(Debug)]
struct Context {
    /// The block id of the current context.
    bid: ir::BlockId,
    /// Current instructions of the block.
    instrs: Vec<Named<ir::Instruction>>,
}

impl Context {
    /// Create a new context with block number bid
    fn new(bid: ir::BlockId) -> Self {
        Self {
            bid,
            instrs: Vec::new(),
        }
    }

    // Adds `instr` to the current context.
    fn insert_instruction(
        &mut self,
        instr: ir::Instruction,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        let dtype = instr.dtype();
        self.instrs.push(Named::new(None, instr));

        Ok(ir::Operand::register(
            ir::RegisterId::temp(self.bid, self.instrs.len() - 1),
            dtype,
        ))
    }
}

/// A C function being translated.
struct IrgenFunc<'i> {
    /// return type of the function.
    return_type: ir::Dtype,
    /// initial block id for the function, typically 0.
    bid_init: ir::BlockId,
    /// arguments represented as initial phinodes. Order must be the same of that given in the C
    /// function.
    phinodes_init: Vec<Named<ir::Dtype>>,
    /// local allocations.
    allocations: Vec<Named<ir::Dtype>>,
    /// Map from block id to basic blocks
    blocks: BTreeMap<ir::BlockId, ir::Block>,
    /// current block id. `blocks` must have an entry for all ids less then this
    bid_counter: usize,
    /// current temporary id. Used to create temporary names in the IR for e.g,
    tempid_counter: usize,
    /// Usable definitions
    typedefs: &'i HashMap<String, ir::Dtype>,
    /// Usable structs
    // TODO: Add examples on how to use properly use this field.
    structs: &'i HashMap<String, Option<ir::Dtype>>,
    /// Current symbol table. The initial symbol table has the global variables.
    symbol_table: Vec<HashMap<String, ir::Operand>>,
}

impl IrgenFunc<'_> {
    /// Allocate a new block id.
    fn alloc_bid(&mut self) -> ir::BlockId {
        let bid = self.bid_counter;
        self.bid_counter += 1;
        ir::BlockId(bid)
    }

    /// Allocate a new temporary id.
    fn alloc_tempid(&mut self) -> String {
        let tempid = self.tempid_counter;
        self.tempid_counter += 1;
        format!("t{tempid}")
    }

    /// Create a new allocation with type given by `alloc`.
    fn insert_alloc(&mut self, alloc: Named<ir::Dtype>) -> usize {
        self.allocations.push(alloc);
        self.allocations.len() - 1
    }

    /// Insert a new block `context` with exit instruction `exit`.
    ///
    /// # Panic
    ///
    /// Panics if another block with the same bid as `context` already existed.
    fn insert_block(&mut self, context: Context, exit: ir::BlockExit) {
        let block = ir::Block {
            phinodes: if context.bid == self.bid_init {
                self.phinodes_init.clone()
            } else {
                Vec::new()
            },
            instructions: context.instrs,
            exit,
        };
        if self.blocks.insert(context.bid, block).is_some() {
            panic!("the bid `{}` is defined multiple time", context.bid)
        }
    }

    /// Enter a scope and create a new symbol table entry, i.e, we are at a `{` in the function.
    fn enter_scope(&mut self) {
        self.symbol_table.push(HashMap::new());
    }

    /// Exit a scope and remove the a oldest symbol table entry. i.e, we are at a `}` in the
    /// function.
    ///
    /// # Panic
    ///
    /// Panics if there are no scopes to exit, i.e, the function has a unmatched `}`.
    fn exit_scope(&mut self) {
        let _unused = self.symbol_table.pop().unwrap();
    }

    /// Inserts `var` with `value` to the current symbol table.
    ///
    /// Returns Ok() if the current scope has no previously-stored entry for a given variable.
    fn insert_symbol_table_entry(
        &mut self,
        var: String,
        value: ir::Operand,
    ) -> Result<(), IrgenErrorMessage> {
        let cur_scope = self
            .symbol_table
            .last_mut()
            .expect("symbol table has no valid scope");
        if cur_scope.insert(var.clone(), value).is_some() {
            return Err(IrgenErrorMessage::Redefinition { name: var });
        }

        Ok(())
    }

    /// Transalte a C statement `stmt` under the current block `context`, with `continue` block
    /// `bid_continue` and break block `bid_break`.
    fn translate_stmt(
        &mut self,
        stmt: &Statement,
        context: &mut Context,
        bid_continue: Option<ir::BlockId>,
        bid_break: Option<ir::BlockId>,
    ) -> Result<(), IrgenError> {
        match stmt {
            Statement::Compound(items) => {
                self.enter_scope();
                for item in items {
                    match &item.node {
                        BlockItem::Declaration(decl) => {
                            self.translate_decl(&decl.node, context)
                                .map_err(|e| IrgenError::new(decl.write_string(), e))?;
                        }
                        BlockItem::Statement(stmt) => {
                            self.translate_stmt(&stmt.node, context, bid_continue, bid_break)?;
                        }
                        BlockItem::StaticAssert(_) => unreachable!(),
                    }
                }
                self.exit_scope();
                Ok(())
            }
            Statement::Expression(expr) => {
                if let Some(expr) = expr {
                    let _x = self
                        .translate_expr_rvalue(&expr.node, context)
                        .map_err(|e| IrgenError::new(expr.write_string(), e))?;
                }
                Ok(())
            }
            Statement::If(stmt) => {
                let bid_then = self.alloc_bid();
                let bid_else = self.alloc_bid();
                let bid_end = self.alloc_bid();
                self.translate_condition(
                    &stmt.node.condition.node,
                    mem::replace(context, Context::new(bid_end)),
                    bid_then,
                    bid_else,
                )?;

                let mut context_then = Context::new(bid_then);
                self.translate_stmt(
                    &stmt.node.then_statement.node,
                    &mut context_then,
                    bid_continue,
                    bid_break,
                )?;
                self.insert_block(
                    context_then,
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_end, Vec::new()),
                    },
                );

                let mut context_else = Context::new(bid_else);
                if let Some(else_stmt) = &stmt.node.else_statement {
                    self.translate_stmt(
                        &else_stmt.node,
                        &mut context_else,
                        bid_continue,
                        bid_break,
                    )?;
                }
                self.insert_block(
                    context_else,
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_end, Vec::new()),
                    },
                );
                Ok(())
            }
            Statement::Return(expr) => {
                let value = match expr {
                    Some(expr) => self
                        .translate_expr_rvalue(&expr.node, context)
                        .map_err(|e| IrgenError::new(expr.write_string(), e))?,
                    None => ir::Operand::constant(ir::Constant::unit()),
                };
                let value = self
                    .translate_typecast(value, self.return_type.clone(), context)
                    .map_err(|e| IrgenError::new(expr.write_string(), e))?;
                let bid_end = self.alloc_bid();
                self.insert_block(
                    mem::replace(context, Context::new(bid_end)),
                    ir::BlockExit::Return { value },
                );
                Ok(())
            }
            Statement::For(for_stmt) => {
                let for_stmt = &for_stmt.node;

                let bid_init = self.alloc_bid();
                self.insert_block(
                    mem::replace(context, Context::new(bid_init)),
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_init, Vec::new()),
                    },
                );

                self.enter_scope();
                self.translate_for_initializer(&for_stmt.initializer.node, context)
                    .map_err(|e| IrgenError::new(for_stmt.write_string(), e))?;

                let bid_cond = self.alloc_bid();
                self.insert_block(
                    mem::replace(context, Context::new(bid_cond)),
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_cond, Vec::new()),
                    },
                );

                let bid_body = self.alloc_bid();
                let bid_step = self.alloc_bid();
                let bid_end = self.alloc_bid();

                self.translate_opt_condition(
                    &for_stmt.condition,
                    mem::replace(context, Context::new(bid_end)),
                    bid_body,
                    bid_end,
                )?;

                self.enter_scope();
                let mut context_body = Context::new(bid_body);
                self.translate_stmt(
                    &for_stmt.statement.node,
                    &mut context_body,
                    Some(bid_step),
                    Some(bid_end),
                )?;
                self.exit_scope();

                self.insert_block(
                    context_body,
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_step, Vec::new()),
                    },
                );

                let mut context_step = Context::new(bid_step);
                if let Some(step_expr) = &for_stmt.step {
                    let _x = self
                        .translate_expr_rvalue(&step_expr.node, &mut context_step)
                        .map_err(|e| IrgenError::new(step_expr.write_string(), e))?;
                }
                self.insert_block(
                    context_step,
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_cond, Vec::new()),
                    },
                );

                self.exit_scope();
                Ok(())
            }
            Statement::While(while_stmt) => {
                let while_stmt = &while_stmt.node;
                let bid_cond = self.alloc_bid();
                self.insert_block(
                    mem::replace(context, Context::new(bid_cond)),
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_cond, Vec::new()),
                    },
                );

                let bid_body = self.alloc_bid();
                let bid_end = self.alloc_bid();
                self.translate_condition(
                    &while_stmt.expression.node,
                    mem::replace(context, Context::new(bid_end)),
                    bid_body,
                    bid_end,
                )?;

                self.enter_scope();

                let mut context_body = Context::new(bid_body);
                self.translate_stmt(
                    &while_stmt.statement.node,
                    &mut context_body,
                    Some(bid_cond),
                    Some(bid_end),
                )?;
                self.insert_block(
                    context_body,
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_cond, Vec::new()),
                    },
                );
                self.exit_scope();
                Ok(())
            }
            Statement::DoWhile(while_stmt) => {
                let while_stmt = &while_stmt.node;
                let bid_body = self.alloc_bid();

                self.insert_block(
                    mem::replace(context, Context::new(bid_body)),
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_body, Vec::new()),
                    },
                );

                let bid_cond = self.alloc_bid();
                let bid_end = self.alloc_bid();

                self.enter_scope();

                self.translate_stmt(
                    &while_stmt.statement.node,
                    context,
                    Some(bid_cond),
                    Some(bid_end),
                )?;
                self.exit_scope();

                self.insert_block(
                    mem::replace(context, Context::new(bid_cond)),
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_cond, Vec::new()),
                    },
                );

                self.translate_condition(
                    &while_stmt.expression.node,
                    mem::replace(context, Context::new(bid_end)),
                    bid_body,
                    bid_end,
                )?;

                Ok(())
            }
            Statement::Switch(switch_stmt) => {
                let value = self
                    .translate_expr_rvalue(&switch_stmt.node.expression.node, context)
                    .map_err(|e| IrgenError::new(switch_stmt.node.expression.write_string(), e))?;

                let bid_end = self.alloc_bid();
                let (cases, bid_default) =
                    self.translate_switch_body(&switch_stmt.node.statement.node, bid_end)?;

                self.insert_block(
                    mem::replace(context, Context::new(bid_end)),
                    ir::BlockExit::Switch {
                        value,
                        default: ir::JumpArg::new(bid_default, Vec::new()),
                        cases,
                    },
                );

                Ok(())
            }
            Statement::Continue => {
                let bid_continue = bid_continue.ok_or_else(|| {
                    IrgenError::new(
                        "continue".to_string(),
                        IrgenErrorMessage::Misc {
                            message: "continue statement not within a loop".to_owned(),
                        },
                    )
                })?;

                let next_context = Context::new(self.alloc_bid());
                self.insert_block(
                    mem::replace(context, next_context),
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_continue, Vec::new()),
                    },
                );
                Ok(())
            }
            Statement::Break => {
                let bid_break = bid_break.ok_or_else(|| {
                    IrgenError::new(
                        "break".to_string(),
                        IrgenErrorMessage::Misc {
                            message: "break statement not within a loop".to_owned(),
                        },
                    )
                })?;

                let next_context = Context::new(self.alloc_bid());
                self.insert_block(
                    mem::replace(context, next_context),
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_break, Vec::new()),
                    },
                );
                Ok(())
            }
            Statement::Goto(_) => unreachable!(),
            Statement::Labeled(_) => unreachable!(),
            Statement::Asm(_) => unreachable!(),
        }
    }

    /// Translate parameter declaration of the functions to IR.
    ///
    /// For example, given the following C function from [`foo.c`][foo]:
    ///
    /// ```C
    /// int foo(int x, int y, int z) {
    ///    if (x == y) { return y; }
    ///    else { return z; }
    /// }
    /// ```
    ///
    /// The IR before this function looks roughly as follows:
    ///
    /// ```text
    /// fun i32 @foo (i32, i32, i32) {
    ///   init:
    ///     bid: b0
    ///     allocations:
    ///
    ///   block b0:
    ///     %b0:p0:i32:x
    ///     %b0:p1:i32:y
    ///     %b0:p2:i32:z
    /// ```
    ///
    /// With the following arguments :
    /// ```ignore
    /// signature = FunctionSignature { ret: ir::INT, params: vec![ir::INT, ir::INT, ir::INT] }
    /// bid_init = 0
    /// name_of_params = ["x", "y", "z"]
    /// context = // omitted
    ///  ```
    ///
    /// Resulting IR after this function should be roughly follows:
    /// ```text
    /// fun i32 @foo (i32, i32, i32) {
    ///   init:
    ///     bid: b0
    ///     allocations:
    ///       %l0:i32:x
    ///       %l1:i32:y
    ///       %l2:i32:z
    ///
    ///   block b0:
    ///     %b0:p0:i32:x
    ///     %b0:p1:i32:y
    ///     %b0:p2:i32:z
    ///     %b0:i0:unit = store %b0:p0:i32 %l0:i32*
    ///     %b0:i1:unit = store %b0:p1:i32 %l1:i32*
    ///     %b0:i2:unit = store %b0:p2:i32 %l2:i32*
    /// ```
    ///
    /// In particular, note that it is added to the local allocation list and store them to the
    /// initial phinodes.
    ///
    /// Note that the resulting IR is **a** solution. If you can think of a better way to
    /// translate parameters, feel free to do so.
    ///
    /// [foo]: https://github.com/kaist-cp/kecc-public/blob/main/examples/c/foo.c
    fn translate_parameter_decl(
        &mut self,
        signature: &ir::FunctionSignature,
        bid_init: ir::BlockId,
        name_of_params: &[String],
        context: &mut Context,
    ) -> Result<(), IrgenErrorMessage> {
        // from youtube
        if signature.params.len() != name_of_params.len() {
            panic!("length doesn't equal");
        }
        for (i, (dtype, var)) in izip!(&signature.params, name_of_params).enumerate() {
            let value = Some((
                ir::Operand::register(ir::RegisterId::arg(bid_init, i), dtype.clone()),
                &mut *context,
            ));
            let _o = self.translate_alloc(var.clone(), dtype.clone(), value)?;
            self.phinodes_init
                .push(Named::new(Some(var.clone()), dtype.clone()));
        }
        Ok(())
    }

    fn translate_alloc(
        &mut self,
        var: String,
        dtype: ir::Dtype,
        value: Option<(ir::Operand, &mut Context)>,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        let id = self.insert_alloc(Named::new(Some(var.clone()), dtype.clone()));

        let pointer_type = ir::Dtype::pointer(dtype.clone());
        let rid = ir::RegisterId::local(id);
        let ptr = ir::Operand::register(rid, pointer_type);
        self.insert_symbol_table_entry(var, ptr.clone())?;

        if let Some((value, context)) = value {
            let value = self.translate_typecast(value, dtype, context)?;
            let _x = context.insert_instruction(ir::Instruction::Store {
                ptr: ptr.clone(),
                value,
            })?;
        }

        Ok(ptr)
    }

    fn translate_typecast(
        &self,
        value: ir::Operand,
        target_dtype: ir::Dtype,
        context: &mut Context,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        if target_dtype == value.dtype() {
            return Ok(value);
        }
        if matches!(target_dtype, ir::Dtype::Pointer { .. }) {
            return Ok(value);
        }
        let instruction = ir::Instruction::TypeCast {
            value,
            target_dtype,
        };
        context.insert_instruction(instruction)
    }

    fn translate_typecast_to_bool(
        &self,
        condition: ir::Operand,
        context: &mut Context,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        let dtype = condition.dtype();
        if dtype == ir::Dtype::BOOL {
            return Ok(condition);
        }
        match &dtype {
            ir::Dtype::Int { .. } => context.insert_instruction(ir::Instruction::BinOp {
                op: BinaryOperator::NotEquals,
                rhs: ir::Operand::Constant(ir::Constant::int(0, dtype)),
                lhs: condition,
                dtype: ir::Dtype::BOOL,
            }),
            ir::Dtype::Float { .. } => context.insert_instruction(ir::Instruction::BinOp {
                op: BinaryOperator::NotEquals,
                rhs: ir::Operand::Constant(ir::Constant::float(0f64, dtype)),
                lhs: condition,
                dtype: ir::Dtype::BOOL,
            }),
            _ => unreachable!(),
        }
    }

    fn translate_decl(
        &mut self,
        decl: &Declaration,
        context: &mut Context,
    ) -> Result<(), IrgenErrorMessage> {
        let (base_type, is_typedef) =
            ir::Dtype::try_from_ast_declaration_specifiers(&decl.specifiers)
                .map_err(|e| IrgenErrorMessage::InvalidDtype { dtype_error: e })?;

        assert!(!is_typedef);

        for init_decl in &decl.declarators {
            let declarator = &init_decl.node.declarator.node;
            let dtype = base_type
                .clone()
                .with_ast_declarator(declarator)
                .map_err(|e| IrgenErrorMessage::InvalidDtype { dtype_error: e })?;
            let dtype = dtype
                .into_inner()
                .resolve_typedefs(self.typedefs)
                .map_err(|e| IrgenErrorMessage::InvalidDtype { dtype_error: e })?;
            let name = name_of_declarator(declarator);
            match &dtype {
                ir::Dtype::Unit { .. } => todo!(),
                ir::Dtype::Float { .. } => {
                    let default_value =
                        ir::Operand::Constant(ir::Constant::float(0f64, dtype.clone()));
                    let value = if let Some(initializer) = &init_decl.node.initializer {
                        Some((
                            self.translate_initializer(&initializer.node, context)
                                .unwrap_or(default_value.clone()),
                            &mut *context,
                        ))
                    } else {
                        Some((default_value, &mut *context))
                    };
                    let _x = self.translate_alloc(name, dtype.clone(), value)?;
                }
                ir::Dtype::Int { .. } => {
                    let default_value = ir::Operand::Constant(ir::Constant::int(0, dtype.clone()));
                    let value = if let Some(initializer) = &init_decl.node.initializer {
                        Some((
                            self.translate_initializer(&initializer.node, context)
                                .unwrap_or(default_value.clone()),
                            &mut *context,
                        ))
                    } else {
                        Some((default_value, &mut *context))
                    };
                    let _x = self.translate_alloc(name, dtype.clone(), value)?;
                }
                ir::Dtype::Pointer { .. } => {
                    let value = if let Some(initializer) = &init_decl.node.initializer {
                        Some((
                            self.translate_initializer(&initializer.node, context)?,
                            &mut *context,
                        ))
                    } else {
                        None
                    };
                    let _x = self.translate_alloc(name, dtype.clone(), value)?;
                }
                ir::Dtype::Array { .. } => {
                    let ptr = self.translate_alloc(name, dtype.clone(), None)?;
                    let inner = ptr
                        .dtype()
                        .get_pointer_inner()
                        .unwrap()
                        .get_array_inner()
                        .unwrap()
                        .clone();

                    let ptr = context.insert_instruction(ir::Instruction::GetElementPtr {
                        offset: ir::Operand::Constant(ir::Constant::int(0, ir::Dtype::INT)),
                        dtype: ir::Dtype::Pointer {
                            inner: Box::new(inner),
                            is_const: false,
                        },
                        ptr,
                    })?;
                    if let Some(initializer) = &init_decl.node.initializer {
                        match &initializer.node {
                            Initializer::Expression(e) => {
                                let value = self.translate_expr_rvalue(&e.node, context)?;
                                let _x = context
                                    .insert_instruction(ir::Instruction::Store { ptr, value })?;
                            }
                            Initializer::List(l) => {
                                self.translate_array_initializer(ptr, l, context)?;
                            }
                        }
                    }
                }
                ir::Dtype::Struct { .. } => {
                    let ptr = self.translate_alloc(name, dtype.clone(), None)?;
                    if let Some(initializer) = &init_decl.node.initializer {
                        match &initializer.node {
                            Initializer::Expression(e) => {
                                let value = self.translate_expr_rvalue(&e.node, context)?;
                                let _x = context
                                    .insert_instruction(ir::Instruction::Store { ptr, value })?;
                            }
                            Initializer::List(l) => {
                                self.translate_struct_initializer(ptr, l, context)?;
                            }
                        }
                    }
                }
                ir::Dtype::Function { .. } => todo!(),
                ir::Dtype::Typedef { .. } => unreachable!(),
            };
        }
        Ok(())
    }

    fn translate_initializer(
        &mut self,
        initializer: &Initializer,
        context: &mut Context,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        match initializer {
            Initializer::Expression(expr) => self.translate_expr_rvalue(&expr.node, context),
            Initializer::List(_) => unreachable!(),
        }
    }

    fn translate_expr_rvalue(
        &mut self,
        expr: &Expression,
        context: &mut Context,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        match expr {
            Expression::Identifier(identifier) => {
                let ptr = self.lookup_symbol_table(&identifier.node.name)?;
                let dtype_of_ptr = ptr.dtype();
                let ptr_inner_type = dtype_of_ptr.get_pointer_inner().ok_or_else(|| panic!())?;

                if ptr_inner_type.get_function_inner().is_some() {
                    return Ok(ptr);
                }
                if let Some(array_inner) = ptr_inner_type.get_array_inner() {
                    return context.insert_instruction(ir::Instruction::GetElementPtr {
                        ptr,
                        offset: ir::Operand::Constant(ir::Constant::int(0, ir::Dtype::INT)),
                        dtype: ir::Dtype::Pointer {
                            inner: Box::new(array_inner.clone()),
                            is_const: false,
                        },
                    });
                }

                context.insert_instruction(ir::Instruction::Load { ptr })
            }
            Expression::Constant(constant) => {
                let constant = ir::Constant::try_from(&constant.node).expect("must success");
                Ok(ir::Operand::constant(constant))
            }
            Expression::Call(call) => self.translate_func_call(&call.node, context),
            Expression::SizeOfTy(type_name) => {
                let dtype = ir::Dtype::try_from(&type_name.node.0.node)
                    .map_err(|e| IrgenErrorMessage::InvalidDtype { dtype_error: e })?;
                let (size_of, _) = dtype
                    .size_align_of(self.structs)
                    .map_err(|e| IrgenErrorMessage::InvalidDtype { dtype_error: e })?;
                Ok(ir::Operand::constant(ir::Constant::int(
                    size_of as u128,
                    ir::Dtype::LONG,
                )))
            }
            Expression::AlignOf(type_name) => {
                let dtype = ir::Dtype::try_from(&type_name.node.0.node)
                    .map_err(|e| IrgenErrorMessage::InvalidDtype { dtype_error: e })?;
                let (_, align_of) = dtype
                    .size_align_of(self.structs)
                    .map_err(|e| IrgenErrorMessage::InvalidDtype { dtype_error: e })?;
                Ok(ir::Operand::constant(ir::Constant::int(
                    align_of as u128,
                    ir::Dtype::LONG,
                )))
            }
            Expression::UnaryOperator(unary) => self.translate_unary_op(&unary.node, context),
            Expression::BinaryOperator(binary) => self.translate_binary_op(
                binary.node.operator.node.clone(),
                &binary.node.lhs.node,
                &binary.node.rhs.node,
                context,
            ),
            Expression::Conditional(condition) => {
                self.translate_conditional(&condition.node, context)
            }
            Expression::Cast(cast) => {
                let dtype = ir::Dtype::try_from(&cast.node.type_name.node)
                    .map_err(|e| IrgenErrorMessage::InvalidDtype { dtype_error: e })?;
                let dtype = dtype
                    .resolve_typedefs(self.typedefs)
                    .map_err(|e| IrgenErrorMessage::InvalidDtype { dtype_error: e })?;

                let operand = self.translate_expr_rvalue(&cast.node.expression.node, context)?;
                self.translate_typecast(operand, dtype, context)
            }
            Expression::Comma(exprs) => self.translate_comma(exprs, context),
            Expression::SizeOfVal(expr) => {
                let dtype = if let Expression::Identifier(iden) = &expr.node.0.node {
                    if let Some(x) = self
                        .lookup_symbol_table(&iden.node.name)
                        .unwrap()
                        .dtype()
                        .get_pointer_inner()
                    {
                        x.clone()
                    } else {
                        unreachable!()
                    }
                } else {
                    let mut context_temp = Context::new(self.alloc_bid());
                    self.translate_expr_rvalue(&expr.node.0.node, &mut context_temp)?
                        .dtype()
                };
                let (size_of, _) = dtype
                    .size_align_of(self.structs)
                    .map_err(|e| IrgenErrorMessage::InvalidDtype { dtype_error: e })?;
                Ok(ir::Operand::constant(ir::Constant::int(
                    size_of as u128,
                    ir::Dtype::Int {
                        width: 64,
                        is_signed: false,
                        is_const: false,
                    },
                )))
            }
            Expression::Member(expr) => {
                let mut ptr = self.translate_expr_lvalue(&expr.node.expression.node, context)?;
                match &expr.node.operator.node {
                    MemberOperator::Direct => {}
                    MemberOperator::Indirect => {
                        ptr = context.insert_instruction(ir::Instruction::Load { ptr })?;
                    }
                }
                let struct_dtype = ptr.dtype().get_pointer_inner().unwrap().clone();
                let identifier = &expr.node.identifier.node.name;
                let (offset, dtype) = self.struct_offset(&struct_dtype, identifier).unwrap();
                let ptr = context.insert_instruction(ir::Instruction::GetElementPtr {
                    ptr,
                    offset: ir::Operand::Constant(ir::Constant::int(
                        offset as u128,
                        ir::Dtype::LONG,
                    )),
                    dtype: ir::Dtype::Pointer {
                        inner: Box::new(dtype),
                        is_const: false,
                    },
                })?;
                match ptr.dtype().get_pointer_inner().unwrap() {
                    ir::Dtype::Int { .. } | ir::Dtype::Float { .. } => {
                        context.insert_instruction(ir::Instruction::Load { ptr })
                    }
                    ir::Dtype::Array { inner, .. } | ir::Dtype::Pointer { inner, .. } => context
                        .insert_instruction(ir::Instruction::GetElementPtr {
                            ptr,
                            offset: ir::Operand::Constant(ir::Constant::int(0u128, ir::Dtype::INT)),
                            dtype: ir::Dtype::Pointer {
                                inner: inner.clone(),
                                is_const: false,
                            },
                        }),
                    ir::Dtype::Struct { .. } => unreachable!(),
                    ir::Dtype::Unit { .. } => unreachable!(),
                    ir::Dtype::Typedef { .. } => unreachable!(),
                    ir::Dtype::Function { .. } => unreachable!(),
                }
            }
            Expression::GenericSelection(_) => unreachable!(),
            Expression::StringLiteral(_) => unreachable!(),
            Expression::CompoundLiteral(_) => unreachable!(),
            Expression::OffsetOf(_) => unreachable!(),
            Expression::VaArg(_) => unreachable!(),
            Expression::Statement(_) => unreachable!(),
        }
    }

    fn translate_expr_lvalue(
        &mut self,
        expr: &Expression,
        context: &mut Context,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        match expr {
            Expression::Identifier(identifier) => self.lookup_symbol_table(&identifier.node.name),
            Expression::UnaryOperator(unary) => match unary.node.operator.node {
                UnaryOperator::Indirection => {
                    self.translate_expr_rvalue(&unary.node.operand.node, context)
                }
                _ => Err(IrgenErrorMessage::Misc {
                    message: "translate_expr_lvalue unary operator".to_owned(),
                }),
            },
            Expression::BinaryOperator(binary) => match binary.node.operator.node {
                BinaryOperator::Index => {
                    self.translate_index_op(&binary.node.lhs.node, &binary.node.rhs.node, context)
                }
                _ => Err(IrgenErrorMessage::Misc {
                    message: "translate_expr_lvalue binary operator".to_owned(),
                }),
            },
            Expression::Member(expr) => {
                let ptr = self.translate_expr_lvalue(&expr.node.expression.node, context)?;
                let struct_dtype = if let ir::Dtype::Pointer { inner, .. } = ptr.dtype() {
                    *inner
                } else {
                    unreachable!()
                };
                let identifier = &expr.node.identifier.node.name;
                let (offset, dtype) = self.struct_offset(&struct_dtype, identifier).unwrap();
                context.insert_instruction(ir::Instruction::GetElementPtr {
                    ptr,
                    offset: ir::Operand::Constant(ir::Constant::int(
                        offset as u128,
                        ir::Dtype::LONG,
                    )),
                    dtype: ir::Dtype::Pointer {
                        inner: Box::new(dtype),
                        is_const: false,
                    },
                })
            }
            Expression::Call(_) => {
                let value = self.translate_expr_rvalue(expr, context)?;
                let var = self.alloc_tempid();
                self.translate_alloc(var, value.dtype(), Some((value, context)))
            }
            Expression::Constant(_) => unreachable!(),
            Expression::StringLiteral(_) => unreachable!(),
            Expression::GenericSelection(_) => unreachable!(),
            Expression::CompoundLiteral(_) => unreachable!(),
            Expression::SizeOfTy(_) => unreachable!(),
            Expression::SizeOfVal(_) => unreachable!(),
            Expression::AlignOf(_) => unreachable!(),
            Expression::Cast(_) => unreachable!(),
            Expression::Conditional(_) => unreachable!(),
            Expression::Comma(_) => unreachable!(),
            Expression::OffsetOf(_) => unreachable!(),
            Expression::VaArg(_) => unreachable!(),
            Expression::Statement(_) => unreachable!(),
        }
    }

    fn lookup_symbol_table(&self, name: &str) -> Result<ir::Operand, IrgenErrorMessage> {
        for hm in self.symbol_table.iter().rev() {
            if let Some(v) = hm.get(name) {
                return Ok(v.clone());
            }
        }
        Err(IrgenErrorMessage::Misc {
            message: "symbol not found".to_owned(),
        })
    }

    fn translate_for_initializer(
        &mut self,
        initializer: &ForInitializer,
        context: &mut Context,
    ) -> Result<(), IrgenErrorMessage> {
        match initializer {
            ForInitializer::Empty => Ok(()),
            ForInitializer::Expression(expr) => {
                let _x = self.translate_expr_rvalue(&expr.node, context)?;
                Ok(())
            }
            ForInitializer::Declaration(decl) => self.translate_decl(&decl.node, context),
            ForInitializer::StaticAssert(_) => unreachable!(),
        }
    }

    fn translate_opt_condition(
        &mut self,
        condition: &Option<Box<Node<Expression>>>,
        context: Context,
        bid_then: ir::BlockId,
        bid_else: ir::BlockId,
    ) -> Result<(), IrgenError> {
        if let Some(condition) = condition {
            self.translate_condition(&condition.node, context, bid_then, bid_else)
        } else {
            self.insert_block(
                context,
                ir::BlockExit::Jump {
                    arg: ir::JumpArg::new(bid_then, Vec::new()),
                },
            );
            Ok(())
        }
    }

    fn translate_condition(
        &mut self,
        condition: &Expression,
        mut context: Context,
        bid_then: ir::BlockId,
        bid_else: ir::BlockId,
    ) -> Result<(), IrgenError> {
        let condition = self
            .translate_expr_rvalue(condition, &mut context)
            .map_err(|e| IrgenError::new("error0".to_owned(), e))?;
        let condition = self
            .translate_typecast_to_bool(condition, &mut context)
            .unwrap();
        self.insert_block(
            context,
            ir::BlockExit::ConditionalJump {
                condition,
                arg_then: ir::JumpArg::new(bid_then, Vec::new()),
                arg_else: ir::JumpArg::new(bid_else, Vec::new()),
            },
        );
        Ok(())
    }

    fn translate_conditional(
        &mut self,
        conditional_expr: &ConditionalExpression,
        context: &mut Context,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        let bid_then = self.alloc_bid();
        let bid_else = self.alloc_bid();
        let bid_end = self.alloc_bid();

        self.translate_condition(
            &conditional_expr.condition.node,
            mem::replace(context, Context::new(bid_end)),
            bid_then,
            bid_else,
        )
        .unwrap();

        let mut context_then = Context::new(bid_then);
        let val_then =
            self.translate_expr_rvalue(&conditional_expr.then_expression.node, &mut context_then)?;

        let mut context_else = Context::new(bid_else);
        let val_else =
            self.translate_expr_rvalue(&conditional_expr.else_expression.node, &mut context_else)?;

        let merged_dtype = usual_arithmetic_conversions(val_then.dtype(), val_else.dtype());
        let val_then =
            self.translate_typecast(val_then, merged_dtype.clone(), &mut context_then)?;
        let val_else =
            self.translate_typecast(val_else, merged_dtype.clone(), &mut context_else)?;

        let var = self.alloc_tempid();
        let ptr = self.translate_alloc(var, merged_dtype, None)?;
        let _x = context_then
            .insert_instruction(ir::Instruction::Store {
                ptr: ptr.clone(),
                value: val_then,
            })
            .unwrap();
        self.insert_block(
            context_then,
            ir::BlockExit::Jump {
                arg: ir::JumpArg::new(bid_end, Vec::new()),
            },
        );

        let _x = context_else
            .insert_instruction(ir::Instruction::Store {
                ptr: ptr.clone(),
                value: val_else,
            })
            .unwrap();
        self.insert_block(
            context_else,
            ir::BlockExit::Jump {
                arg: ir::JumpArg::new(bid_end, Vec::new()),
            },
        );

        context.insert_instruction(ir::Instruction::Load { ptr })
    }

    fn translate_switch_body(
        &mut self,
        stmt: &Statement,
        bid_end: ir::BlockId,
    ) -> Result<(Vec<(ir::Constant, ir::JumpArg)>, ir::BlockId), IrgenError> {
        let mut cases = vec![];
        let mut default = None;

        let items = if let Statement::Compound(items) = stmt {
            items
        } else {
            panic!("don't support")
        };

        self.enter_scope();
        for item in items {
            match &item.node {
                BlockItem::Statement(stmt) => {
                    self.translate_switch_body_inner(
                        &stmt.node,
                        &mut cases,
                        &mut default,
                        bid_end,
                    )?;
                }
                _ => unreachable!(),
            }
        }
        self.exit_scope();

        let default = default.unwrap_or(bid_end);

        Ok((cases, default))
    }

    fn translate_unary_op(
        &mut self,
        expr: &UnaryOperatorExpression,
        context: &mut Context,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        match expr.operator.node {
            UnaryOperator::PostIncrement => {
                let ptr = self.translate_expr_lvalue(&expr.operand.node, context)?;
                let o1 = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let dtype = o1.dtype();
                let value = match &dtype {
                    ir::Dtype::Int { .. } => {
                        context.insert_instruction(ir::Instruction::BinOp {
                            op: BinaryOperator::Plus,
                            lhs: o1.clone(),
                            rhs: ir::Operand::Constant(ir::Constant::int(1, dtype.clone())),
                            dtype,
                        })?
                    }
                    ir::Dtype::Pointer { .. } => {
                        context.insert_instruction(ir::Instruction::GetElementPtr {
                            ptr: o1.clone(),
                            offset: ir::Operand::Constant(ir::Constant::Int {
                                value: 4,
                                width: 64,
                                is_signed: true,
                            }),
                            dtype,
                        })?
                    }
                    _ => unreachable!(),
                };
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _o3 = context.insert_instruction(ir::Instruction::Store { ptr, value })?;
                Ok(o1)
            }
            UnaryOperator::PostDecrement => {
                let ptr = self.translate_expr_lvalue(&expr.operand.node, context)?;
                let o1 = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let dtype = o1.dtype();
                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::Minus,
                    lhs: o1.clone(),
                    rhs: ir::Operand::Constant(ir::Constant::int(1, dtype.clone())),
                    dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _o3 = context.insert_instruction(ir::Instruction::Store { ptr, value })?;
                Ok(o1)
            }
            UnaryOperator::PreIncrement => {
                let ptr = self.translate_expr_lvalue(&expr.operand.node, context)?;
                let o1 = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let dtype = o1.dtype();
                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::Plus,
                    lhs: o1,
                    rhs: ir::Operand::Constant(ir::Constant::int(1, dtype.clone())),
                    dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _o3 = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            UnaryOperator::PreDecrement => {
                let ptr = self.translate_expr_lvalue(&expr.operand.node, context)?;
                let o1 = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let dtype = o1.dtype();
                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::Minus,
                    lhs: o1,
                    rhs: ir::Operand::Constant(ir::Constant::int(1, dtype.clone())),
                    dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _o3 = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            UnaryOperator::Address => self.translate_expr_lvalue(&expr.operand.node, context),
            UnaryOperator::Indirection => {
                let ptr = self.translate_expr_rvalue(&expr.operand.node, context)?;
                context.insert_instruction(ir::Instruction::Load { ptr })
            }
            UnaryOperator::Complement => {
                let lhs = self.translate_expr_rvalue(&expr.operand.node, context)?;
                let merged_dtype = integer_promotions(lhs.dtype())?;
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                match merged_dtype {
                    ir::Dtype::Int {
                        width,
                        is_signed: true,
                        ..
                    } => {
                        context.insert_instruction(
                ir::Instruction::BinOp {
                    op: BinaryOperator::BitwiseXor,
                    rhs: ir::Operand::Constant(ir::Constant::Int {
                        // -1
                        value: 0b11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
                        ,
                        width,
                        is_signed: true,
                    }),
                    dtype: lhs.dtype(),
                    lhs,
                }
                )
                    }
                    ir::Dtype::Int {
                        width,
                        is_signed: false,
                        ..
                    } => {
                        context.insert_instruction(
                ir::Instruction::BinOp {
                    op: BinaryOperator::Minus,
                    rhs: ir::Operand::Constant(ir::Constant::Int {
                        // max value in unsigned type 
                        value: 0b11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
                        ,
                        width,
                        is_signed: false,
                    }),
                    dtype: lhs.dtype(),
                    lhs,
                }
                )
                    }
                    _ => unreachable!(),
                }
            }
            UnaryOperator::Minus | UnaryOperator::Plus => {
                let operand = self.translate_expr_rvalue(&expr.operand.node, context)?;
                let dtype = operand.dtype();
                match dtype {
                    ir::Dtype::Int { .. } => {
                        let target_type = integer_promotions(dtype)?;
                        let operand = self.translate_typecast(operand, target_type, context)?;

                        let instruction = ir::Instruction::UnaryOp {
                            op: expr.operator.node.clone(),
                            operand: operand.clone(),
                            dtype: operand.dtype(),
                        };
                        context.insert_instruction(instruction)
                    }
                    ir::Dtype::Float { .. } => {
                        let instruction = ir::Instruction::UnaryOp {
                            op: expr.operator.node.clone(),
                            operand: operand.clone(),
                            dtype: operand.dtype(),
                        };
                        context.insert_instruction(instruction)
                    }
                    _ => unreachable!(),
                }
            }
            UnaryOperator::Negate => {
                let value = self.translate_expr_rvalue(&expr.operand.node, context)?;
                let operand = self.translate_typecast_to_bool(value, context)?;
                let value = context.insert_instruction(ir::Instruction::UnaryOp {
                    op: expr.operator.node.clone(),
                    operand,
                    dtype: ir::Dtype::BOOL,
                })?;
                context.insert_instruction(ir::Instruction::TypeCast {
                    value,
                    target_dtype: ir::Dtype::INT,
                })
            }
        }
    }

    fn translate_binary_op(
        &mut self,
        op: BinaryOperator,
        node_1: &Expression,
        node_2: &Expression,
        context: &mut Context,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        match op {
            BinaryOperator::Index => {
                let ptr = self.translate_expr_rvalue(node_1, context)?;

                let mut dtype = ptr.dtype();
                if let ir::Dtype::Array { inner, .. } = &dtype {
                    dtype = ir::Dtype::Pointer {
                        inner: inner.clone(),
                        is_const: false,
                    };
                }
                let (size_of, _) = dtype
                    .get_pointer_inner()
                    .unwrap()
                    .size_align_of(self.structs)
                    .unwrap();
                let offset = self.translate_expr_rvalue(node_2, context)?;
                let offset = self.translate_typecast(offset, ir::Dtype::LONG, context)?;
                let offset = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::Multiply,
                    lhs: offset,
                    rhs: ir::Operand::Constant(ir::Constant::Int {
                        value: size_of as u128,
                        width: 64,
                        is_signed: true,
                    }),
                    dtype: ir::Dtype::LONG,
                })?;

                let ptr = context.insert_instruction(ir::Instruction::GetElementPtr {
                    offset,
                    dtype,
                    ptr,
                })?;
                match ptr.dtype().get_pointer_inner().unwrap() {
                    ir::Dtype::Array { inner, .. } => {
                        context.insert_instruction(ir::Instruction::GetElementPtr {
                            ptr,
                            offset: ir::Operand::Constant(ir::Constant::int(0, ir::Dtype::INT)),
                            dtype: ir::Dtype::Pointer {
                                inner: inner.clone(),
                                is_const: false,
                            },
                        })
                    }
                    _ => context.insert_instruction(ir::Instruction::Load { ptr }),
                }
            }
            BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo
            | BinaryOperator::Plus
            | BinaryOperator::Minus
            | BinaryOperator::BitwiseAnd
            | BinaryOperator::BitwiseXor
            | BinaryOperator::BitwiseOr => {
                let lhs = self.translate_expr_rvalue(node_1, context)?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;
                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype.clone(), context)?;
                let instruction = ir::Instruction::BinOp {
                    op,
                    lhs,
                    rhs,
                    dtype: merged_dtype,
                };
                context.insert_instruction(instruction)
            }
            BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => {
                let lhs = self.translate_expr_rvalue(node_1, context)?;
                let l_dtype = integer_promotions(lhs.dtype())?;
                let lhs = self.translate_typecast(lhs, l_dtype, context)?;

                let rhs = self.translate_expr_rvalue(node_2, context)?;
                let r_dtype = integer_promotions(rhs.dtype())?;
                let rhs = self.translate_typecast(rhs, r_dtype, context)?;

                let dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, dtype.clone(), context)?;

                let instruction = ir::Instruction::BinOp {
                    dtype,
                    op,
                    lhs,
                    rhs,
                };
                context.insert_instruction(instruction)
            }
            BinaryOperator::Less
            | BinaryOperator::Greater
            | BinaryOperator::LessOrEqual
            | BinaryOperator::GreaterOrEqual
            | BinaryOperator::Equals
            | BinaryOperator::NotEquals => {
                let lhs = self.translate_expr_rvalue(node_1, context)?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;

                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype, context)?;
                let instruction = ir::Instruction::BinOp {
                    op,
                    lhs,
                    rhs,
                    dtype: ir::Dtype::BOOL,
                };
                context.insert_instruction(instruction)
            }
            BinaryOperator::LogicalAnd => {
                let var = self.alloc_tempid();
                let ptr = self.translate_alloc(var, ir::Dtype::BOOL, None)?;

                let bid_true = self.alloc_bid();
                let bid_false = self.alloc_bid();
                let bid_next = self.alloc_bid();
                let bid_end = self.alloc_bid();

                let mut context_true = Context::new(bid_true);
                let mut context_false = Context::new(bid_false);
                let mut context_next = Context::new(bid_next);
                let context_end = Context::new(bid_end);

                let lhs = self.translate_expr_rvalue(node_1, context)?;
                let lhs = self.translate_typecast_to_bool(lhs, context).unwrap();
                self.insert_block(
                    mem::replace(context, context_end),
                    ir::BlockExit::ConditionalJump {
                        condition: lhs,
                        arg_then: ir::JumpArg::new(bid_next, Vec::new()),
                        arg_else: ir::JumpArg::new(bid_false, Vec::new()),
                    },
                );

                let _x = context_true.insert_instruction(ir::Instruction::Store {
                    ptr: ptr.clone(),
                    value: ir::Operand::Constant(ir::Constant::int(1, ir::Dtype::BOOL)),
                })?;
                self.insert_block(
                    context_true,
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_end, Vec::new()),
                    },
                );

                let _x = context_false.insert_instruction(ir::Instruction::Store {
                    ptr: ptr.clone(),
                    value: ir::Operand::Constant(ir::Constant::int(0, ir::Dtype::BOOL)),
                })?;
                self.insert_block(
                    context_false,
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_end, Vec::new()),
                    },
                );

                let rhs = self.translate_expr_rvalue(node_2, &mut context_next)?;
                let rhs = self
                    .translate_typecast_to_bool(rhs, &mut context_next)
                    .unwrap();
                self.insert_block(
                    context_next,
                    ir::BlockExit::ConditionalJump {
                        condition: rhs,
                        arg_then: ir::JumpArg::new(bid_true, Vec::new()),
                        arg_else: ir::JumpArg::new(bid_false, Vec::new()),
                    },
                );

                context.insert_instruction(ir::Instruction::Load { ptr })
            }
            BinaryOperator::LogicalOr => {
                let var = self.alloc_tempid();
                let ptr = self.translate_alloc(var, ir::Dtype::BOOL, None)?;

                let bid_true = self.alloc_bid();
                let bid_false = self.alloc_bid();
                let bid_next = self.alloc_bid();
                let bid_end = self.alloc_bid();

                let mut context_true = Context::new(bid_true);
                let mut context_false = Context::new(bid_false);
                let mut context_next = Context::new(bid_next);
                let context_end = Context::new(bid_end);

                let lhs = self.translate_expr_rvalue(node_1, context)?;
                let lhs = self.translate_typecast_to_bool(lhs, context).unwrap();
                self.insert_block(
                    mem::replace(context, context_end),
                    ir::BlockExit::ConditionalJump {
                        condition: lhs,
                        arg_then: ir::JumpArg::new(bid_true, Vec::new()),
                        arg_else: ir::JumpArg::new(bid_next, Vec::new()),
                    },
                );

                let _x = context_true.insert_instruction(ir::Instruction::Store {
                    ptr: ptr.clone(),
                    value: ir::Operand::Constant(ir::Constant::int(1, ir::Dtype::BOOL)),
                })?;
                self.insert_block(
                    context_true,
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_end, Vec::new()),
                    },
                );

                let _x = context_false.insert_instruction(ir::Instruction::Store {
                    ptr: ptr.clone(),
                    value: ir::Operand::Constant(ir::Constant::int(0, ir::Dtype::BOOL)),
                })?;
                self.insert_block(
                    context_false,
                    ir::BlockExit::Jump {
                        arg: ir::JumpArg::new(bid_end, Vec::new()),
                    },
                );

                let rhs = self.translate_expr_rvalue(node_2, &mut context_next)?;
                let rhs = self
                    .translate_typecast_to_bool(rhs, &mut context_next)
                    .unwrap();
                self.insert_block(
                    context_next,
                    ir::BlockExit::ConditionalJump {
                        condition: rhs,
                        arg_then: ir::JumpArg::new(bid_true, Vec::new()),
                        arg_else: ir::JumpArg::new(bid_false, Vec::new()),
                    },
                );

                context.insert_instruction(ir::Instruction::Load { ptr })
            }
            BinaryOperator::Assign => {
                let ptr = self.translate_expr_lvalue(node_1, context)?;
                let (ir::Dtype::Pointer { inner , .. } | ir::Dtype::Array { inner, ..}) = ptr.dtype() else { return Err(IrgenErrorMessage::Misc { message: "expect ptr".to_owned() }) };
                let value = self.translate_expr_rvalue(node_2, context)?;
                let value = self.translate_typecast(value, *inner, context)?;
                let _x = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            BinaryOperator::AssignMultiply => {
                let ptr = self.translate_expr_lvalue(node_1, context)?;
                let lhs = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;

                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype.clone(), context)?;

                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::Multiply,
                    lhs,
                    rhs,
                    dtype: merged_dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _x = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            BinaryOperator::AssignDivide => {
                let ptr = self.translate_expr_lvalue(node_1, context)?;
                let lhs = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;

                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype.clone(), context)?;

                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::Divide,
                    lhs,
                    rhs,
                    dtype: merged_dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _x = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            BinaryOperator::AssignModulo => {
                let ptr = self.translate_expr_lvalue(node_1, context)?;
                let lhs = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;

                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype.clone(), context)?;

                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::Modulo,
                    lhs,
                    rhs,
                    dtype: merged_dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _x = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            BinaryOperator::AssignPlus => {
                let ptr = self.translate_expr_lvalue(node_1, context)?;
                let lhs = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;

                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype.clone(), context)?;

                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::Plus,
                    lhs,
                    rhs,
                    dtype: merged_dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _x = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            BinaryOperator::AssignMinus => {
                let ptr = self.translate_expr_lvalue(node_1, context)?;
                let lhs = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;

                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype.clone(), context)?;

                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::Minus,
                    lhs,
                    rhs,
                    dtype: merged_dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _x = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            BinaryOperator::AssignShiftLeft => {
                let ptr = self.translate_expr_lvalue(node_1, context)?;
                let lhs = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;

                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype.clone(), context)?;

                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::ShiftLeft,
                    lhs,
                    rhs,
                    dtype: merged_dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _x = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            BinaryOperator::AssignShiftRight => {
                let ptr = self.translate_expr_lvalue(node_1, context)?;
                let lhs = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;

                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype.clone(), context)?;

                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::ShiftRight,
                    lhs,
                    rhs,
                    dtype: merged_dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _x = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            BinaryOperator::AssignBitwiseAnd => {
                let ptr = self.translate_expr_lvalue(node_1, context)?;
                let lhs = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;

                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype.clone(), context)?;

                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::BitwiseAnd,
                    lhs,
                    rhs,
                    dtype: merged_dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _x = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            BinaryOperator::AssignBitwiseXor => {
                let ptr = self.translate_expr_lvalue(node_1, context)?;
                let lhs = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;

                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype.clone(), context)?;

                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::BitwiseXor,
                    lhs,
                    rhs,
                    dtype: merged_dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _x = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
            BinaryOperator::AssignBitwiseOr => {
                let ptr = self.translate_expr_lvalue(node_1, context)?;
                let lhs = context.insert_instruction(ir::Instruction::Load { ptr: ptr.clone() })?;
                let rhs = self.translate_expr_rvalue(node_2, context)?;

                let merged_dtype = usual_arithmetic_conversions(lhs.dtype(), rhs.dtype());
                let lhs = self.translate_typecast(lhs, merged_dtype.clone(), context)?;
                let rhs = self.translate_typecast(rhs, merged_dtype.clone(), context)?;

                let value = context.insert_instruction(ir::Instruction::BinOp {
                    op: BinaryOperator::BitwiseOr,
                    lhs,
                    rhs,
                    dtype: merged_dtype,
                })?;
                let value = self.translate_typecast(
                    value,
                    ptr.dtype().get_pointer_inner().unwrap().clone(),
                    context,
                )?;
                let _x = context.insert_instruction(ir::Instruction::Store {
                    ptr,
                    value: value.clone(),
                })?;
                Ok(value)
            }
        }
    }

    fn translate_func_call(
        &mut self,
        call: &CallExpression,
        context: &mut Context,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        let callee = self.translate_expr_rvalue(&call.callee.node, context)?;
        let function_pointer_type = callee.dtype();
        let function = function_pointer_type.get_pointer_inner().ok_or_else(|| {
            IrgenErrorMessage::NeedFunctionOrFunctionPointer {
                callee: callee.clone(),
            }
        })?;
        let (return_type, parameters) = function.get_function_inner().ok_or_else(|| {
            IrgenErrorMessage::NeedFunctionOrFunctionPointer {
                callee: callee.clone(),
            }
        })?;

        let args = call
            .arguments
            .iter()
            .map(|a| self.translate_expr_rvalue(&a.node, context))
            .collect::<Result<Vec<_>, _>>()?;

        if args.len() != parameters.len() {
            return Err(IrgenErrorMessage::Misc {
                message: "too few arguments".to_owned(),
            });
        }

        let args = izip!(args, parameters)
            .map(|(a, p)| self.translate_typecast(a, p.clone(), context))
            .collect::<Result<Vec<_>, _>>()?;

        let return_type = return_type.clone().set_const(false);
        context.insert_instruction(ir::Instruction::Call {
            callee,
            args,
            return_type,
        })
    }

    fn translate_comma(
        &mut self,
        exprs: &[Node<Expression>],
        context: &mut Context,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        let mut results = exprs
            .iter()
            .map(|e| self.translate_expr_rvalue(&e.node, context))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(results.pop().expect("comma expression expect expression"))
    }

    // lvalue
    fn translate_index_op(
        &mut self,
        node_1: &Expression,
        node_2: &Expression,
        context: &mut Context,
    ) -> Result<ir::Operand, IrgenErrorMessage> {
        let ptr = self.translate_expr_lvalue(node_1, context)?;
        let mut dtype = ptr.dtype().get_pointer_inner().unwrap().clone();
        if let ir::Dtype::Array { inner, .. } = &dtype {
            dtype = ir::Dtype::Pointer {
                inner: inner.clone(),
                is_const: false,
            };
        }

        // if inner is array
        // elementptr
        // else
        // load

        let ptr = match ptr.dtype().get_pointer_inner().unwrap() {
            ir::Dtype::Array { .. } => {
                context.insert_instruction(ir::Instruction::GetElementPtr {
                    offset: ir::Operand::Constant(ir::Constant::int(0, ir::Dtype::INT)),
                    dtype: dtype.clone(),
                    ptr,
                })?
            }
            _ => context.insert_instruction(ir::Instruction::Load { ptr })?,
        };

        let (size_of, _) = ptr
            .dtype()
            .get_pointer_inner()
            .unwrap()
            .size_align_of(self.structs)
            .unwrap();

        let offset = self.translate_expr_rvalue(node_2, context)?;
        let offset = self.translate_typecast(offset, ir::Dtype::LONG, context)?;
        let offset = context.insert_instruction(ir::Instruction::BinOp {
            op: BinaryOperator::Multiply,
            lhs: offset,
            rhs: ir::Operand::Constant(ir::Constant::Int {
                value: size_of as u128,
                width: 64,
                is_signed: true,
            }),
            dtype: ir::Dtype::LONG,
        })?;

        context.insert_instruction(ir::Instruction::GetElementPtr { offset, dtype, ptr })
    }

    fn translate_switch_body_inner(
        &mut self,
        stmt: &Statement,
        cases: &mut Vec<(ir::Constant, ir::JumpArg)>,
        default: &mut Option<ir::BlockId>,
        bid_end: ir::BlockId,
    ) -> Result<(), IrgenError> {
        let label_stmt = if let Statement::Labeled(label_stmt) = stmt {
            &label_stmt.node
        } else {
            unreachable!()
        };

        let bid = self.alloc_bid();
        let case = self.translate_switch_body_label_statement(label_stmt, bid, bid_end)?;

        if let Some(case) = case {
            if !case.is_integer_constant() {
                return Err(IrgenError::new(
                    label_stmt.write_string(),
                    IrgenErrorMessage::Misc {
                        message: "expression is not an integer".to_owned(),
                    },
                ));
            }

            if cases.iter().any(|(c, _)| &case == c) {
                return Err(IrgenError::new(
                    label_stmt.write_string(),
                    IrgenErrorMessage::Misc {
                        message: "duplicate case value".to_owned(),
                    },
                ));
            }
            cases.push((case, ir::JumpArg::new(bid, Vec::new())));
        } else if default.is_some() {
            return Err(IrgenError::new(
                label_stmt.write_string(),
                IrgenErrorMessage::Misc {
                    message: "previous default already exists".to_owned(),
                },
            ));
        } else {
            *default = Some(bid);
        }
        Ok(())
    }

    fn translate_switch_body_label_statement(
        &mut self,
        label_stmt: &LabeledStatement,
        bid: ir::BlockId,
        bid_end: ir::BlockId,
    ) -> Result<Option<ir::Constant>, IrgenError> {
        let case = match &label_stmt.label.node {
            Label::Case(expr) => {
                let constant = ir::Constant::try_from(&expr.node).map_err(|_| {
                    IrgenError::new(
                        expr.write_string(),
                        IrgenErrorMessage::Misc {
                            message: "case label does not reduce to an integer constant".to_owned(),
                        },
                    )
                })?;
                Some(constant)
            }
            Label::Identifier(_) => unreachable!(),
            Label::CaseRange(_) => unreachable!(),
            Label::Default => None,
        };

        let items = if let Statement::Compound(items) = &label_stmt.statement.node {
            items
        } else {
            unreachable!()
        };

        self.enter_scope();
        let (last, items) = items.split_last().expect("compound has no item");
        assert!(matches!(
            last,
            Node {
                node: BlockItem::Statement(Node {
                    node: Statement::Break,
                    ..
                }),
                ..
            }
        ));

        let mut context = Context::new(bid);
        for item in items {
            match &item.node {
                BlockItem::Declaration(decl) => {
                    self.translate_decl(&decl.node, &mut context)
                        .map_err(|e| IrgenError::new(decl.write_string(), e))?;
                }
                BlockItem::Statement(stmt) => {
                    self.translate_stmt(&stmt.node, &mut context, None, None)?;
                }
                BlockItem::StaticAssert(_) => unreachable!(),
            }
        }
        self.insert_block(
            context,
            ir::BlockExit::Jump {
                arg: ir::JumpArg::new(bid_end, Vec::new()),
            },
        );

        self.exit_scope();

        Ok(case)
    }

    fn struct_offset(
        &self,
        struct_type: &ir::Dtype,
        identifier: &str,
    ) -> Option<(usize, ir::Dtype)> {
        let ir::Dtype::Struct {
            name,
            fields,
            size_align_offsets,
            ..
        } = struct_type else { panic!("expect a struct type, meet {struct_type}") };

        if let (Some(fields), Some(size_align_offsets)) =
            (fields.as_ref(), size_align_offsets.as_ref())
        {
            let offsets = &size_align_offsets.2;
            for (field, offset) in izip!(fields, offsets) {
                match field.name() {
                    Some(name) => {
                        if name == identifier {
                            return Some((*offset, field.clone().into_inner()));
                        }
                    }
                    None => {
                        // annoymous inner struct
                        if let Some((offset_2, dtype)) = self.struct_offset(field, identifier) {
                            return Some((offset + offset_2, dtype));
                        }
                    }
                }
            }
            None
        } else if let Some(name) = name {
            let dtype = self.structs.get(name).unwrap().as_ref().unwrap();
            return self.struct_offset(dtype, identifier);
        } else {
            unreachable!()
        }
    }

    fn translate_struct_or_array_initializer(
        &mut self,
        ptr: ir::Operand,
        initializer: &Vec<Node<InitializerListItem>>,
        context: &mut Context,
    ) -> Result<(), IrgenErrorMessage> {
        let dtype = ptr.dtype();
        match &dtype {
            ir::Dtype::Pointer { inner, .. } => match &**inner {
                ir::Dtype::Struct { .. } => {
                    self.translate_struct_initializer(ptr, initializer, context)
                }
                ir::Dtype::Array { inner, .. } => {
                    let ptr = context.insert_instruction(ir::Instruction::GetElementPtr {
                        ptr,
                        offset: ir::Operand::Constant(ir::Constant::int(0, ir::Dtype::LONG)),
                        dtype: ir::Dtype::Pointer {
                            inner: inner.clone(),
                            is_const: false,
                        },
                    })?;
                    self.translate_array_initializer(ptr, initializer, context)
                }
                _ => unreachable!(),
            },

            ir::Dtype::Array { .. } => {
                let ptr = convert_array_to_pointer(ptr, context)?;
                self.translate_array_initializer(ptr, initializer, context)
            }
            _ => unreachable!("{dtype}"),
        }
    }

    fn translate_array_initializer(
        &mut self,
        ptr: ir::Operand,
        initializer: &[Node<InitializerListItem>],
        context: &mut Context,
    ) -> Result<(), IrgenErrorMessage> {
        let inner_dtype = ptr.dtype().get_pointer_inner().unwrap().clone();
        let (size_of, _) = inner_dtype.size_align_of(self.structs).unwrap();
        let dtype = ir::Dtype::Pointer {
            inner: Box::new(inner_dtype.clone()),
            is_const: false,
        };
        match &inner_dtype {
            ir::Dtype::Int { .. }
            | ir::Dtype::Float { .. }
            | ir::Dtype::Function { .. }
            | ir::Dtype::Pointer { .. } => {
                for (i, x) in initializer.iter().enumerate() {
                    let ptr = context.insert_instruction(ir::Instruction::GetElementPtr {
                        ptr: ptr.clone(),
                        offset: ir::Operand::Constant(ir::Constant::int(
                            (i * size_of) as u128,
                            ir::Dtype::LONG,
                        )),
                        dtype: dtype.clone(),
                    })?;
                    match &x.node.initializer.node {
                        Initializer::Expression(e) => {
                            let value = self.translate_expr_rvalue(&e.node, context)?;
                            let value = self.translate_typecast(
                                value,
                                ptr.dtype().get_pointer_inner().unwrap().clone(),
                                context,
                            )?;
                            let _x = context
                                .insert_instruction(ir::Instruction::Store { ptr, value })?;
                        }
                        Initializer::List(_) => {
                            unreachable!();
                        }
                    }
                }
                Ok(())
            }
            ir::Dtype::Array { .. } => {
                for (i, x) in initializer.iter().enumerate() {
                    let ptr = context.insert_instruction(ir::Instruction::GetElementPtr {
                        ptr: ptr.clone(),
                        offset: ir::Operand::Constant(ir::Constant::int(
                            (i * size_of) as u128,
                            ir::Dtype::LONG,
                        )),
                        dtype: dtype.clone(),
                    })?;
                    match &x.node.initializer.node {
                        Initializer::Expression(_) => unreachable!(),
                        Initializer::List(l) => {
                            self.translate_array_initializer(ptr, l, context)?;
                        }
                    }
                }
                Ok(())
            }
            ir::Dtype::Struct { .. } => {
                for (i, x) in initializer.iter().enumerate() {
                    let ptr = context.insert_instruction(ir::Instruction::GetElementPtr {
                        ptr: ptr.clone(),
                        offset: ir::Operand::Constant(ir::Constant::int(
                            (i * size_of) as u128,
                            ir::Dtype::INT,
                        )),
                        dtype: dtype.clone(),
                    })?;
                    match &x.node.initializer.node {
                        Initializer::Expression(_) => unreachable!(),
                        Initializer::List(l) => {
                            self.translate_struct_initializer(ptr, l, context)?;
                        }
                    }
                }
                Ok(())
            }
            ir::Dtype::Unit { .. } => unreachable!(),
            ir::Dtype::Typedef { .. } => unreachable!(),
        }
    }

    fn translate_struct_initializer(
        &mut self,
        ptr: ir::Operand,
        initializer: &Vec<Node<InitializerListItem>>,
        context: &mut Context,
    ) -> Result<(), IrgenErrorMessage> {
        let struct_dtype = ptr.dtype().get_pointer_inner().unwrap().clone();
        let struct_dtype = self
            .structs
            .get(struct_dtype.get_struct_name().unwrap().as_ref().unwrap())
            .unwrap()
            .as_ref()
            .unwrap();
        let ir::Dtype::Struct {fields : Some(fields),  size_align_offsets : Some((_, _, offsets)), ..} = struct_dtype else {panic!("expect struct")};
        for (field, &offset, init_value) in izip!(fields, offsets, initializer) {
            let dtype = ir::Dtype::Pointer {
                inner: Box::new(field.deref().clone()),
                is_const: false,
            };
            let ptr = context.insert_instruction(ir::Instruction::GetElementPtr {
                ptr: ptr.clone(),
                offset: ir::Operand::Constant(ir::Constant::int(offset as u128, ir::Dtype::LONG)),
                dtype,
            })?;
            match &init_value.node.initializer.node {
                Initializer::Expression(expr) => {
                    let value = self.translate_expr_rvalue(&expr.node, context)?;
                    let value = self.translate_typecast(
                        value,
                        ptr.dtype().get_pointer_inner().unwrap().clone(),
                        context,
                    )?;
                    let _x = context.insert_instruction(ir::Instruction::Store { ptr, value })?;
                }
                Initializer::List(x) => {
                    self.translate_struct_or_array_initializer(ptr, x, context)?
                }
            }
        }
        Ok(())
    }
}

#[inline]
fn name_of_declarator(declarator: &Declarator) -> String {
    let declarator_kind = &declarator.kind;
    match &declarator_kind.node {
        DeclaratorKind::Abstract => panic!("DeclaratorKind::Abstract is unsupported"),
        DeclaratorKind::Identifier(identifier) => identifier.node.name.clone(),
        DeclaratorKind::Declarator(declarator) => name_of_declarator(&declarator.node),
    }
}

#[inline]
fn name_of_params_from_function_declarator(declarator: &Declarator) -> Option<Vec<String>> {
    let declarator_kind = &declarator.kind;
    match &declarator_kind.node {
        DeclaratorKind::Abstract => panic!("DeclaratorKind::Abstract is unsupported"),
        DeclaratorKind::Identifier(_) => {
            name_of_params_from_derived_declarators(&declarator.derived)
        }
        DeclaratorKind::Declarator(next_declarator) => {
            name_of_params_from_function_declarator(&next_declarator.node)
                .or_else(|| name_of_params_from_derived_declarators(&declarator.derived))
        }
    }
}

#[inline]
fn name_of_params_from_derived_declarators(
    derived_decls: &[Node<DerivedDeclarator>],
) -> Option<Vec<String>> {
    for derived_decl in derived_decls {
        match &derived_decl.node {
            DerivedDeclarator::Function(func_decl) => {
                let name_of_params = func_decl
                    .node
                    .parameters
                    .iter()
                    .map(|p| name_of_parameter_declaration(&p.node))
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_default();
                return Some(name_of_params);
            }
            DerivedDeclarator::KRFunction(_kr_func_decl) => {
                // K&R function is allowed only when it has no parameter
                return Some(Vec::new());
            }
            _ => (),
        };
    }

    None
}

#[inline]
fn name_of_parameter_declaration(parameter_declaration: &ParameterDeclaration) -> Option<String> {
    let declarator = some_or!(parameter_declaration.declarator.as_ref(), return None);
    Some(name_of_declarator(&declarator.node))
}

#[inline]
fn is_valid_initializer(
    initializer: &Initializer,
    dtype: &ir::Dtype,
    structs: &HashMap<String, Option<ir::Dtype>>,
) -> bool {
    match initializer {
        Initializer::Expression(expr) => match dtype {
            ir::Dtype::Int { .. } | ir::Dtype::Float { .. } | ir::Dtype::Pointer { .. } => {
                match &expr.node {
                    Expression::Constant(_) => true,
                    Expression::UnaryOperator(unary) => matches!(
                        &unary.node.operator.node,
                        UnaryOperator::Minus | UnaryOperator::Plus
                    ),
                    _ => false,
                }
            }
            _ => false,
        },
        Initializer::List(items) => match dtype {
            ir::Dtype::Array { inner, .. } => items
                .iter()
                .all(|i| is_valid_initializer(&i.node.initializer.node, inner, structs)),
            ir::Dtype::Struct { name, .. } => {
                let name = name.as_ref().expect("struct should have its name");
                let struct_type = structs
                    .get(name)
                    .expect("struct type matched with `name` must exist")
                    .as_ref()
                    .expect("`struct_type` must have its definition");
                let fields = struct_type
                    .get_struct_fields()
                    .expect("`struct_type` must be struct type")
                    .as_ref()
                    .expect("`fields` must be `Some`");

                izip!(fields, items).all(|(f, i)| {
                    is_valid_initializer(&i.node.initializer.node, f.deref(), structs)
                })
            }
            _ => false,
        },
    }
}

#[inline]
fn is_invalid_structure(dtype: &ir::Dtype, structs: &HashMap<String, Option<ir::Dtype>>) -> bool {
    // When `dtype` is `Dtype::Struct`, `structs` has real definition of `dtype`
    if let ir::Dtype::Struct { name, fields, .. } = dtype {
        assert!(name.is_some() && fields.is_none());
        let name = name.as_ref().unwrap();
        let struct_type = some_or!(structs.get(name), return true);

        struct_type.is_none()
    } else {
        false
    }
}

fn integer_promotions(int_type: ir::Dtype) -> Result<ir::Dtype, IrgenErrorMessage> {
    match int_type {
        ir::Dtype::Int { width, .. } => {
            if width < 32 {
                Ok(ir::Dtype::INT)
            } else {
                Ok(int_type)
            }
        }
        _ => Err(IrgenErrorMessage::Misc {
            message: format!("expect int, meet {int_type}"),
        }),
    }
}

fn usual_arithmetic_conversions(dtype_1: ir::Dtype, dtype_2: ir::Dtype) -> ir::Dtype {
    match (
        matches!(dtype_1, ir::Dtype::Float { width: 64, .. }),
        matches!(dtype_2, ir::Dtype::Float { width: 64, .. }),
    ) {
        (false, false) => {}
        _ => return ir::Dtype::DOUBLE,
    };
    match (
        matches!(dtype_1, ir::Dtype::Float { width: 32, .. }),
        matches!(dtype_2, ir::Dtype::Float { width: 32, .. }),
    ) {
        (false, false) => {}
        _ => return ir::Dtype::FLOAT,
    };

    let int_1 = integer_promotions(dtype_1).unwrap();
    let int_2 = integer_promotions(dtype_2).unwrap();

    if int_1 == int_2 {
        return int_1;
    }
    let ir::Dtype::Int{ width : width_1, is_signed: is_signed_1, ..} = int_1 else { unreachable!()};
    let ir::Dtype::Int{ width : width_2, is_signed: is_signed_2, ..} = int_2 else { unreachable!()};

    if is_signed_1 == is_signed_2 {
        return ir::Dtype::Int {
            width: Ord::max(width_1, width_2),
            is_signed: is_signed_1,
            is_const: false,
        };
    }

    match (width_1.cmp(&width_2), is_signed_1, is_signed_2) {
        (_, true, true) | (_, false, false) => unreachable!(),
        (std::cmp::Ordering::Less | std::cmp::Ordering::Equal, true, false) => int_2,
        (std::cmp::Ordering::Less, false, true) => int_2,
        (std::cmp::Ordering::Greater, true, false) => int_1,
        (std::cmp::Ordering::Greater | std::cmp::Ordering::Equal, false, true) => int_1,
    }
}

fn convert_array_to_pointer(
    ptr: ir::Operand,
    context: &mut Context,
) -> Result<ir::Operand, IrgenErrorMessage> {
    let dtype = ptr.dtype().get_array_inner().unwrap().clone();
    context.insert_instruction(ir::Instruction::GetElementPtr {
        ptr,
        offset: ir::Operand::Constant(ir::Constant::int(0, ir::Dtype::LONG)),
        dtype: ir::Dtype::Pointer {
            inner: Box::new(dtype),
            is_const: false,
        },
    })
}
