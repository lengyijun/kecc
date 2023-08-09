use lang_c::ast::*;
use lang_c::span::Node;

use core::ops::Deref;
use std::io::{Result, Write};

use crate::write_base::*;

impl<T: WriteLine> WriteLine for Node<T> {
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        self.node.write_line(indent, write)
    }
}

impl<T: WriteString> WriteString for Node<T> {
    fn write_string(&self) -> String {
        self.node.write_string()
    }
}

impl<T: WriteString> WriteString for Box<T> {
    fn write_string(&self) -> String {
        self.deref().write_string()
    }
}

impl<T: WriteString> WriteString for &T {
    fn write_string(&self) -> String {
        (*self).write_string()
    }
}

impl<T: WriteString> WriteString for Option<T> {
    fn write_string(&self) -> String {
        if let Some(this) = self {
            this.write_string()
        } else {
            "".to_string()
        }
    }
}

impl WriteLine for TranslationUnit {
    // from youtube
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        for ext_decl in &self.0 {
            ext_decl.write_line(indent, write)?;
            writeln!(write)?;
        }
        Ok(())
    }
}

impl WriteLine for ExternalDeclaration {
    // from youtube
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        match self {
            ExternalDeclaration::Declaration(decl) => decl.write_line(indent, write),
            ExternalDeclaration::StaticAssert(_) => unreachable!(),
            ExternalDeclaration::FunctionDefinition(fdef) => fdef.write_line(indent, write),
        }
    }
}

impl WriteLine for BlockItem {
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        match self {
            BlockItem::Declaration(decl) => decl.write_line(indent, write),
            BlockItem::StaticAssert(_) => unreachable!(),
            BlockItem::Statement(stmt) => stmt.write_line(indent, write),
        }
    }
}

impl WriteLine for Declaration {
    // from youtube
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        write_indent(indent, write)?;
        writeln!(write, "{};", self.write_string())?;
        Ok(())
    }
}

impl WriteString for (&Vec<Node<DeclarationSpecifier>>, &Declarator) {
    fn write_string(&self) -> String {
        format!(
            "{} {}",
            self.0
                .iter()
                .map(WriteString::write_string)
                .collect::<Vec<_>>()
                .join(" "),
            self.1.write_string()
        )
    }
}

impl WriteLine for FunctionDefinition {
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        write_indent(indent, write)?;
        writeln!(
            write,
            "{}",
            (&self.specifiers, &self.declarator.node).write_string(),
        )?;
        self.statement.write_line(indent + 1, write)?;
        write_indent(indent, write)
    }
}

impl WriteLine for Statement {
    fn write_line(&self, indent: usize, write: &mut dyn Write) -> Result<()> {
        match self {
            Statement::Labeled(stmt) => {
                // from youtube
                write_indent(indent, write)?;
                writeln!(write, "{}", stmt.node.label.write_string())?;
                stmt.node.statement.write_line(indent + 1, write)?;
                Ok(())
            }
            Statement::Compound(items) => {
                // from youtube
                write_indent(indent, write)?;
                writeln!(write, "{{")?;

                for item in items.iter() {
                    item.write_line(indent + 1, write)?;
                }
                write_indent(indent, write)?;
                writeln!(write, "}}")?;
                Ok(())
            }
            Statement::Expression(expr) => {
                // from youtube
                write_indent(indent, write)?;
                writeln!(write, "{};", expr.as_ref().write_string())?;
                Ok(())
            }
            Statement::If(stmt) => {
                // from youtube
                write_indent(indent, write)?;
                writeln!(write, "if ({})", stmt.node.condition.write_string())?;
                write_indent(indent, write)?;
                stmt.node.then_statement.write_line(indent + 1, write)?;
                if let Some(else_stmt) = &stmt.node.else_statement {
                    write_indent(indent, write)?;
                    writeln!(write, "else")?;
                    else_stmt.write_line(indent + 1, write)?;
                }
                Ok(())
            }
            Statement::Switch(stmt) => {
                write_indent(indent, write)?;
                writeln!(write, "switch ({}) ", stmt.node.expression.write_string())?;
                stmt.node.statement.write_line(indent + 1, write)?;
                write_indent(indent, write)?;
                Ok(())
            }
            Statement::While(stmt) => {
                write_indent(indent, write)?;
                writeln!(write, "while ({})", stmt.node.expression.write_string())?;
                stmt.node.statement.write_line(indent + 1, write)?;
                Ok(())
            }
            Statement::DoWhile(stmt) => {
                write_indent(indent, write)?;
                writeln!(write, "do")?;
                stmt.node.statement.write_line(indent + 1, write)?;
                write_indent(indent, write)?;
                writeln!(write, "while({});", stmt.node.expression.write_string())?;
                Ok(())
            }
            Statement::For(stmt) => {
                write_indent(indent, write)?;
                writeln!(
                    write,
                    "for({} ; {}; {}) ",
                    stmt.node.initializer.write_string(),
                    stmt.node
                        .condition
                        .as_ref()
                        .map_or("".to_owned(), |x| x.write_string()),
                    stmt.node
                        .step
                        .as_ref()
                        .map_or("".to_owned(), |x| x.write_string())
                )?;
                write_indent(indent, write)?;
                stmt.node.statement.write_line(indent, write)?;
                Ok(())
            }
            Statement::Goto(g) => {
                write_indent(indent, write)?;
                writeln!(write, "goto {};", g.node.name)
            }
            Statement::Continue => {
                write_indent(indent, write)?;
                writeln!(write, "continue;")
            }
            Statement::Break => {
                write_indent(indent, write)?;
                writeln!(write, "break;")
            }
            Statement::Return(r) => {
                write_indent(indent, write)?;
                writeln!(
                    write,
                    "return {};",
                    r.as_ref().map_or("".to_owned(), |x| x.write_string())
                )
            }
            Statement::Asm(_) => unreachable!(),
        }
    }
}

impl WriteString for Initializer {
    fn write_string(&self) -> String {
        match self {
            Initializer::Expression(e) => e.write_string(),
            Initializer::List(v) => {
                format!(
                    "{{{}}}",
                    v.iter()
                        .map(|x| x.write_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
        }
    }
}

impl WriteString for InitializerListItem {
    fn write_string(&self) -> String {
        self.initializer.write_string()
    }
}

impl WriteString for Expression {
    fn write_string(&self) -> String {
        match self {
            Expression::Identifier(iden) => iden.write_string(),
            Expression::Constant(c) => c.write_string(),
            Expression::StringLiteral(_s) => unimplemented!(),
            Expression::Member(member) => match &member.node.operator.node {
                MemberOperator::Direct => format!(
                    "{}.{}",
                    member.node.expression.write_string(),
                    member.node.identifier.write_string()
                ),
                MemberOperator::Indirect => format!(
                    "{}->{}",
                    member.node.expression.write_string(),
                    member.node.identifier.write_string()
                ),
            },
            Expression::Call(call) => {
                format!(
                    "{}({})",
                    call.node.callee.write_string(),
                    call.node
                        .arguments
                        .iter()
                        .map(|x| x.write_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            Expression::CompoundLiteral(_) => todo!(),
            Expression::SizeOfTy(size_of_ty) => {
                format!("sizeof ({})", size_of_ty.as_ref().node.0.write_string())
            }
            Expression::SizeOfVal(size_of_val) => {
                format!("sizeof ({})", size_of_val.as_ref().node.0.write_string())
            }
            Expression::AlignOf(align_of) => {
                format!("_Alignof ({})", align_of.as_ref().node.0.write_string())
            }
            Expression::UnaryOperator(uniary_expression) => uniary_expression.write_string(),
            Expression::Cast(cast) => {
                format!(
                    "({}) ({})",
                    cast.node.type_name.write_string(),
                    cast.node.expression.write_string()
                )
            }
            Expression::BinaryOperator(binary_operation) => binary_operation.write_string(),
            Expression::Conditional(conditional) => format!(
                "({}) ? ({}) : ({})",
                conditional.as_ref().node.condition.write_string(),
                conditional.as_ref().node.then_expression.write_string(),
                conditional.as_ref().node.else_expression.write_string()
            ),
            Expression::Comma(comma) => format!(
                "({})",
                comma
                    .iter()
                    .map(WriteString::write_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Expression::Statement(_) => unreachable!(),
            Expression::GenericSelection(_) => unreachable!(),
            Expression::OffsetOf(_) => unreachable!(),
            Expression::VaArg(_) => unreachable!(),
        }
    }
}

impl WriteString for Label {
    fn write_string(&self) -> String {
        match self {
            Label::Identifier(iden) => iden.write_string(),
            Label::Case(expr) => format!("case {}:", expr.write_string()),
            Label::CaseRange(Node {
                node: CaseRange { low, high },
                ..
            }) => {
                format!("case {} ... {}:", low.write_string(), high.write_string())
            }
            Label::Default => String::from("default: "),
        }
    }
}

impl WriteString for ArraySize {
    fn write_string(&self) -> String {
        match self {
            ArraySize::Unknown => "[]".to_owned(),
            ArraySize::VariableUnknown => "[*]".to_owned(),
            ArraySize::VariableExpression(e) => format!("[{}]", e.write_string()),
            ArraySize::StaticExpression(e) => format!("[static {}]", e.write_string()),
        }
    }
}

impl WriteString for Declarator {
    fn write_string(&self) -> String {
        match &self.kind.node {
            DeclaratorKind::Abstract => {
                self.derived.iter().fold("".to_owned(), |acc, x| -> String {
                    match &x.node {
                        DerivedDeclarator::Pointer(_) => {
                            format!("*{acc}")
                        }
                        DerivedDeclarator::Array(a) => {
                            format!("{acc}{}", a.node.size.write_string())
                        }
                        DerivedDeclarator::Function(y) => {
                            let z = y
                                .node
                                .parameters
                                .iter()
                                .map(|x| x.write_string())
                                .collect::<Vec<_>>()
                                .join(",");

                            format!("{acc}({z})")
                        }
                        DerivedDeclarator::KRFunction(_) => {
                            format!("{}()", acc)
                        }
                        DerivedDeclarator::Block(_) => todo!(),
                    }
                })
            }
            DeclaratorKind::Identifier(i) => {
                self.derived
                    .iter()
                    .fold(i.write_string(), |acc, x| -> String {
                        match &x.node {
                            DerivedDeclarator::Pointer(_) => {
                                format!("*{acc}")
                            }
                            DerivedDeclarator::Array(a) => {
                                format!("{acc}{}", a.node.size.write_string())
                            }
                            DerivedDeclarator::Function(y) => {
                                let z = y
                                    .node
                                    .parameters
                                    .iter()
                                    .map(|x| x.write_string())
                                    .collect::<Vec<_>>()
                                    .join(",");

                                format!("{acc}({z})")
                            }
                            DerivedDeclarator::KRFunction(_) => {
                                format!("{}()", acc)
                            }
                            DerivedDeclarator::Block(_) => todo!(),
                        }
                    })
            }
            DeclaratorKind::Declarator(d) => {
                let x = format!("({})", d.write_string());
                self.derived
                    .iter()
                    .enumerate()
                    .fold(x, |acc, (i, x)| -> String {
                        match &x.node {
                            DerivedDeclarator::Pointer(_) => {
                                if i == 0 {
                                    format!("*{acc}")
                                } else {
                                    format!("*({acc})")
                                }
                            }
                            DerivedDeclarator::Array(a) => {
                                format!("{acc}{}", a.node.size.write_string())
                            }
                            DerivedDeclarator::Function(y) => {
                                let z = y
                                    .node
                                    .parameters
                                    .iter()
                                    .map(|x| x.write_string())
                                    .collect::<Vec<_>>()
                                    .join(",");

                                format!("{acc}({z})")
                            }
                            DerivedDeclarator::KRFunction(_) => {
                                format!("{}()", acc)
                            }
                            DerivedDeclarator::Block(_) => todo!(),
                        }
                    })
            }
        }
    }
}

impl WriteString for ParameterDeclaration {
    fn write_string(&self) -> String {
        format!(
            "{} {}",
            self.specifiers
                .iter()
                .map(|x| x.write_string())
                .collect::<Vec<_>>()
                .join(" "),
            self.declarator
                .as_ref()
                .map_or("".to_owned(), |x| x.write_string())
        )
    }
}

impl WriteString for Declaration {
    fn write_string(&self) -> String {
        format!(
            "{} {}",
            self.specifiers
                .iter()
                .map(|x| x.write_string())
                .collect::<Vec::<_>>()
                .join(" "),
            self.declarators
                .iter()
                .map(|x| x.write_string())
                .collect::<Vec::<_>>()
                .join(","),
        )
    }
}

impl WriteString for Enumerator {
    fn write_string(&self) -> String {
        let tail = self
            .expression
            .as_ref()
            .map_or("".to_owned(), |x| x.write_string());
        format!("{}{}", self.identifier.write_string(), tail)
    }
}

impl WriteString for InitDeclarator {
    fn write_string(&self) -> String {
        format!(
            "{} {}",
            self.declarator.write_string(),
            self.initializer
                .as_ref()
                .map_or("".to_owned(), |x| format!("= {}", x.write_string()))
        )
    }
}

impl WriteString for TypeSpecifier {
    fn write_string(&self) -> String {
        match self {
            TypeSpecifier::Void => "void".to_owned(),
            TypeSpecifier::Char => "char".to_owned(),
            TypeSpecifier::Short => "short".to_owned(),
            TypeSpecifier::Int => "int".to_owned(),
            TypeSpecifier::Long => "long".to_owned(),
            TypeSpecifier::Float => "float".to_owned(),
            TypeSpecifier::Double => "double".to_owned(),
            TypeSpecifier::Signed => "signed".to_owned(),
            TypeSpecifier::Unsigned => "unsigned".to_owned(),
            TypeSpecifier::Bool => "_Bool".to_owned(),
            TypeSpecifier::Complex => "_Complex".to_owned(),
            TypeSpecifier::Atomic(_) => "_Atomic".to_owned(),
            TypeSpecifier::Struct(s) => {
                let header = s
                    .node
                    .identifier
                    .as_ref()
                    .map_or("".to_owned(), |x| x.write_string());
                let fields = s.node.declarations.as_ref().map_or("".to_owned(), |x| {
                    format!(
                        "{{{}}}",
                        x.iter()
                            .map(|x| x.write_string())
                            .collect::<Vec<_>>()
                            .join(";")
                    )
                });
                match s.node.kind.node {
                    StructKind::Struct => format!("struct {header} {fields}"),
                    StructKind::Union => format!("union {header}"),
                }
            }
            TypeSpecifier::Enum(e) => {
                let header = e
                    .node
                    .identifier
                    .as_ref()
                    .map_or("".to_owned(), |x| x.write_string());
                format!(
                    "enum {header} {{ {} }}",
                    e.node
                        .enumerators
                        .iter()
                        .map(|x| x.write_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            TypeSpecifier::TypedefName(t) => t.node.name.to_owned(),
            TypeSpecifier::TypeOf(_) => unreachable!(),
            TypeSpecifier::TS18661Float(_) => unreachable!(),
        }
    }
}

impl WriteString for TypeQualifier {
    fn write_string(&self) -> String {
        match self {
            TypeQualifier::Const => "const".to_owned(),
            TypeQualifier::Restrict => unreachable!(),
            TypeQualifier::Volatile => unreachable!(),
            TypeQualifier::Nonnull => unreachable!(),
            TypeQualifier::NullUnspecified => unreachable!(),
            TypeQualifier::Nullable => unreachable!(),
            TypeQualifier::Atomic => "_Atomic".to_owned(),
        }
    }
}

impl WriteString for StorageClassSpecifier {
    fn write_string(&self) -> String {
        match self {
            StorageClassSpecifier::Typedef => "typedef".to_owned(),
            StorageClassSpecifier::Extern => "extern".to_owned(),
            StorageClassSpecifier::Static => "static".to_owned(),
            StorageClassSpecifier::ThreadLocal => "_Thread_local".to_owned(),
            StorageClassSpecifier::Auto => "auto".to_owned(),
            StorageClassSpecifier::Register => "register".to_owned(),
        }
    }
}

impl WriteString for DeclarationSpecifier {
    fn write_string(&self) -> String {
        match self {
            DeclarationSpecifier::StorageClass(s) => s.write_string(),
            DeclarationSpecifier::TypeSpecifier(t) => t.write_string(),
            DeclarationSpecifier::TypeQualifier(t) => t.write_string(),
            DeclarationSpecifier::Function(_) => unreachable!(),
            DeclarationSpecifier::Alignment(_) => unreachable!(),
            DeclarationSpecifier::Extension(_) => unreachable!(),
        }
    }
}

impl WriteString for StructDeclaration {
    fn write_string(&self) -> String {
        match self {
            StructDeclaration::Field(f) => f.write_string(),
            StructDeclaration::StaticAssert(_) => unreachable!(),
        }
    }
}

impl WriteString for StructDeclarator {
    fn write_string(&self) -> String {
        self.declarator
            .as_ref()
            .map_or("".to_owned(), |x| x.write_string())
    }
}

impl WriteString for StructField {
    fn write_string(&self) -> String {
        let a = self
            .specifiers
            .iter()
            .map(|x| x.write_string())
            .collect::<Vec<_>>()
            .join(" ");
        let b = self
            .declarators
            .iter()
            .map(|x| x.write_string())
            .collect::<Vec<_>>()
            .join(" ");
        format!("{a} {b};")
    }
}

impl WriteString for SpecifierQualifier {
    fn write_string(&self) -> String {
        match self {
            SpecifierQualifier::TypeSpecifier(t) => t.write_string(),
            SpecifierQualifier::TypeQualifier(t) => t.write_string(),
            SpecifierQualifier::Extension(_) => unreachable!(),
        }
    }
}

impl WriteString for Identifier {
    fn write_string(&self) -> String {
        self.name.to_owned()
    }
}

impl WriteString for IntegerSize {
    fn write_string(&self) -> String {
        match self {
            IntegerSize::Int => "".to_owned(),
            IntegerSize::Long => "L".to_owned(),
            IntegerSize::LongLong => "LL".to_owned(),
        }
    }
}

impl WriteString for IntegerSuffix {
    fn write_string(&self) -> String {
        format!(
            "{}{}",
            if self.unsigned { "U" } else { "" },
            self.size.write_string()
        )
    }
}

impl WriteString for FloatBase {
    fn write_string(&self) -> String {
        match self {
            FloatBase::Decimal => "".to_owned(),
            FloatBase::Hexadecimal => "0x".to_owned(),
        }
    }
}

impl WriteString for FloatFormat {
    fn write_string(&self) -> String {
        match self {
            FloatFormat::Float => "f".to_owned(),
            FloatFormat::Double => "".to_owned(),
            FloatFormat::LongDouble => "l".to_owned(),
            FloatFormat::TS18661Format(_) => unreachable!(),
        }
    }
}

impl WriteString for FloatSuffix {
    fn write_string(&self) -> String {
        self.format.write_string()
    }
}

impl WriteString for Constant {
    fn write_string(&self) -> String {
        match self {
            Constant::Integer(i) => match i.base {
                IntegerBase::Decimal => format!("{}{}", i.number, i.suffix.write_string()),
                IntegerBase::Octal => unreachable!(),
                IntegerBase::Hexadecimal => format!("0x{}{}", i.number, i.suffix.write_string()),
                IntegerBase::Binary => unreachable!(),
            },
            Constant::Float(f) => format!(
                "{}{}{}",
                f.base.write_string(),
                f.number,
                f.suffix.write_string()
            ),
            Constant::Character(c) => c.to_owned(),
        }
    }
}

impl WriteString for TypeName {
    fn write_string(&self) -> String {
        self.specifiers
            .iter()
            .map(|x| x.write_string())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl WriteString for ForInitializer {
    fn write_string(&self) -> String {
        match self {
            ForInitializer::Empty => "".to_owned(),
            ForInitializer::Expression(e) => e.write_string(),
            ForInitializer::Declaration(d) => d.write_string(),
            ForInitializer::StaticAssert(_) => unreachable!(),
        }
    }
}

impl WriteString for UnaryOperatorExpression {
    fn write_string(&self) -> String {
        match self.operator.node {
            UnaryOperator::PostIncrement => format!("{}++", self.operand.write_string()),
            UnaryOperator::PostDecrement => format!("{}--", self.operand.write_string()),
            UnaryOperator::PreIncrement => format!("++{}", self.operand.write_string()),
            UnaryOperator::PreDecrement => format!("--{}", self.operand.write_string()),
            UnaryOperator::Address => format!("&{}", self.operand.write_string()),
            UnaryOperator::Indirection => format!("*{}", self.operand.write_string()),
            UnaryOperator::Plus => format!("+({})", self.operand.write_string()),
            UnaryOperator::Minus => format!("-({})", self.operand.write_string()),
            UnaryOperator::Complement => format!("~({})", self.operand.write_string()),
            UnaryOperator::Negate => format!("!({})", self.operand.write_string()),
        }
    }
}

impl WriteString for BinaryOperatorExpression {
    fn write_string(&self) -> String {
        match self.operator.node {
            BinaryOperator::Index => {
                format!("{}[{}]", self.lhs.write_string(), self.rhs.write_string())
            }
            BinaryOperator::Multiply => {
                format!(
                    "({}) * ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::Divide => {
                format!(
                    "({}) / ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::Modulo => {
                format!(
                    "({}) % ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::Plus => {
                format!(
                    "({}) + ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::Minus => {
                format!(
                    "({}) - ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::ShiftLeft => {
                format!(
                    "({}) << ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::ShiftRight => {
                format!(
                    "({}) >> ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::Less => {
                format!(
                    "({}) < ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::Greater => {
                format!(
                    "({}) > ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::LessOrEqual => {
                format!(
                    "({}) <= ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::GreaterOrEqual => {
                format!(
                    "({}) >= ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::Equals => {
                format!(
                    "({}) == ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::NotEquals => {
                format!(
                    "({}) != ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::BitwiseAnd => {
                format!(
                    "({}) & ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::BitwiseXor => {
                format!(
                    "({}) ^ ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::BitwiseOr => {
                format!(
                    "({}) | ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::LogicalAnd => {
                format!(
                    "({}) && ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::LogicalOr => {
                format!(
                    "({}) || ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::Assign => {
                format!(
                    "{} = ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::AssignMultiply => {
                format!(
                    "{} *= ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::AssignDivide => {
                format!(
                    "{} /= ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::AssignModulo => {
                format!(
                    "{} %= ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::AssignPlus => {
                format!(
                    "{} += ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::AssignMinus => {
                format!(
                    "{} -= ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::AssignShiftLeft => format!(
                "{} <<= ({})",
                self.lhs.write_string(),
                self.rhs.write_string()
            ),
            BinaryOperator::AssignShiftRight => format!(
                "{} >>= ({})",
                self.lhs.write_string(),
                self.rhs.write_string()
            ),
            BinaryOperator::AssignBitwiseAnd => {
                format!(
                    "{} &= ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::AssignBitwiseXor => {
                format!(
                    "{} ^= ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
            BinaryOperator::AssignBitwiseOr => {
                format!(
                    "{} |= ({})",
                    self.lhs.write_string(),
                    self.rhs.write_string()
                )
            }
        }
    }
}

impl WriteString for LabeledStatement {
    // TODO: blocked by no statement.write_string()
    fn write_string(&self) -> String {
        format!("{}: ", self.label.write_string())
    }
}

impl WriteString for ForStatement {
    // TODO: blocked by no statement.write_string()
    fn write_string(&self) -> String {
        format!(
            "for({}; {}; {}) {{  }}",
            self.initializer.write_string(),
            self.condition.write_string(),
            self.step.write_string()
        )
    }
}
