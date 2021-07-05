#![deny(clippy::pedantic)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![no_std]

extern crate alloc;
use alloc::{vec, vec::Vec};

pub trait Mem {
    fn addr(&mut self, addr: u32);
    fn read(&mut self) -> u32;
    fn write(&mut self, data: u32);
}

fn read_bits(value: u32, offset: usize, bits: usize) -> u32 {
    (value >> offset) & ((1 << bits) - 1)
}

#[derive(Debug, Clone)]
enum AluOp {
    Nop,
    Add,
    AddU,
    Sub,
    SubU,
    And,
    Nor,
    Or,
    Xor,
    LT,
    LTU,
    SLL,
    SRA,
    SRL,
}

#[derive(Debug, Clone)]
enum BranchOn {
    Always,
    Z,
    GEZ,
    NZ,
    GTZ,
    LTZ,
    Never,
}

#[derive(Debug, Clone)]
enum BranchTarget {
    Reg,
    Rel,
    Abs,
    AbsUpper,
    None,
}

#[derive(Debug, Clone)]
enum MemOp {
    Load(MemSize),
    Store(MemSize),
    None,
}

#[derive(Debug, Clone)]
enum MemSize {
    Byte,
    ByteU,
    HalfWord,
    HalfWordU,
    Word,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
struct Ctrl {
    shift_amount: bool,
    immediate: bool,
    op: AluOp,
    branch_on: BranchOn,
    branch_target: BranchTarget,
    mem_op: MemOp,
    reg_write: bool,
    link: bool,
    rd: usize,
}

#[rustfmt::skip]
impl Ctrl {
    fn decode_r(instr: u32) -> Self {
        let funct = read_bits(instr, 0, 6) as usize;
        let rd = read_bits(instr, 11, 5) as usize;

        match funct {
            /* ADD  */ 0b100000 => Self::new_alu_op(false, false, AluOp::Add, rd),
            /* ADDU */ 0b100001 => Self::new_alu_op(false, false, AluOp::AddU, rd),
            /* AND  */ 0b100100 => Self::new_alu_op(false, false, AluOp::And, rd),
            /* NOR  */ 0b100111 => Self::new_alu_op(false, false, AluOp::Nor, rd),
            /* OR   */ 0b100101 => Self::new_alu_op(false, false, AluOp::Or, rd),
            /* SLT  */ 0b101010 => Self::new_alu_op(false, false, AluOp::LT, rd),
            /* SLTU */ 0b101011 => Self::new_alu_op(false, false, AluOp::LTU, rd),
            /* SUB  */ 0b100010 => Self::new_alu_op(false, false, AluOp::Sub, rd),
            /* SUBU */ 0b100011 => Self::new_alu_op(false, false, AluOp::SubU, rd),
            /* XOR  */ 0b100110 => Self::new_alu_op(false, false, AluOp::Xor, rd),

            /* SLL  */ 0b000000 => Self::new_alu_op(true, false, AluOp::SLL, rd),
            /* SLLV */ 0b000100 => Self::new_alu_op(false, false, AluOp::SLL, rd),
            /* SRA  */ 0b000011 => Self::new_alu_op(true, false, AluOp::SRA, rd),
            /* SRAV */ 0b000111 => Self::new_alu_op(false, false, AluOp::SRA, rd),
            /* SRL  */ 0b000010 => Self::new_alu_op(true, false, AluOp::SRL, rd),
            /* SRLV */ 0b000110 => Self::new_alu_op(false, false, AluOp::SRL, rd),

            /* DIV   */
            /* DIVU  */
            /* MFHI  */
            /* MFLO  */
            /* MTHI  */ 0b010001 => Self::new_alu_op(false, false, AluOp::AddU, 34),
            /* MTLO  */ 0b010011 => Self::new_alu_op(false, false, AluOp::AddU, 33),
            /* MULT  */
            /* MULTU */

            /* JALR */ 0b001001 => Self::new_branch_op(BranchOn::Always, BranchTarget::Reg, true),
            /* JR   */ 0b001000 => Self::new_branch_op(BranchOn::Always, BranchTarget::Reg, false),
            _ => unimplemented!("funct: {:#08b}; instr: {:#010X}", funct, instr),
        }
    }

   fn decode_branch(instr: u32) -> Self {
        let rt = read_bits(instr, 16, 5);

        match rt {
            /* BGEZ   */ 0b00001 => Self::new_branch_op(BranchOn::GEZ, BranchTarget::Rel, false),
            /* BGEZAL */ 0b10001 => Self::new_branch_op(BranchOn::GEZ, BranchTarget::Rel, true),
            /* BLTZ   */ 0b00000 => Self::new_branch_op(BranchOn::LTZ, BranchTarget::Rel, false),
            /* BLTZAL */ 0b10000 => Self::new_branch_op(BranchOn::LTZ, BranchTarget::Rel, true),
            _ => unimplemented!("rt: {:#07b}; instr: {:#010X}", rt, instr),
        }
    }

    fn decode(instr: u32) -> Self {
        let op = read_bits(instr, 26, 6) as usize;
        let rt = read_bits(instr, 16, 5) as usize;

        match op {
            /* ALU   */ 0b000000 => Self::decode_r(instr),
            /* ADDI  */ 0b001000 => Self::new_alu_op(false, true, AluOp::Add, rt),
            /* ADDIU */ 0b001001 => Self::new_alu_op(false, true, AluOp::AddU, rt),
            /* ANDI  */ 0b001100 => Self::new_alu_op(false, true, AluOp::And, rt),
            /* ORI   */ 0b001101 => Self::new_alu_op(false, true, AluOp::Or, rt),
            /* SLTI  */ 0b001010 => Self::new_alu_op(false, true, AluOp::LT, rt),
            /* SLTIU */ 0b001011 => Self::new_alu_op(false, true, AluOp::LTU, rt),
            /* XORI  */ 0b001110 => Self::new_alu_op(false, true, AluOp::Xor, rt),

            /* BEQ  */ 0b000100 => Self::new_branch_op(BranchOn::Z, BranchTarget::Rel, false),
            /* B    */ 0b000001 => Self::decode_branch(instr),
            /* BGTZ */ 0b000111 => Self::new_branch_op(BranchOn::GTZ, BranchTarget::Rel, false),
            /* BLEZ */ 0b000110 => Self::new_branch_op(BranchOn::LTZ, BranchTarget::Rel, false),
            /* BNE  */ 0b000101 => Self::new_branch_op(BranchOn::NZ, BranchTarget::Rel, false),

            /* BREAK */

            /* J   */ 0b000010 => Self::new_branch_op(BranchOn::Always, BranchTarget::AbsUpper, false),
            /* JAL */ 0b000011 => Self::new_branch_op(BranchOn::Always, BranchTarget::Abs, true),

            /* MFCO */
            /* MTCO */
            /* SYSCALL */

            /* LB  */ 0b100000 => Self::new_mem_op(true, MemOp::Load(MemSize::Byte), rt),
            /* LBU */ 0b100100 => Self::new_mem_op(true, MemOp::Load(MemSize::ByteU), rt),
            /* LH  */ 0b100001 => Self::new_mem_op(true, MemOp::Load(MemSize::HalfWord), rt),
            /* LHU */ 0b100101 => Self::new_mem_op(true, MemOp::Load(MemSize::HalfWordU), rt),
            /* LW  */ 0b100011 => Self::new_mem_op(true, MemOp::Load(MemSize::Word), rt),
            /* SB  */ 0b101000 => Self::new_mem_op(false, MemOp::Store(MemSize::Byte), rt),
            /* SH  */ 0b101001 => Self::new_mem_op(false, MemOp::Store(MemSize::HalfWord), rt),
            /* SW  */ 0b101011 => Self::new_mem_op(false, MemOp::Store(MemSize::Word), rt),
            _ => unimplemented!("opcode: {:#08b}; instr: {:#010X}", op, instr),
        }
    }

    fn new_alu_op(shift_amount: bool, immediate: bool, op: AluOp, rd: usize) -> Self {
        Self {
            shift_amount,
            immediate,
            op,
            branch_on: BranchOn::Never,
            branch_target: BranchTarget::None,
            mem_op: MemOp::None,
            reg_write: true,
            link: false,
            rd,
        }
    }

    fn new_branch_op(branch_on: BranchOn, branch_target: BranchTarget, link: bool) -> Self {
        Self {
            shift_amount: false,
            immediate: false,
            op: AluOp::Nop,
            branch_on,
            branch_target,
            mem_op: MemOp::None,
            reg_write: false,
            link,
            rd: 0,
        }
    }

    fn new_mem_op(load: bool, mem_op: MemOp, rd: usize) -> Self {
        Self {
            shift_amount: false,
            immediate: true,
            op: AluOp::AddU,
            branch_on: BranchOn::Never,
            branch_target: BranchTarget::None,
            mem_op,
            reg_write: load,
            link: false,
            rd,
        }
    }
}

#[derive(Debug, Clone)]
struct IF {
    cycle: usize,
}

impl IF {
    fn next(self, cpu: &mut Cpu, mem: &mut impl Mem) -> PipelineStage {
        match self.cycle {
            0 => {
                mem.addr(cpu.pc);
                cpu.pc += 4;
                PipelineStage::IF(IF { cycle: 1 })
            }
            _ => PipelineStage::ID(ID {
                cycle: 0,
                pc_4: cpu.pc,
                instr: mem.read(),
            }),
        }
    }
}

#[derive(Debug, Clone)]
struct ID {
    cycle: usize,
    pc_4: u32,
    instr: u32,
}

impl ID {
    fn next(self, cpu: &mut Cpu) -> PipelineStage {
        match self.cycle {
            0 => PipelineStage::ID(ID { cycle: 1, ..self }),
            _ => DH {
                cycle: 1,
                instr: self.instr,
                pc_4: self.pc_4,
                ctrl: Ctrl::decode(self.instr),
            }
            .next(cpu),
        }
    }
}

// Data Hazard: used for stalling in case of data dependencies
#[derive(Debug, Clone)]
struct DH {
    cycle: usize,
    instr: u32,
    pc_4: u32,
    ctrl: Ctrl,
}

impl DH {
    fn next(self, cpu: &mut Cpu) -> PipelineStage {
        if self.cycle == 0 {
            PipelineStage::DH(DH { cycle: 1, ..self })
        } else {
            let rs = read_bits(self.instr, 21, 5) as usize;
            let rt = read_bits(self.instr, 16, 5) as usize;

            if cpu.is_free(rs) && cpu.is_free(rt) {
                cpu.use_reg(self.ctrl.rd);

                PipelineStage::EX(EX {
                    cycle: 0,
                    a: cpu.read_reg(rs),
                    b: cpu.read_reg(rt),
                    imm: read_bits(self.instr, 0, 16) as u32,
                    sa: read_bits(self.instr, 6, 5) as u32,
                    target: read_bits(self.instr, 0, 26) as u32,
                    pc_4: self.pc_4,
                    ctrl: self.ctrl,
                })
            } else {
                PipelineStage::DH(DH { cycle: 0, ..self })
            }
        }
    }
}

#[derive(Debug, Clone)]
struct EX {
    cycle: usize,
    a: u32,
    b: u32,
    imm: u32,
    sa: u32,
    target: u32,
    pc_4: u32,
    ctrl: Ctrl,
}

impl EX {
    fn next(self) -> PipelineStage {
        if self.cycle == 0 {
            PipelineStage::EX(EX { cycle: 1, ..self })
        } else {
            let a = if false { self.sa } else { self.a };
            let b = if self.ctrl.immediate {
                match self.ctrl.op {
                    AluOp::AddU | AluOp::SubU => self.imm,
                    _ => ((self.imm as i32) << 16 >> 16) as u32,
                }
            } else {
                self.b
            };

            let alu_output = match self.ctrl.op {
                AluOp::Nop => a,
                AluOp::Add => (a as i32 + b as i32) as u32,
                AluOp::AddU => a + b,
                AluOp::Sub => (a as i32 - b as i32) as u32,
                AluOp::SubU => a - b,
                AluOp::And => a & b,
                AluOp::Nor => !(a | b),
                AluOp::Or => a | b,
                AluOp::Xor => a ^ b,
                AluOp::LT => ((a as i32) < (b as i32)) as u32,
                AluOp::LTU => (a < b) as u32,
                AluOp::SLL => b << a,
                AluOp::SRL => b >> a,
                AluOp::SRA => ((b as i32) << a) as u32,
            };

            let target = match self.ctrl.branch_target {
                BranchTarget::Reg => a,
                BranchTarget::Rel => self.pc_4 + (self.imm << 2),
                BranchTarget::Abs => self.target << 2,
                BranchTarget::AbsUpper => self.pc_4 & 0xFC000000 | self.target << 2,
                BranchTarget::None => 0,
            };

            PipelineStage::MEM(MEM {
                cycle: 0,
                d: alu_output,
                b: self.b,
                target,
                pc_4: self.pc_4,
                ctrl: self.ctrl,
            })
        }
    }
}

#[derive(Debug, Clone)]
struct MEM {
    cycle: usize,
    d: u32,
    b: u32,
    target: u32,
    pc_4: u32,
    ctrl: Ctrl,
}

impl MEM {
    fn next(self, cpu: &mut Cpu, mem: &mut impl Mem) -> PipelineStage {
        if self.cycle == 0 {
            mem.addr(self.d);
            PipelineStage::MEM(MEM { cycle: 1, ..self })
        } else {
            let pc_sel = match self.ctrl.branch_on {
                BranchOn::Always => true,
                BranchOn::Z => self.d == 0,
                BranchOn::NZ => self.d != 0,
                BranchOn::GEZ => self.d as i32 >= 0,
                BranchOn::GTZ => self.d as i32 > 0,
                BranchOn::LTZ => (self.d as i32) < 0,
                BranchOn::Never => false,
            };

            if pc_sel {
                cpu.pc = self.target;
            }

            if let MemOp::Store(ref s) = self.ctrl.mem_op {
                let data = match s {
                    MemSize::Byte | MemSize::ByteU => self.b as u8 as u32,
                    MemSize::HalfWord | MemSize::HalfWordU => self.b as u16 as u32,
                    MemSize::Word => self.b,
                };

                mem.write(data);
            }

            let m = if let MemOp::Load(_) = self.ctrl.mem_op {
                mem.read()
            } else {
                0
            };

            PipelineStage::WB(WB {
                cycle: 0,
                d: self.d,
                m,
                jump_taken: pc_sel,
                pc_4: self.pc_4,
                ctrl: self.ctrl,
            })
        }
    }
}

#[derive(Debug, Clone)]
struct WB {
    cycle: usize,
    d: u32,
    m: u32,
    jump_taken: bool,
    pc_4: u32,
    ctrl: Ctrl,
}

impl WB {
    fn next(self, cpu: &mut Cpu) -> PipelineStage {
        match self.cycle {
            0 => {
                cpu.free_reg(self.ctrl.rd);

                if self.ctrl.link {
                    cpu.write_reg(31, self.pc_4);
                }

                if self.ctrl.reg_write {
                    cpu.write_reg(
                        self.ctrl.rd,
                        if let MemOp::Load(ref s) = self.ctrl.mem_op {
                            match s {
                                MemSize::Byte => self.m as i8 as i32 as u32,
                                MemSize::ByteU => self.m as u8 as u32,
                                MemSize::HalfWord => self.m as i16 as i32 as u32,
                                MemSize::HalfWordU => self.m as u16 as u32,
                                MemSize::Word => self.m,
                            }
                        } else {
                            self.d
                        },
                    );
                }

                PipelineStage::WB(WB { cycle: 1, ..self })
            }
            _ => PipelineStage::Done,
        }
    }
}

#[derive(Debug, Clone)]
enum PipelineStage {
    IF(IF),
    ID(ID),
    DH(DH),
    EX(EX),
    MEM(MEM),
    WB(WB),
    Done,
}

impl PipelineStage {
    fn new() -> Self {
        PipelineStage::IF(IF { cycle: 0 })
    }

    fn next(self, cpu: &mut Cpu, instr: &mut impl Mem, data: &mut impl Mem) -> PipelineStage {
        match self {
            PipelineStage::IF(s) => s.next(cpu, instr),
            PipelineStage::ID(s) => s.next(cpu),
            PipelineStage::DH(s) => s.next(cpu),
            PipelineStage::EX(s) => s.next(),
            PipelineStage::MEM(s) => s.next(cpu, data),
            PipelineStage::WB(s) => s.next(cpu),
            PipelineStage::Done => PipelineStage::Done,
        }
    }
}

#[derive(Debug)]
pub struct Cpu {
    half_cycle: u32,
    instr_count: u32,
    regs: [u32; 34],
    pc: u32,
    regs_in_use: u32,
    pipeline: Vec<PipelineStage>,
}

impl Cpu {
    #[must_use]
    pub fn new() -> Self {
        Self {
            half_cycle: 0,
            instr_count: 0,
            regs: [0; 34],
            pc: 0x400000,
            regs_in_use: 0,
            pipeline: vec![PipelineStage::new()],
        }
    }

    #[must_use]
    pub fn read_reg(&self, reg: usize) -> u32 {
        self.regs[reg]
    }

    pub fn write_reg(&mut self, reg: usize, value: u32) {
        if reg != 0 {
            self.regs[reg] = value;
        }
    }

    pub fn half_step(&mut self, instr: &mut impl Mem, data: &mut impl Mem) {
        let mut data_hazard = false;
        let mut control_hazard = false;
        for s in self.pipeline.drain(0..).collect::<Vec<_>>() {
            let p = if data_hazard {
                s
            } else {
                s.next(self, instr, data)
            };

            // stop execution of future instructions
            // if theres a data hazard
            if let PipelineStage::DH(_) = p {
                data_hazard = true;
            }

            // drop every instruction after a branch taken
            //
            // since instructions are executed from the
            // back of the pipeline first, dropping instructions
            // here means the next instruction's MEM won't be
            // executed, causing no side effects
            if let PipelineStage::WB(WB { jump_taken, .. }) = p {
                control_hazard = jump_taken;
            }

            match p {
                PipelineStage::Done => self.instr_count += 1,
                _ => self.pipeline.push(p),
            }

            if control_hazard {
                self.regs_in_use = 0;
                break;
            }
        }

        self.half_cycle += 1;

        if !data_hazard && self.half_cycle & 1 == 0 {
            self.pipeline.push(PipelineStage::new());
        }
    }

    pub fn exec_instr(&mut self, instr: &mut impl Mem, data: &mut impl Mem) {
        PipelineStage::new()
            .next(self, instr, data)
            .next(self, instr, data)
            .next(self, instr, data)
            .next(self, instr, data)
            .next(self, instr, data);
    }

    #[must_use]
    pub fn pc(&self) -> u32 {
        self.pc
    }

    #[must_use]
    pub fn cycle(&self) -> u32 {
        self.half_cycle / 2
    }

    #[must_use]
    pub fn instr_count(&self) -> u32 {
        self.instr_count
    }

    fn use_reg(&mut self, reg: usize) {
        self.regs_in_use |= 1 << reg;
    }

    fn free_reg(&mut self, reg: usize) {
        self.regs_in_use &= !(1 << reg);
    }

    fn is_free(&self, reg: usize) -> bool {
        reg == 0 || (self.regs_in_use & (1 << reg)) == 0
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}
