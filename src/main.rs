use mips_emu::{Cpu, Mem};

struct BasicMem {
    addr: u32,
    data: Vec<u32>,
}

impl BasicMem {
    fn new(data: &[u32]) -> Self {
        Self {
            addr: 0,
            data: data.to_owned(),
        }
    }
}

impl Mem for BasicMem {
    fn addr(&mut self, addr: u32) {
        self.addr = addr - 0x400000;
    }

    fn read(&mut self) -> u32 {
        self.data[(self.addr / 4) as usize]
    }

    fn write(&mut self, data: u32) {
        self.data[(self.addr / 4) as usize] = data;
    }
}

fn main() {
    let mut c = Cpu::new();
    let mut instr = [0x0; 1 << 8];
    instr[0..(0x54 >> 2) + 1].copy_from_slice(&[
        0x23bdfff8, 0xafb00004, 0xafbf0000, 0x14800002, 0x20a20001, 0x08100012, 0x14a00004,
        0x2084ffff, 0x20050001, 0x0c100000, 0x08100012, 0x00808020, 0x20a5ffff, 0x0c100000,
        0x2204ffff, 0x00402820, 0x0c100000, 0x08100012, 0x8fb00004, 0x8fbf0000, 0x23bd0008,
        0x03e00008,
    ]);

    let m = 2;
    let n = 2;

    c.write_reg(4, m); // $a0
    c.write_reg(5, n); // $a1

    c.write_reg(31, 0x40); // $sp
    c.write_reg(29, 32 << 2); // $ra

    let mut instr_mem = BasicMem::new(&instr);
    let mut data_mem = BasicMem::new(&[0; 32]);

    while c.pc() != 0x40 {
        println!(
            "---- DATA ({:#X}, {}, {}) ----",
            c.pc(), c.cycle(), c.instr_count()
        );
        c.half_step(&mut instr_mem, &mut data_mem);
        /*
        for s in &c.pipeline {
            println!("{:?}", s);
        }
        */
        println!("$sp: {:#X}", c.read_reg(29));
        println!("regs: {:?}", (0..31).map(|r| c.read_reg(r)).collect::<Vec<_>>());
        println!(
            "data: {:?}",
            data_mem.data.iter().map(|n| *n as i32).collect::<Vec<_>>()
        );
    }

    println!("ackermann({}, {}) = {}", m, n, c.read_reg(2));
}
