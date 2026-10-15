//! interp — a small stack-based bytecode VM running a Collatz step counter.

#[derive(Debug, Clone, Copy)]
enum Instr {
    Push(i64),
    Load(usize),
    Store(usize),
    Add,
    #[allow(dead_code)] // part of the instruction set; unused by this program
    Sub,
    Mul,
    Div,
    Mod,
    Lt,
    Eq,
    Jmp(usize),
    Jz(usize),
    Halt,
}

const NUM_SLOTS: usize = 8;

struct Vm {
    stack: Vec<i64>,
    locals: [i64; NUM_SLOTS],
    executed: u64,
}

impl Vm {
    fn new() -> Self {
        Vm {
            stack: Vec::with_capacity(16),
            locals: [0; NUM_SLOTS],
            executed: 0,
        }
    }

    fn pop(&mut self) -> i64 {
        self.stack.pop().expect("stack underflow")
    }

    fn binary(&mut self, op: impl FnOnce(i64, i64) -> i64) {
        let b = self.pop();
        let a = self.pop();
        self.stack.push(op(a, b));
    }

    fn run(&mut self, program: &[Instr]) {
        let mut pc = 0;
        loop {
            let instr = program[pc];
            self.executed += 1;
            pc += 1;
            match instr {
                Instr::Push(v) => self.stack.push(v),
                Instr::Load(slot) => self.stack.push(self.locals[slot]),
                Instr::Store(slot) => self.locals[slot] = self.pop(),
                Instr::Add => self.binary(|a, b| a + b),
                Instr::Sub => self.binary(|a, b| a - b),
                Instr::Mul => self.binary(|a, b| a * b),
                Instr::Div => self.binary(|a, b| a / b),
                Instr::Mod => self.binary(|a, b| a % b),
                Instr::Lt => self.binary(|a, b| i64::from(a < b)),
                Instr::Eq => self.binary(|a, b| i64::from(a == b)),
                Instr::Jmp(target) => pc = target,
                Instr::Jz(target) => {
                    if self.pop() == 0 {
                        pc = target;
                    }
                }
                Instr::Halt => return,
            }
        }
    }
}

fn program() -> Vec<Instr> {
    use Instr::*;
    vec![
        // 0: n = 1
        Push(1),
        Store(0),
        // 2: outer: if !(n < LIMIT) goto end
        Load(0),
        Push(300_000),
        Lt,
        Jz(38),
        // 6: x = n
        Load(0),
        Store(1),
        // 8: inner: if x == 1 goto next
        Load(1),
        Push(1),
        Eq,
        Jz(13),
        Jmp(33),
        // 13: if x % 2 == 0 goto even
        Load(1),
        Push(2),
        Mod,
        Jz(24),
        // 17: x = 3*x + 1
        Push(3),
        Load(1),
        Mul,
        Push(1),
        Add,
        Store(1),
        Jmp(28),
        // 24: even: x = x / 2
        Load(1),
        Push(2),
        Div,
        Store(1),
        // 28: total += 1; goto inner
        Load(2),
        Push(1),
        Add,
        Store(2),
        Jmp(8),
        // 33: next: n += 1; goto outer
        Load(0),
        Push(1),
        Add,
        Store(0),
        Jmp(2),
        // 38
        Halt,
    ]
}

fn main() {
    let program = program();
    let mut vm = Vm::new();
    vm.run(&program);
    println!("total {}", vm.locals[2]);
    println!("executed {}", vm.executed);
}
