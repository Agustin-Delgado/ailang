enum Tree {
    Leaf,
    Node(Box<Tree>, Box<Tree>),
}

fn make(d: i64) -> Tree {
    if d == 0 {
        Tree::Node(Box::new(Tree::Leaf), Box::new(Tree::Leaf))
    } else {
        Tree::Node(Box::new(make(d - 1)), Box::new(make(d - 1)))
    }
}

fn check(t: &Tree) -> i64 {
    match t {
        Tree::Leaf => 0,
        Tree::Node(l, r) => 1 + check(l) + check(r),
    }
}

fn main() {
    let max_depth = 20;
    let stretch = max_depth + 1;
    println!("stretch tree of depth {}\t check: {}", stretch, check(&make(stretch)));
    let long_lived = make(max_depth);
    let mut d = 4;
    while d <= max_depth {
        let iters = 1i64 << (max_depth - d + 4);
        let mut chk = 0;
        for _ in 0..iters {
            chk += check(&make(d));
        }
        println!("{}\t trees of depth {}\t check: {}", iters, d, chk);
        d += 2;
    }
    println!("long lived tree of depth {}\t check: {}", max_depth, check(&long_lived));
}
