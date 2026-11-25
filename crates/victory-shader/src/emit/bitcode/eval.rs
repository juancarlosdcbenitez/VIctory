//! An interpreter of the module under construction, for tests: it runs
//! `main` for one thread, or for the lanes of one wave in lockstep, so the
//! lowering of an instruction can be checked against a reference on any
//! machine, without a GPU or `dxil.dll`.
//!
//! It runs the symbolic module (before value numbering). Where DXIL leaves
//! a choice to the driver, it makes one, and it refuses what LLVM leaves
//! undefined, so a lowering that depends on either shows up:
//!
//! * `FMax`/`FMin` of +0 and -0 give the first operand (a compare and
//!   select), a NaN operand the other one;
//! * `FMad` rounds the product (unfused);
//! * float-to-integer conversions saturate and give 0 for NaN (D3D's rule,
//!   which the hardware follows);
//! * a shift by the bit width or more, an integer division by zero, a
//!   branch on an undefined value and a store of one are errors;
//! * reads of lanes outside the wave (quad and wave reads) give undefined
//!   values, which are errors only where they are used;
//! * lanes of a wave all run every instruction: a branch the lanes take
//!   different ways is an error (divergence is not modelled), so wave
//!   operations see every lane active.
//!
//! Supported: integer and float arithmetic, casts, compares, selects, phis,
//! branches, group-shared memory and constant tables, and the `dx.op`
//! calls of compute code without textures: math, user data and root
//! constant buffers, raw and structured buffers (heap and root SRVs),
//! buffer atomics, thread ids and single-lane wave operations.

use std::collections::HashMap;

use super::{Bin, Cast, Const, Module, Op, Rmw, Term, Ty, TypeDef, V, pred};

/// A runtime value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Val {
    /// An integer of `bits` bits, zero-extended.
    Int(u32, u64),
    F32(u32),
    Handle(Handle),
    Agg(Vec<Val>),
    /// Element `index` of global `global` (`None`: the global itself).
    Ptr(u32, Option<u64>),
    /// Undefined (`undef`, or another lane's value).
    Undef,
}

impl Val {
    fn bool(b: bool) -> Val {
        Val::Int(1, u64::from(b))
    }
    fn i32(x: u32) -> Val {
        Val::Int(32, u64::from(x))
    }
}

/// A resource handle: where its descriptor came from and its annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Handle {
    pub place: Place,
    /// `annotateHandle`'s two property words.
    pub props: Option<(u32, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Place {
    /// `createHandleFromBinding`: class (0 SRV, 1 UAV, 2 CBV, 3 sampler),
    /// space and register.
    Binding {
        class: u8,
        space: u32,
        register: u32,
    },
    /// `createHandleFromHeap`.
    Heap { index: u32, sampler: bool },
}

/// What the thread sees: its ids and the resources bound.
#[derive(Debug, Clone)]
pub(crate) struct Env {
    /// The user-data constant buffer (`b0, space 0`), in dwords.
    pub user_data: Vec<u32>,
    /// Root CBVs (`b<n>, space 1`) by register, in dwords.
    pub root_cbvs: HashMap<u32, Vec<u32>>,
    /// Root SRVs (`t<n>, space 1`, raw buffers) by register.
    pub root_srvs: HashMap<u32, Vec<u8>>,
    /// Buffers in the CBV/SRV/UAV heap by index, as bytes.
    pub heap: HashMap<u32, Vec<u8>>,
    /// Of those, the typed texture UAVs: width, height, and depth or
    /// layers. Their texels (4 bytes, or 8 for a 64-bit component type) lie
    /// in `heap` row by row, slice by slice.
    pub textures: HashMap<u32, [u32; 3]>,
    /// The probe constants (`b0, space 2`): the view and sampler heaps'
    /// slot counts.
    pub probe_slots: [u32; 2],
    /// The probe buffer (`u0, space 2`), as bytes.
    pub probe: Vec<u8>,
    pub group_size: [u32; 3],
    pub group: [u32; 3],
    /// The id in its group of the wave's first thread; lane n is n further
    /// along x.
    pub thread: [u32; 3],
    /// Lanes in the host wave (`WaveGetLaneCount`).
    pub wave_lanes: u32,
    /// Lanes run in lockstep (at most `wave_lanes`): 1 runs one thread.
    pub lanes: u32,
    /// Buffer accesses out of bounds (D3D defines them: reads give 0,
    /// writes are dropped), counted.
    pub out_of_bounds: u32,
    /// Basic blocks run before the run is stopped as endless.
    pub max_blocks: u64,
}

impl Default for Env {
    fn default() -> Self {
        Env {
            user_data: Vec::new(),
            root_cbvs: HashMap::new(),
            root_srvs: HashMap::new(),
            heap: HashMap::new(),
            textures: HashMap::new(),
            probe_slots: [0, 0],
            probe: Vec::new(),
            group_size: [1, 1, 1],
            group: [0, 0, 0],
            thread: [0, 0, 0],
            wave_lanes: 32,
            lanes: 1,
            out_of_bounds: 0,
            max_blocks: 1 << 24,
        }
    }
}

/// Runs `main` of `m` for the thread `env` describes; buffer writes land
/// in `env`.
pub(crate) fn run(m: &Module, env: &mut Env) -> Result<(), String> {
    let globals = m
        .globals
        .iter()
        .map(|g| match g.init {
            Some(c) => match constant(m, c) {
                Val::Agg(elems) => Ok(elems),
                v => Ok(vec![v]),
            },
            None => match &m.types[g.ty.0 as usize] {
                // Group-shared memory starts undefined.
                TypeDef::Array(n, _) => Ok(vec![Val::Undef; *n as usize]),
                t => Err(format!("a global of type {t:?}")),
            },
        })
        .collect::<Result<Vec<_>, String>>()?;
    let lanes = env.lanes.max(1) as usize;
    if lanes as u32 > env.wave_lanes {
        return Err(format!("{lanes} lanes in a wave of {}", env.wave_lanes));
    }
    let mut x = Exec {
        m,
        env,
        vals: vec![vec![None; m.insts.len()]; lanes],
        globals,
        lane: 0,
    };
    x.main()
}

/// The value of module constant `i`.
fn constant(m: &Module, i: u32) -> Val {
    let (ty, c) = &m.consts[i as usize];
    match c {
        Const::Int(v) => match m.types[ty.0 as usize] {
            TypeDef::Int(bits) => Val::Int(bits, mask(bits, *v as u64)),
            _ => Val::Undef,
        },
        Const::Float(bits) => Val::F32(*bits),
        Const::Undef => Val::Undef,
        Const::Aggregate(ids) => Val::Agg(ids.iter().map(|&i| constant(m, i)).collect()),
    }
}

fn mask(bits: u32, v: u64) -> u64 {
    if bits >= 64 {
        v
    } else {
        v & ((1u64 << bits) - 1)
    }
}

fn sext(bits: u32, v: u64) -> i64 {
    if bits >= 64 {
        v as i64
    } else {
        ((v << (64 - bits)) as i64) >> (64 - bits)
    }
}

/// `f32` to binary16 bits, rounded to nearest even: what GFX10's f16
/// conversions and f16 arithmetic do in the MODE register's default
/// rounding, so the reference the tests hold f16 results to. Not what
/// D3D's `f32tof16` does: see [`f32_to_f16_d3d`].
pub(crate) fn f32_to_f16_nearest_even(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let mant = x & 0x7F_FFFF;
    if exp == 0xFF {
        // Infinity, or a NaN that stays one (quiet, payload kept).
        return sign
            | 0x7C00
            | if mant != 0 {
                0x200 | (mant >> 13) as u16
            } else {
                0
            };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        // A half denormal or zero: shift the full mantissa into place.
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rest > half || (rest == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1FFF;
    if rest > 0x1000 || (rest == 0x1000 && r & 1 == 1) {
        // A carry into the exponent is right, up to infinity.
        r += 1;
    }
    sign | r as u16
}

/// `f32` to binary16 bits as D3D's `f32tof16` (DXIL `legacyF32ToF16`)
/// converts them. The D3D11.3 functional specification's entry (22.13.2)
/// "Follows D3D rules for floating point conversion", which (3.2.2) are:
/// round toward zero; a value too large becomes the largest finite value
/// of its sign, "NOT including signed infinity"; infinity stays infinity;
/// NaN stays NaN (its bits are not specified, 3.1.3.2: kept quiet with the
/// payload's top bits here); a denormal becomes the half denormal it
/// truncates to, or zero.
pub(crate) fn f32_to_f16_d3d(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let ax = x & 0x7FFF_FFFF;
    if ax > 0x7F80_0000 {
        return sign | 0x7E00 | ((ax >> 13) & 0x3FF) as u16;
    }
    if ax == 0x7F80_0000 {
        return sign | 0x7C00;
    }
    if ax >= 0x4780_0000 {
        return sign | 0x7BFF;
    }
    let exp = ax >> 23;
    if exp < 113 {
        // Below 2^-14: a half denormal (counts of 2^-24), or zero.
        if exp < 103 {
            return sign;
        }
        let m = (ax & 0x7F_FFFF) | 0x80_0000;
        return sign | (m >> (126 - exp)) as u16;
    }
    sign | ((ax >> 13) - (112 << 10)) as u16
}

/// Binary16 bits to `f32`, exactly.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1F);
    let mant = u32::from(h & 0x3FF);
    let bits = match exp {
        0 if mant == 0 => sign,
        0 => {
            // Denormal: normalise.
            let shift = mant.leading_zeros() - 21;
            let m = (mant << shift) & 0x3FF;
            sign | ((113 - shift) << 23) | (m << 13)
        }
        0x1F => sign | 0x7F80_0000 | (mant << 13),
        _ => sign | ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

struct Exec<'m, 'e> {
    m: &'m Module,
    env: &'e mut Env,
    /// Per lane, each instruction's value.
    vals: Vec<Vec<Option<Val>>>,
    globals: Vec<Vec<Val>>,
    /// The lane an instruction is evaluated for.
    lane: usize,
}

impl Exec<'_, '_> {
    fn main(&mut self) -> Result<(), String> {
        let (mut block, mut prev) = (0usize, None);
        let lanes = self.vals.len();
        for _ in 0..self.env.max_blocks {
            let blk = &self.m.blocks[block];
            // Phis read their incoming values all at once.
            for lane in 0..lanes {
                self.lane = lane;
                let mut phis = Vec::with_capacity(blk.phis.len());
                for &i in &blk.phis {
                    let Op::Phi(incoming) = &self.m.insts[i as usize].op else {
                        unreachable!()
                    };
                    let v = match incoming.iter().find(|(_, b)| Some(*b) == prev) {
                        Some(&(v, _)) => self.value(v)?,
                        // A phi nothing reads may lack an edge.
                        None => Val::Undef,
                    };
                    phis.push((i, v));
                }
                for (i, v) in phis {
                    self.vals[lane][i as usize] = Some(v);
                }
            }
            for &i in &blk.body {
                if let Some(results) = self.across_lanes(i)? {
                    for (lane, v) in results.into_iter().enumerate() {
                        self.vals[lane][i as usize] = Some(v);
                    }
                    continue;
                }
                for lane in 0..lanes {
                    self.lane = lane;
                    let v = self.inst(i)?;
                    self.vals[lane][i as usize] = Some(v);
                }
            }
            prev = Some(block);
            block = match blk.term {
                Some(Term::Ret) => return Ok(()),
                Some(Term::Br(t)) => t,
                Some(Term::CondBr(c, t, f)) => {
                    let mut to = None;
                    for lane in 0..lanes {
                        self.lane = lane;
                        let next = match self.value(c)? {
                            Val::Int(1, x) => {
                                if x != 0 {
                                    t
                                } else {
                                    f
                                }
                            }
                            v => return Err(format!("a branch on {v:?}")),
                        };
                        if to.is_some_and(|b| b != next) {
                            return Err("lanes branch different ways (divergence)".into());
                        }
                        to = Some(next);
                    }
                    to.expect("a lane")
                }
                None => return Err(format!("block {block} has no terminator")),
            };
        }
        Err("no return within the block limit".into())
    }

    /// A wave operation (it reads other lanes): every lane's result, all
    /// lanes active. `None` for any other instruction.
    fn across_lanes(&mut self, i: u32) -> Result<Option<Vec<Val>>, String> {
        let Op::Call(f, args) = &self.m.insts[i as usize].op else {
            return Ok(None);
        };
        let Some(&V::Const(code)) = args.first() else {
            return Ok(None);
        };
        let Val::Int(32, opcode) = constant(self.m, code) else {
            return Ok(None);
        };
        if !self.m.funcs[*f as usize].name.starts_with("dx.op.") || !(113..=123).contains(&opcode) {
            return Ok(None);
        }
        let lanes = self.vals.len();
        let mut per_lane = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            self.lane = lane;
            let a = args[1..]
                .iter()
                .map(|&a| self.value(a))
                .collect::<Result<Vec<_>, _>>()?;
            per_lane.push(a);
        }
        let x = |l: usize| per_lane[l][0].clone();
        let small = |l: usize, k: usize| match per_lane[l].get(k) {
            Some(Val::Int(_, v)) => Ok(*v),
            v => Err(format!("wave operation {opcode}: operand {k} is {v:?}")),
        };
        let undefined = per_lane.iter().any(|a| a[0] == Val::Undef);
        let all = |v: Val| vec![v; lanes];
        // Lanes past the wave read nothing defined.
        let pick = |l: usize| if l < lanes { x(l) } else { Val::Undef };
        Ok(Some(match opcode {
            // WaveAnyTrue, WaveAllTrue.
            113 | 114 => {
                if undefined {
                    all(Val::Undef)
                } else {
                    let set: Vec<bool> = (0..lanes).map(|l| x(l) == Val::bool(true)).collect();
                    let r = if opcode == 113 {
                        set.iter().any(|&b| b)
                    } else {
                        set.iter().all(|&b| b)
                    };
                    all(Val::bool(r))
                }
            }
            // WaveActiveAllEqual.
            115 => all(Val::bool((0..lanes).all(|l| x(l) == x(0)))),
            // WaveActiveBallot: a bit per lane in four dwords.
            116 => {
                if undefined {
                    all(Val::Undef)
                } else {
                    let mut words = [0u32; 4];
                    for (l, word) in (0..lanes).map(|l| (l, l / 32)) {
                        if x(l) == Val::bool(true) {
                            words[word] |= 1 << (l % 32);
                        }
                    }
                    all(Val::Agg(words.iter().map(|&w| Val::i32(w)).collect()))
                }
            }
            // WaveReadLaneAt.
            117 => (0..lanes)
                .map(|l| Ok(pick(small(l, 1)? as usize)))
                .collect::<Result<Vec<_>, String>>()?,
            // WaveReadLaneFirst.
            118 => all(x(0)),
            // WaveActiveOp (sum, product, min, max; signed or unsigned),
            // WaveActiveBit (and, or, xor), WavePrefixOp (exclusive).
            119..=121 => {
                if undefined {
                    return Ok(Some(all(Val::Undef)));
                }
                let op = small(0, 1)?;
                let unsigned = opcode != 120 && small(0, 2)? == 1;
                let combine = |a: &Val, b: &Val| -> Result<Val, String> {
                    Ok(match (a, b) {
                        (Val::Int(n, a), Val::Int(_, b)) => {
                            let r = match (opcode, op) {
                                (120, 0) => a & b,
                                (120, 1) => a | b,
                                (120, _) => a ^ b,
                                (_, 0) => a.wrapping_add(*b),
                                (_, 1) => a.wrapping_mul(*b),
                                _ => {
                                    let less = if unsigned {
                                        a < b
                                    } else {
                                        sext(*n, *a) < sext(*n, *b)
                                    };
                                    if less == (op == 2) { *a } else { *b }
                                }
                            };
                            Val::Int(*n, mask(*n, r))
                        }
                        (Val::F32(a), Val::F32(b)) if opcode != 120 => {
                            let (a, b) = (f32::from_bits(*a), f32::from_bits(*b));
                            let r = match op {
                                0 => a + b,
                                1 => a * b,
                                2 => a.min(b),
                                _ => a.max(b),
                            };
                            Val::F32(r.to_bits())
                        }
                        (a, b) => return Err(format!("wave operation {opcode} of {a:?}, {b:?}")),
                    })
                };
                if opcode == 121 {
                    // Each lane gets the lanes below it, after the identity.
                    let mut acc = match (x(0), op) {
                        (Val::Int(n, _), 0) => Val::Int(n, 0),
                        (Val::Int(n, _), 1) => Val::Int(n, 1),
                        (Val::F32(_), 0) => Val::F32(0f32.to_bits()),
                        (Val::F32(_), 1) => Val::F32(1f32.to_bits()),
                        (v, op) => return Err(format!("a wave prefix {op} of {v:?}")),
                    };
                    let mut out = Vec::with_capacity(lanes);
                    for l in 0..lanes {
                        out.push(acc.clone());
                        acc = combine(&acc, &x(l))?;
                    }
                    out
                } else {
                    let mut acc = x(0);
                    for l in 1..lanes {
                        acc = combine(&acc, &x(l))?;
                    }
                    all(acc)
                }
            }
            // QuadReadLaneAt; QuadOp (across x, across y, diagonal).
            122 | 123 => (0..lanes)
                .map(|l| {
                    let from = if opcode == 122 {
                        (l & !3) + (small(l, 1)? as usize & 3)
                    } else {
                        l ^ (small(l, 1)? as usize + 1)
                    };
                    Ok(pick(from))
                })
                .collect::<Result<Vec<_>, String>>()?,
            _ => unreachable!(),
        }))
    }

    fn value(&self, v: V) -> Result<Val, String> {
        Ok(match v {
            V::Const(i) => constant(self.m, i),
            V::Inst(i) => self.vals[self.lane][i as usize]
                .clone()
                .ok_or_else(|| format!("instruction {i} read before it ran"))?,
            V::Global(g) => Val::Ptr(g, None),
            V::Func(_) => return Err("a function used as a value".into()),
        })
    }

    fn ty(&self, t: Ty) -> &TypeDef {
        &self.m.types[t.0 as usize]
    }

    fn inst(&mut self, i: u32) -> Result<Val, String> {
        let inst = &self.m.insts[i as usize];
        let ty = inst.ty;
        Ok(match &inst.op {
            Op::Bin(op, a, b) => bin(*op, self.value(*a)?, self.value(*b)?)?,
            Op::Cast(op, v) => self.cast(*op, self.value(*v)?, ty)?,
            Op::Cmp(p, a, b) => cmp(*p, self.value(*a)?, self.value(*b)?)?,
            Op::Select(c, t, f) => {
                let (t, f) = (self.value(*t)?, self.value(*f)?);
                match self.value(*c)? {
                    Val::Int(1, x) => {
                        if x != 0 {
                            t
                        } else {
                            f
                        }
                    }
                    Val::Undef if t == f => t,
                    Val::Undef => Val::Undef,
                    v => return Err(format!("a select on {v:?}")),
                }
            }
            Op::Extract(a, k) => match self.value(*a)? {
                Val::Agg(elems) => elems
                    .get(*k as usize)
                    .cloned()
                    .ok_or("extractvalue out of range")?,
                Val::Undef => Val::Undef,
                v => return Err(format!("extractvalue from {v:?}")),
            },
            Op::Call(f, args) => {
                let args = args
                    .iter()
                    .map(|&a| self.value(a))
                    .collect::<Result<Vec<_>, _>>()?;
                let name = self.m.funcs[*f as usize].name.clone();
                self.call(&name, &args, ty)?
            }
            Op::Phi(_) => return Err("a phi in a block body".into()),
            Op::Gep(_, p, idx) => match (self.value(*p)?, &idx[..]) {
                (Val::Ptr(g, None), [zero, index]) => {
                    if self.value(*zero)? != Val::i32(0) {
                        return Err("getelementptr with a non-zero first index".into());
                    }
                    match self.value(*index)? {
                        Val::Int(_, x) => Val::Ptr(g, Some(x)),
                        v => return Err(format!("an element index of {v:?}")),
                    }
                }
                (p, _) => return Err(format!("getelementptr from {p:?}")),
            },
            Op::Load(p) => {
                let (g, e) = self.element(self.value(*p)?)?;
                self.globals[g][e].clone()
            }
            Op::Store(p, v) => {
                let v = self.value(*v)?;
                let (g, e) = self.element(self.value(*p)?)?;
                self.global_writable(g)?;
                self.globals[g][e] = v;
                Val::Undef
            }
            Op::AtomicRmw(op, p, v) => {
                let v = self.value(*v)?;
                let (g, e) = self.element(self.value(*p)?)?;
                self.global_writable(g)?;
                let old = self.globals[g][e].clone();
                self.globals[g][e] = rmw(*op, &old, &v)?;
                old
            }
            Op::CmpXchg(p, expected, new) => {
                let (expected, new) = (self.value(*expected)?, self.value(*new)?);
                let (g, e) = self.element(self.value(*p)?)?;
                self.global_writable(g)?;
                let old = self.globals[g][e].clone();
                if matches!(old, Val::Undef) {
                    return Err("a compare-exchange of undefined memory".into());
                }
                let swapped = old == expected;
                if swapped {
                    self.globals[g][e] = new;
                }
                Val::Agg(vec![old, Val::bool(swapped)])
            }
        })
    }

    fn element(&self, p: Val) -> Result<(usize, usize), String> {
        match p {
            Val::Ptr(g, Some(e)) => {
                let g = g as usize;
                let n = self.globals[g].len() as u64;
                if e >= n {
                    return Err(format!(
                        "element {e} of global {} ({n} elements)",
                        self.m.globals[g].name
                    ));
                }
                Ok((g, e as usize))
            }
            p => Err(format!("an access through {p:?}")),
        }
    }

    fn global_writable(&self, g: usize) -> Result<(), String> {
        if self.m.globals[g].init.is_some() {
            return Err(format!(
                "a write to the constant {}",
                self.m.globals[g].name
            ));
        }
        Ok(())
    }

    fn cast(&self, op: Cast, v: Val, to: Ty) -> Result<Val, String> {
        let to = self.ty(to).clone();
        if v == Val::Undef {
            return Ok(Val::Undef);
        }
        Ok(match (op, v, to) {
            (Cast::Trunc | Cast::ZExt, Val::Int(_, x), TypeDef::Int(b)) => Val::Int(b, mask(b, x)),
            (Cast::SExt, Val::Int(a, x), TypeDef::Int(b)) => {
                Val::Int(b, mask(b, sext(a, x) as u64))
            }
            (Cast::FpToUi, Val::F32(f), TypeDef::Int(b)) => {
                let f = f32::from_bits(f);
                // Saturating, NaN to 0 (Rust's `as` is D3D's rule).
                let x = if b <= 32 {
                    u64::from(f as u32)
                } else {
                    f as u64
                };
                Val::Int(b, mask(b, x))
            }
            (Cast::FpToSi, Val::F32(f), TypeDef::Int(b)) => {
                let f = f32::from_bits(f);
                let x = if b <= 32 {
                    i64::from(f as i32)
                } else {
                    f as i64
                };
                Val::Int(b, mask(b, x as u64))
            }
            (Cast::UiToFp, Val::Int(_, x), TypeDef::Float) => Val::F32((x as f32).to_bits()),
            (Cast::SiToFp, Val::Int(a, x), TypeDef::Float) => {
                Val::F32((sext(a, x) as f32).to_bits())
            }
            (Cast::Bitcast, Val::Int(32, x), TypeDef::Float) => Val::F32(x as u32),
            (Cast::Bitcast, Val::F32(f), TypeDef::Int(32)) => Val::i32(f),
            (op, v, to) => return Err(format!("{op:?} of {v:?} to {to:?}")),
        })
    }

    fn call(&mut self, name: &str, args: &[Val], ret: Ty) -> Result<Val, String> {
        let Some(class) = name.strip_prefix("dx.op.") else {
            return Err(format!("a call of {name}"));
        };
        let Some(Val::Int(32, opcode)) = args.first() else {
            return Err(format!("{name} without an opcode"));
        };
        let a = &args[1..];
        let f = |k: usize| -> Result<f32, String> {
            match a.get(k) {
                Some(Val::F32(x)) => Ok(f32::from_bits(*x)),
                v => Err(format!("{name}: operand {k} is {v:?}")),
            }
        };
        let u = |k: usize| -> Result<u32, String> {
            match a.get(k) {
                Some(Val::Int(32, x)) => Ok(*x as u32),
                Some(Val::Int(1 | 8, x)) => Ok(*x as u32),
                v => Err(format!("{name}: operand {k} is {v:?}")),
            }
        };
        // Undefined operands make undefined results, except where the
        // operation has an effect or reads a resource.
        let pure = !matches!(
            opcode,
            59 | 66..=69 | 78 | 79 | 80 | 139 | 140 | 216..=218
        );
        if pure && a.contains(&Val::Undef) {
            return Ok(Val::Undef);
        }
        let float = |x: f32| Val::F32(x.to_bits());
        Ok(match *opcode as u32 {
            6 => Val::F32(f(0)?.to_bits() & 0x7FFF_FFFF),
            7 => {
                let x = f(0)?;
                float(if x.is_nan() || x <= 0.0 {
                    0.0
                } else {
                    x.min(1.0)
                })
            }
            8 => Val::bool(f(0)?.is_nan()),
            9 => Val::bool(f(0)?.is_infinite()),
            10 => Val::bool(f(0)?.is_finite()),
            11 => Val::bool(f(0)?.is_normal()),
            12 => float(f(0)?.cos()),
            13 => float(f(0)?.sin()),
            14 => float(f(0)?.tan()),
            15 => float(f(0)?.acos()),
            16 => float(f(0)?.asin()),
            17 => float(f(0)?.atan()),
            18 => float(f(0)?.cosh()),
            19 => float(f(0)?.sinh()),
            20 => float(f(0)?.tanh()),
            21 => float(f(0)?.exp2()),
            22 => {
                let x = f(0)?;
                float(x - x.floor())
            }
            23 => float(f(0)?.log2()),
            24 => float(f(0)?.sqrt()),
            25 => float(1.0 / f(0)?.sqrt()),
            26 => float(f(0)?.round_ties_even()),
            27 => float(f(0)?.floor()),
            28 => float(f(0)?.ceil()),
            29 => float(f(0)?.trunc()),
            30 => Val::i32(u(0)?.reverse_bits()),
            31 => Val::i32(u(0)?.count_ones()),
            32 => {
                let x = u(0)?;
                Val::i32(if x == 0 { u32::MAX } else { x.trailing_zeros() })
            }
            // FirstbitHi and FirstbitSHi count from the top bit (DXC adds
            // `31 -` for HLSL's `firstbithigh`).
            33 => {
                let x = u(0)?;
                Val::i32(if x == 0 { u32::MAX } else { x.leading_zeros() })
            }
            34 => {
                let x = u(0)?;
                let x = if (x as i32) < 0 { !x } else { x };
                Val::i32(if x == 0 { u32::MAX } else { x.leading_zeros() })
            }
            35 | 36 => {
                let (x, y) = (f(0)?, f(1)?);
                let max = *opcode == 35;
                float(if x.is_nan() {
                    y
                } else if y.is_nan() || x == y || (x > y) == max {
                    x
                } else {
                    y
                })
            }
            37 => Val::i32((u(0)? as i32).max(u(1)? as i32) as u32),
            38 => Val::i32((u(0)? as i32).min(u(1)? as i32) as u32),
            39 => Val::i32(u(0)?.max(u(1)?)),
            40 => Val::i32(u(0)?.min(u(1)?)),
            46 => float(f(0)? * f(1)? + f(2)?),
            48 | 49 => Val::i32(u(0)?.wrapping_mul(u(1)?).wrapping_add(u(2)?)),
            51 | 52 => {
                // (width, offset, value), D3D's bit-field extract.
                let (w, o, x) = (u(0)? & 31, u(1)? & 31, u(2)?);
                let r = if w == 0 {
                    0
                } else if w + o < 32 {
                    let shifted = x << (32 - w - o);
                    if *opcode == 51 {
                        ((shifted as i32) >> (32 - w)) as u32
                    } else {
                        shifted >> (32 - w)
                    }
                } else if *opcode == 51 {
                    ((x as i32) >> o) as u32
                } else {
                    x >> o
                };
                Val::i32(r)
            }
            53 => {
                // (width, offset, value, replaced)
                let (w, o, x, y) = (u(0)? & 31, u(1)? & 31, u(2)?, u(3)?);
                let m = ((1u64 << w) - 1) as u32;
                let m = m.wrapping_shl(o);
                Val::i32((x.wrapping_shl(o) & m) | (y & !m))
            }
            59 => {
                let Some(Val::Handle(h)) = a.first() else {
                    return Err("cbufferLoadLegacy without a handle".into());
                };
                let row = u(1)? as usize;
                let words = self.cbuffer(h)?;
                let elems = self.fields(ret)?;
                let mut out = Vec::new();
                for (k, t) in elems.iter().enumerate() {
                    let w = words.get(4 * row + k).copied();
                    if w.is_none() {
                        self.env.out_of_bounds += 1;
                    }
                    out.push(self.make(*t, w.unwrap_or(0))?);
                }
                Val::Agg(out)
            }
            78 | 79 => self.atomic(*opcode == 79, a)?,
            80 => Val::Undef,
            93 => {
                let c = u(0)? as usize;
                Val::i32(self.env.group[c] * self.env.group_size[c] + self.thread()[c])
            }
            94 => Val::i32(self.env.group[u(0)? as usize]),
            95 => Val::i32(self.thread()[u(0)? as usize]),
            96 => {
                let [x, y, z] = self.thread();
                let [sx, sy, _] = self.env.group_size;
                Val::i32(x + sx * (y + sy * z))
            }
            111 => Val::i32(self.lane as u32),
            112 => Val::i32(self.env.wave_lanes),
            130 => Val::i32(u32::from(f32_to_f16_d3d(f(0)?))),
            131 => float(f16_to_f32(u(0)? as u16)),
            139 => {
                let Some(Val::Handle(h)) = a.first() else {
                    return Err("rawBufferLoad without a handle".into());
                };
                let at = self.buffer_offset(h, &a[1], &a[2])?;
                let mask = u(3)?;
                let elems = self.fields(ret)?;
                let mut out = Vec::new();
                for (k, t) in elems.iter().take(4).enumerate() {
                    let w = if mask >> k & 1 != 0 {
                        self.read_dword(h, at + 4 * k as u64)?
                    } else {
                        0
                    };
                    out.push(self.make(*t, w)?);
                }
                out.push(Val::i32(0));
                Val::Agg(out)
            }
            140 => {
                let Some(Val::Handle(h)) = a.first() else {
                    return Err("rawBufferStore without a handle".into());
                };
                let at = self.buffer_offset(h, &a[1], &a[2])?;
                let mask = u(7)?;
                for k in 0..4 {
                    if mask >> k & 1 == 0 {
                        continue;
                    }
                    let w = match &a[3 + k] {
                        Val::Int(32, x) => *x as u32,
                        Val::F32(x) => *x,
                        v => return Err(format!("a buffer store of {v:?}")),
                    };
                    self.write_dword(h, at + 4 * k as u64, w)?;
                }
                Val::Undef
            }
            216 => {
                let (Some(Val::Handle(h)), Some(Val::Agg(p))) = (a.first(), a.get(1)) else {
                    return Err("annotateHandle of a non-handle".into());
                };
                let word = |v: &Val| match v {
                    Val::Int(32, x) => Ok(*x as u32),
                    v => Err(format!("a property word {v:?}")),
                };
                Val::Handle(Handle {
                    place: h.place,
                    props: Some((word(&p[0])?, word(&p[1])?)),
                })
            }
            217 => {
                let Some(Val::Agg(bind)) = a.first() else {
                    return Err("createHandleFromBinding without a binding".into());
                };
                let field = |k: usize| match bind.get(k) {
                    Some(Val::Int(_, x)) => Ok(*x as u32),
                    v => Err(format!("a binding field {v:?}")),
                };
                let (lo, hi, space, class) = (field(0)?, field(1)?, field(2)?, field(3)?);
                let register = u(1)?;
                if !(lo..=hi).contains(&register) {
                    return Err(format!("register {register} outside [{lo}, {hi}]"));
                }
                Val::Handle(Handle {
                    place: Place::Binding {
                        class: class as u8,
                        space,
                        register,
                    },
                    props: None,
                })
            }
            218 => Val::Handle(Handle {
                place: Place::Heap {
                    index: u(0)?,
                    sampler: u(1)? != 0,
                },
                props: None,
            }),
            _ => return Err(format!("dx.op.{class} (opcode {opcode}) is not modelled")),
        })
    }

    /// The id in its group of the thread the current lane runs.
    fn thread(&self) -> [u32; 3] {
        let [x, y, z] = self.env.thread;
        [x + self.lane as u32, y, z]
    }

    /// The field types of struct type `t`.
    fn fields(&self, t: Ty) -> Result<Vec<Ty>, String> {
        match self.ty(t) {
            TypeDef::Struct(_, f) => Ok(f.clone()),
            t => Err(format!("a result of type {t:?}")),
        }
    }

    /// A value of type `t` from 32 bits.
    fn make(&self, t: Ty, bits: u32) -> Result<Val, String> {
        match self.ty(t) {
            TypeDef::Float => Ok(Val::F32(bits)),
            TypeDef::Int(32) => Ok(Val::i32(bits)),
            t => Err(format!("a 32-bit {t:?}")),
        }
    }

    fn cbuffer(&self, h: &Handle) -> Result<Vec<u32>, String> {
        match h.place {
            Place::Binding {
                class: 2,
                space: 0,
                register: 0,
            } => Ok(self.env.user_data.clone()),
            Place::Binding {
                class: 2,
                space: 1,
                register,
            } => self
                .env
                .root_cbvs
                .get(&register)
                .cloned()
                .ok_or(format!("no root CBV b{register}")),
            Place::Binding {
                class: 2,
                space: 2,
                register: 0,
            } => Ok(vec![self.env.probe_slots[0], self.env.probe_slots[1], 0, 0]),
            Place::Heap { index, .. } => {
                let bytes = self
                    .env
                    .heap
                    .get(&index)
                    .ok_or(format!("heap {index} empty"))?;
                Ok(bytes
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect())
            }
            p => Err(format!("a constant buffer at {p:?}")),
        }
    }

    /// The bytes behind a buffer handle.
    fn bytes(&mut self, h: &Handle) -> Result<&mut Vec<u8>, String> {
        match h.place {
            Place::Heap {
                index,
                sampler: false,
            } => self
                .env
                .heap
                .get_mut(&index)
                .ok_or(format!("heap index {index} holds no buffer")),
            Place::Binding {
                class: 0,
                space: 1,
                register,
            } => self
                .env
                .root_srvs
                .get_mut(&register)
                .ok_or(format!("no root SRV t{register}")),
            Place::Binding {
                class: 1,
                space: 2,
                register: 0,
            } => Ok(&mut self.env.probe),
            p => Err(format!("a buffer at {p:?}")),
        }
    }

    /// The byte offset a raw (index: bytes) or structured (index: element,
    /// then a byte offset) access addresses.
    fn buffer_offset(&self, h: &Handle, index: &Val, offset: &Val) -> Result<u64, String> {
        let Some((kind, stride)) = h.props else {
            return Err("a buffer access through an unannotated handle".into());
        };
        let word = |v: &Val| match v {
            Val::Int(32, x) => Ok(*x),
            v => Err(format!("a buffer coordinate {v:?}")),
        };
        match kind & 0xFF {
            11 => {
                if *offset != Val::Undef {
                    return Err("a raw buffer access with an element offset".into());
                }
                word(index)
            }
            12 => Ok(word(index)? * u64::from(stride) + word(offset)?),
            k => Err(format!("a raw access to a resource of kind {k}")),
        }
    }

    fn read_dword(&mut self, h: &Handle, at: u64) -> Result<u32, String> {
        let bytes = self.bytes(h)?;
        match usize::try_from(at)
            .ok()
            .and_then(|at| bytes.get(at..at.checked_add(4)?))
        {
            Some(b) => Ok(u32::from_le_bytes(b.try_into().expect("4 bytes"))),
            None => {
                self.env.out_of_bounds += 1;
                Ok(0)
            }
        }
    }

    fn write_dword(&mut self, h: &Handle, at: u64, w: u32) -> Result<(), String> {
        if h.props.is_some_and(|(k, _)| k & 0x1000 == 0) {
            return Err("a write through a read-only view".into());
        }
        let bytes = self.bytes(h)?;
        match usize::try_from(at)
            .ok()
            .and_then(|at| bytes.get_mut(at..at.checked_add(4)?))
        {
            Some(b) => b.copy_from_slice(&w.to_le_bytes()),
            None => self.env.out_of_bounds += 1,
        }
        Ok(())
    }

    /// The byte offset and size of the texel a typed texture UAV access
    /// addresses (coordinates by the resource kind: x, then y or the layer
    /// of a 1D array, then z or the layer of a 2D array), or `None` outside
    /// the texture.
    fn texel(&self, h: &Handle, coords: &[Val]) -> Result<Option<(u64, u32)>, String> {
        let (Some((kind, props)), Place::Heap { index, .. }) = (h.props, h.place) else {
            return Err("a texture access through an unannotated handle".into());
        };
        let [w, height, depth] = *self
            .env
            .textures
            .get(&index)
            .ok_or(format!("heap index {index} holds no texture"))?;
        // The component type: 4 and 5 are 32-bit integers, 6 and 7 64-bit.
        let size = match props & 0xFF {
            4 | 5 => 4,
            6 | 7 => 8,
            t => {
                return Err(format!(
                    "a texture atomic on component type {t} (properties {kind:#x}, {props:#x})"
                ));
            }
        };
        let c = |k: usize| match coords.get(k) {
            Some(Val::Int(32, x)) => Ok(*x),
            v => Err(format!("a texture coordinate {v:?}")),
        };
        let (x, y, z) = match kind & 0xFF {
            1 => (c(0)?, 0, 0),
            2 => (c(0)?, c(1)?, 0),
            4 | 7 => (c(0)?, c(1)?, c(2)?),
            6 => (c(0)?, 0, c(1)?),
            k => return Err(format!("a texture atomic on resource kind {k}")),
        };
        if x >= u64::from(w) || y >= u64::from(height) || z >= u64::from(depth) {
            return Ok(None);
        }
        let texel = (z * u64::from(height) + y) * u64::from(w) + x;
        Ok(Some((texel * u64::from(size), size)))
    }

    /// `atomicBinOp` (handle, op, c0, c1, c2, value) or
    /// `atomicCompareExchange` (handle, c0, c1, c2, compare, new), on a raw
    /// or structured buffer (32-bit) or a typed texture (32- or 64-bit).
    fn atomic(&mut self, cmpxchg: bool, a: &[Val]) -> Result<Val, String> {
        let Some(Val::Handle(h)) = a.first() else {
            return Err("an atomic without a handle".into());
        };
        let int = |v: &Val| match v {
            Val::Int(bits @ (32 | 64), x) => Ok((*bits, *x)),
            v => Err(format!("an atomic operand {v:?}")),
        };
        let coords = if cmpxchg { &a[1..4] } else { &a[2..5] };
        let texture = h.props.is_some_and(|(k, _)| matches!(k & 0xFF, 1..=9));
        let (at, size) = if texture {
            match self.texel(h, coords)? {
                Some(t) => t,
                None => {
                    // D3D drops atomics outside the resource.
                    self.env.out_of_bounds += 1;
                    let (bits, _) = int(&a[a.len() - 1])?;
                    return Ok(Val::Int(bits, 0));
                }
            }
        } else {
            (self.buffer_offset(h, &coords[0], &coords[1])?, 4)
        };
        let bits = size * 8;
        let (vbits, _) = int(&a[a.len() - 1])?;
        if vbits != bits {
            return Err(format!("an i{vbits} atomic on {bits}-bit data"));
        }
        let old = match size {
            4 => u64::from(self.read_dword(h, at)?),
            _ => {
                let lo = self.read_dword(h, at)?;
                let hi = self.read_dword(h, at + 4)?;
                u64::from(hi) << 32 | u64::from(lo)
            }
        };
        let new = if cmpxchg {
            let (cmp, new) = (int(&a[4])?.1, int(&a[5])?.1);
            if old == cmp { new } else { old }
        } else {
            let x = int(&a[5])?.1;
            let (sold, sx) = (sext(bits, old), sext(bits, x));
            let r = match int(&a[1])?.1 {
                0 => old.wrapping_add(x),
                1 => old & x,
                2 => old | x,
                3 => old ^ x,
                4 => sold.min(sx) as u64,
                5 => sold.max(sx) as u64,
                6 => old.min(x),
                7 => old.max(x),
                8 => x,
                op => return Err(format!("atomic operation {op}")),
            };
            mask(bits, r)
        };
        self.write_dword(h, at, new as u32)?;
        if size == 8 {
            self.write_dword(h, at + 4, (new >> 32) as u32)?;
        }
        Ok(Val::Int(bits, old))
    }
}

fn bin(op: Bin, a: Val, b: Val) -> Result<Val, String> {
    Ok(match (a, b) {
        (Val::Undef, _) | (_, Val::Undef) => Val::Undef,
        (Val::Int(bits, x), Val::Int(bits2, y)) => {
            if bits != bits2 {
                return Err(format!("{op:?} of i{bits} and i{bits2}"));
            }
            let shift = || {
                if y >= u64::from(bits) {
                    Err(format!("{op:?} of i{bits} by {y}"))
                } else {
                    Ok(y as u32)
                }
            };
            let r = match op {
                Bin::Add => x.wrapping_add(y),
                Bin::Sub => x.wrapping_sub(y),
                Bin::Mul => x.wrapping_mul(y),
                Bin::UDiv | Bin::URem if y == 0 => {
                    return Err(format!("{op:?} by zero"));
                }
                Bin::UDiv => x / y,
                Bin::URem => x % y,
                Bin::SDiv => {
                    let (sx, sy) = (sext(bits, x), sext(bits, y));
                    if sy == 0 || (sy == -1 && sx == sext(bits, 1 << (bits - 1))) {
                        return Err(format!("SDiv of {sx} by {sy}"));
                    }
                    (sx / sy) as u64
                }
                Bin::Shl => x << shift()?,
                Bin::LShr => x >> shift()?,
                Bin::AShr => (sext(bits, x) >> shift()?) as u64,
                Bin::And => x & y,
                Bin::Or => x | y,
                Bin::Xor => x ^ y,
            };
            Val::Int(bits, mask(bits, r))
        }
        (Val::F32(x), Val::F32(y)) => {
            let (x, y) = (f32::from_bits(x), f32::from_bits(y));
            let r = match op {
                Bin::Add => x + y,
                Bin::Sub => x - y,
                Bin::Mul => x * y,
                // The float form of code 4 is `fdiv`.
                Bin::SDiv => x / y,
                op => return Err(format!("float {op:?}")),
            };
            Val::F32(r.to_bits())
        }
        (a, b) => return Err(format!("{op:?} of {a:?} and {b:?}")),
    })
}

fn cmp(p: u8, a: Val, b: Val) -> Result<Val, String> {
    Ok(match (a, b) {
        (Val::Undef, _) | (_, Val::Undef) => Val::Undef,
        (Val::F32(x), Val::F32(y)) => {
            let (x, y) = (f32::from_bits(x), f32::from_bits(y));
            let unordered = x.is_nan() || y.is_nan();
            let ordered = |r: bool| !unordered && r;
            let or_unordered = |r: bool| unordered || r;
            Val::bool(match p {
                pred::FALSE => false,
                pred::OEQ => ordered(x == y),
                pred::OGT => ordered(x > y),
                pred::OGE => ordered(x >= y),
                pred::OLT => ordered(x < y),
                pred::OLE => ordered(x <= y),
                pred::ONE => ordered(x != y),
                pred::ORD => !unordered,
                pred::UNO => unordered,
                pred::UEQ => or_unordered(x == y),
                pred::UGT => or_unordered(x > y),
                pred::UGE => or_unordered(x >= y),
                pred::ULT => or_unordered(x < y),
                pred::ULE => or_unordered(x <= y),
                pred::UNE => or_unordered(x != y),
                pred::TRUE => true,
                p => return Err(format!("float compare {p}")),
            })
        }
        (Val::Int(bits, x), Val::Int(bits2, y)) if bits == bits2 => {
            let (sx, sy) = (sext(bits, x), sext(bits, y));
            Val::bool(match p {
                pred::EQ => x == y,
                pred::NE => x != y,
                pred::IUGT => x > y,
                pred::IUGE => x >= y,
                pred::IULT => x < y,
                pred::IULE => x <= y,
                pred::SGT => sx > sy,
                pred::SGE => sx >= sy,
                pred::SLT => sx < sy,
                pred::SLE => sx <= sy,
                p => return Err(format!("integer compare {p}")),
            })
        }
        (a, b) => return Err(format!("compare {p} of {a:?} and {b:?}")),
    })
}

fn rmw(op: Rmw, old: &Val, v: &Val) -> Result<Val, String> {
    let (Val::Int(bits, x), Val::Int(_, y)) = (old, v) else {
        return Err(format!("atomic {op:?} of {old:?} and {v:?}"));
    };
    let (bits, x, y) = (*bits, *x, *y);
    let (sx, sy) = (sext(bits, x), sext(bits, y));
    let r = match op {
        Rmw::Xchg => y,
        Rmw::Add => x.wrapping_add(y),
        Rmw::Sub => x.wrapping_sub(y),
        Rmw::And => x & y,
        Rmw::Or => x | y,
        Rmw::Xor => x ^ y,
        Rmw::Max => sx.max(sy) as u64,
        Rmw::Min => sx.min(sy) as u64,
        Rmw::UMax => x.max(y),
        Rmw::UMin => x.min(y),
    };
    Ok(Val::Int(bits, mask(bits, r)))
}

#[cfg(test)]
mod tests {
    use super::{f16_to_f32, f32_to_f16_d3d, f32_to_f16_nearest_even as f32_to_f16};

    #[test]
    fn half_conversions_round_trip_and_round_to_nearest_even() {
        // Every half (NaNs keep being NaNs) survives a round trip.
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            let back = f32_to_f16(f);
            if f.is_nan() {
                assert_eq!(back & 0x7C00, 0x7C00, "{h:#06x}");
                assert_ne!(back & 0x3FF, 0, "{h:#06x}");
            } else {
                assert_eq!(back, h, "{h:#06x} -> {f} -> {back:#06x}");
            }
        }
        // Ties to even, at a normal and a denormal step.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_f16(3.0 * 2f32.powi(-25)), 0x0002);
        // Overflow and the largest finite half.
        assert_eq!(f32_to_f16(65504.0), 0x7BFF);
        assert_eq!(f32_to_f16(65520.0), 0x7C00);
        assert_eq!(f32_to_f16(-1e10), 0xFC00);
    }

    /// D3D's `f32tof16` (D3D11.3 3.2.2): exact on every half, toward zero
    /// otherwise, the largest finite half on overflow, infinity and NaN
    /// kept, and never further from the value than nearest-even rounding's
    /// neighbour toward zero.
    #[test]
    fn d3d_half_conversion_rounds_toward_zero_and_clamps() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            let back = f32_to_f16_d3d(f);
            if f.is_nan() {
                assert_eq!(back & 0x7C00, 0x7C00, "{h:#06x}");
                assert_ne!(back & 0x3FF, 0, "{h:#06x}");
            } else {
                assert_eq!(back, h, "{h:#06x} -> {f} -> {back:#06x}");
            }
        }
        // Toward zero where nearest-even goes up.
        assert_eq!(f32_to_f16_d3d(1.0 + 3.0 * 2f32.powi(-12)), 0x3C00);
        assert_eq!(f32_to_f16_d3d(-(1.0 + 3.0 * 2f32.powi(-11))), 0xBC01);
        assert_eq!(f32_to_f16_d3d(3.0 * 2f32.powi(-25)), 0x0001);
        assert_eq!(f32_to_f16_d3d(2f32.powi(-25)), 0x0000);
        // Overflow stops at the largest finite half; infinity stays.
        assert_eq!(f32_to_f16_d3d(65520.0), 0x7BFF);
        assert_eq!(f32_to_f16_d3d(f32::MAX), 0x7BFF);
        assert_eq!(f32_to_f16_d3d(-1e10), 0xFBFF);
        assert_eq!(f32_to_f16_d3d(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16_d3d(f32::NEG_INFINITY), 0xFC00);
        // An f32 denormal is far below the smallest half: a zero of its sign.
        assert_eq!(f32_to_f16_d3d(f32::from_bits(0x8000_0001)), 0x8000);
        // Everywhere: the truncation of the value, so nearest-even's result
        // or one step below it in magnitude.
        let mut x = 0x2F00_0000u32;
        while x < 0x4780_0000 {
            let f = f32::from_bits(x);
            let (d3d, nearest) = (f32_to_f16_d3d(f), f32_to_f16(f));
            assert!(f16_to_f32(d3d) <= f, "{x:#010x}");
            assert!(d3d == nearest || d3d + 1 == nearest, "{x:#010x}");
            x += 0x0000_1F3B;
        }
    }
}
