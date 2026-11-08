//! A fixed PRNG, so a case regenerates byte for byte on any machine.
//!
//! Deliberately not the system RNG and deliberately not seeded by time: the
//! point of the bench is that `gen` on the Mac and `gen` in the container
//! produce identical bytes, and that a failure found today reproduces
//! tomorrow. splitmix64 is used to spread the seed, then xoshiro256++ runs.

pub struct Rng(u64, u64, u64, u64);

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Seeded by what the stream is *for*, not by a counter, so adding a new
    /// tensor or a new variant never shifts the bytes of an existing one.
    pub fn new(label: &str, case_id: u32, extra: u64) -> Self {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in label.bytes() {
            h = (h ^ b as u64).wrapping_mul(0x1000_0000_01b3);
        }
        let mut s = h ^ ((case_id as u64) << 32) ^ extra.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        Rng(splitmix(&mut s), splitmix(&mut s), splitmix(&mut s), splitmix(&mut s))
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.0.wrapping_add(self.3).rotate_left(23).wrapping_add(self.0);
        let t = self.1 << 17;
        self.2 ^= self.0;
        self.3 ^= self.1;
        self.1 ^= self.2;
        self.0 ^= self.3;
        self.2 ^= t;
        self.3 = self.3.rotate_left(45);
        result
    }

    /// Full INT8 range, which `problem.md` §3.4 says the data covers.
    pub fn int8(&mut self) -> i8 {
        (self.next_u64() >> 56) as u8 as i8
    }

    pub fn unit(&mut self) -> f64 {
        // 53 bits is all an f64 mantissa holds.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Log-uniform in [lo, hi]. Scales span [1e-5, 1.0], five decades, so a
    /// linear draw would put almost everything in the top decade and never
    /// exercise the small end.
    pub fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let t = self.unit();
        (lo.ln() + t * (hi.ln() - lo.ln())).exp()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new("x", 7, 0);
        let mut b = Rng::new("x", 7, 0);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_labels_diverge() {
        let mut a = Rng::new("x", 7, 0);
        let mut b = Rng::new("weight", 7, 0);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn int8_covers_the_whole_range() {
        let mut r = Rng::new("x", 1, 0);
        let (mut lo, mut hi) = (0i8, 0i8);
        for _ in 0..200_000 {
            let v = r.int8();
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert_eq!((lo, hi), (-128, 127));
    }

    #[test]
    fn log_uniform_stays_in_range_and_spreads() {
        let mut r = Rng::new("scale", 1, 0);
        let mut decades = [0usize; 5];
        for _ in 0..100_000 {
            let v = r.log_uniform(1e-5, 1.0);
            assert!((1e-5..=1.0).contains(&v), "{v}");
            let d = (-v.log10()).floor() as usize;
            decades[d.min(4)] += 1;
        }
        // Every decade should get a real share, which is the whole point.
        for (i, n) in decades.iter().enumerate() {
            assert!(*n > 5_000, "decade {i} only got {n}");
        }
    }
}
