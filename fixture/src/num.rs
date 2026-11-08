//! FLOAT16 / BFLOAT16 conversion, with round-half-to-even, which `problem.md`
//! §3.2 names explicitly as the output rounding.
//!
//! These are hand-written because the crate has no dependencies and Rust's own
//! `f16` is not stable. They matter more than they look: the accuracy rule is
//! relative error < 1e-3, and BFLOAT16's own representation error is 2^-8 =
//! 3.9e-3 -- already past the budget. So the reference cannot be compared in
//! FLOAT32; it has to be rounded to the output type first, exactly the way the
//! judge's reference does, or every bf16 case fails on representation alone.

/// FLOAT32 -> BFLOAT16 bits, round-half-to-even. bf16 is the top 16 bits of a
/// f32, so rounding is an add of half-an-ulp biased by the kept lsb.
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return 0x7fc0; // NaN, quieted
    }
    let lsb = (bits >> 16) & 1;
    (((bits + 0x7fff + lsb) >> 16) & 0xffff) as u16
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// FLOAT32 -> FLOAT16 bits, round-half-to-even, with subnormals and overflow
/// to infinity. `problem.md` §3.4 promises outputs are finite and |v| <= 60000
/// (below f16's 65504) but also allows "very small output values", so the
/// subnormal path is on the hook.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        // Infinity keeps its mantissa at zero; NaN must stay NaN after the
        // shift, so force a mantissa bit rather than letting it truncate away.
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00; // overflows f16's range
    }
    if unbiased >= -14 {
        // Normal. 23 mantissa bits down to 10, so 13 are dropped.
        let kept = (mant >> 13) as u16;
        let dropped = mant & 0x1fff;
        let mut out = sign | (((unbiased + 15) as u16) << 10) | kept;
        // A carry here walks into the exponent field, which is exactly right:
        // 0x03ff + 1 == 0x0400 is the next binade.
        if dropped > 0x1000 || (dropped == 0x1000 && kept & 1 == 1) {
            out += 1;
        }
        return out;
    }

    // Subnormal: shift the implicit one back in and keep shifting right.
    let shift = (-unbiased - 14) as u32;
    if shift > 24 {
        return sign; // rounds to zero
    }
    let full = mant | 0x0080_0000;
    let drop_bits = 13 + shift;
    let kept = (full >> drop_bits) as u16;
    let dropped = full & ((1u32 << drop_bits) - 1);
    let half = 1u32 << (drop_bits - 1);
    let mut out = sign | kept;
    if dropped > half || (dropped == half && kept & 1 == 1) {
        out += 1; // may carry up into the smallest normal, which is correct
    }
    out
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal f16 is normal in f32: renormalise by shifting the leading
        // one out of the mantissa and paying for it in the exponent. A f16
        // subnormal is mant * 2^-24, so with the top set bit of `mant` at
        // position p the value is 2^(p-24) times a fraction, shift == 10 - p,
        // and the bits left over become f32's top ten mantissa bits.
        let shift = mant.leading_zeros() - 21; // == 10 - p, since mant < 2^10
        let exp32 = 113 - shift; // (p - 24) + 127
        let mant32 = (mant << shift) & 0x03ff;
        return f32::from_bits(sign | (exp32 << 23) | (mant32 << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// The output element type of a case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Out {
    F16,
    Bf16,
}

impl Out {
    pub fn name(self) -> &'static str {
        match self {
            Out::F16 => "fp16",
            Out::Bf16 => "bf16",
        }
    }

    pub fn round(self, x: f32) -> u16 {
        match self {
            Out::F16 => f32_to_f16(x),
            Out::Bf16 => f32_to_bf16(x),
        }
    }

    pub fn decode(self, bits: u16) -> f32 {
        match self {
            Out::F16 => f16_to_f32(bits),
            Out::Bf16 => bf16_to_f32(bits),
        }
    }

    /// Relative step between neighbouring representable values -- the error
    /// floor that representation alone costs, before any computation error.
    pub fn eps(self) -> f32 {
        match self {
            Out::F16 => 2.0f32.powi(-11),  // 4.88e-4
            Out::Bf16 => 2.0f32.powi(-8),  // 3.91e-3
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_exact_values() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-1.0), 0xbc00);
        assert_eq!(f32_to_f16(2.0), 0x4000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(65504.0), 0x7bff); // f16 max
        assert_eq!(f32_to_f16(65536.0), 0x7c00); // overflows to inf
        assert_eq!(f32_to_f16(6.103515625e-5), 0x0400); // smallest normal
        assert_eq!(f32_to_f16(5.960464477539063e-8), 0x0001); // smallest subnormal
    }

    #[test]
    fn f16_round_half_to_even() {
        // 1 + 2^-11 is exactly half way between 1.0 (even mantissa) and the
        // next f16 up, so it must round down to 1.0.
        assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11)), 0x3c00);
        // 1 + 3*2^-11 sits half way between 0x3c01 (odd) and 0x3c02, so it
        // rounds up to the even one.
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3c02);
        // Just past half way always goes up.
        assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11) * 1.5), 0x3c01);
    }

    #[test]
    fn f16_carry_out_of_the_mantissa() {
        // The largest value that still rounds into 1.0's binade must carry
        // into the exponent rather than wrapping the mantissa.
        let just_under_two = 2.0f32 - 2.0f32.powi(-11);
        assert_eq!(f32_to_f16(just_under_two), 0x4000);
    }

    #[test]
    fn f16_roundtrip_is_identity_on_representables() {
        for bits in 0u16..=0xffff {
            let exp = (bits >> 10) & 0x1f;
            if exp == 0x1f {
                continue; // inf/nan
            }
            let back = f32_to_f16(f16_to_f32(bits));
            // -0 and +0 both decode to their own zero, so compare bit for bit.
            assert_eq!(back, bits, "bits {bits:#06x}");
        }
    }

    #[test]
    fn bf16_exact_values() {
        assert_eq!(f32_to_bf16(0.0), 0x0000);
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(f32_to_bf16(-2.0), 0xc000);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
    }

    #[test]
    fn bf16_round_half_to_even() {
        // Halfway between 1.0 (even) and the next bf16: rounds down.
        let half_up = f32::from_bits(0x3f80_8000);
        assert_eq!(f32_to_bf16(half_up), 0x3f80);
        // Halfway between 0x3f81 (odd) and 0x3f82: rounds up to even.
        let half_up_odd = f32::from_bits(0x3f81_8000);
        assert_eq!(f32_to_bf16(half_up_odd), 0x3f82);
    }

    #[test]
    fn bf16_roundtrip_is_identity_on_representables() {
        for hi in 0u32..=0xffff {
            let exp = (hi >> 7) & 0xff;
            if exp == 0xff {
                continue;
            }
            let bits = hi as u16;
            assert_eq!(f32_to_bf16(bf16_to_f32(bits)), bits, "bits {bits:#06x}");
        }
    }

    #[test]
    fn representation_error_is_what_we_claim() {
        // The whole reason golden rounds to the output type: bf16's own error
        // already exceeds the 1e-3 budget, f16's eats half of it.
        assert!(Out::Bf16.eps() > 1e-3, "bf16 eps {}", Out::Bf16.eps());
        assert!(Out::F16.eps() < 1e-3 && Out::F16.eps() > 4e-4);
    }
}
