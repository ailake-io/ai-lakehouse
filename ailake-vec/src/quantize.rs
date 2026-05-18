use half::f16;

#[derive(Debug, Clone, Copy)]
pub struct ScalingParams {
    pub scale: f32,
    pub zero_point: f32,
}

pub struct Quantizer;

impl Quantizer {
    pub fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 2);
        for &x in v {
            out.extend_from_slice(&f16::from_f32(x).to_le_bytes());
        }
        out
    }

    pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()
    }

    pub fn f32_to_i8(v: &[f32]) -> (Vec<i8>, ScalingParams) {
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let scale = if range == 0.0 { 1.0 } else { range / 254.0 };
        let zero_point = -128.0 - min / scale;
        let quant = v
            .iter()
            .map(|&x| ((x / scale + zero_point).round().clamp(-128.0, 127.0)) as i8)
            .collect();
        (quant, ScalingParams { scale, zero_point })
    }

    pub fn i8_to_f32(v: &[i8], params: &ScalingParams) -> Vec<f32> {
        v.iter()
            .map(|&x| (x as f32 - params.zero_point) * params.scale)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        let original: Vec<f32> = vec![0.1, -0.5, 1.0, 0.0, 100.0];
        let bytes = Quantizer::f32_to_f16_bytes(&original);
        let decoded = Quantizer::f16_bytes_to_f32(&bytes);
        for (a, b) in original.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.01, "f16 roundtrip error: {a} vs {b}");
        }
    }

    #[test]
    fn i8_roundtrip() {
        let original: Vec<f32> = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let (quant, params) = Quantizer::f32_to_i8(&original);
        let decoded = Quantizer::i8_to_f32(&quant, &params);
        for (a, b) in original.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.02, "i8 roundtrip error: {a} vs {b}");
        }
    }
}
