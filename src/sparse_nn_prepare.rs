//! Opt-in FP32 preparation experiments. No learned weights or solver units change.
use super::{sparse_nn_forward_f32, SparseCpuModel, CANONICAL_HANDS};
use std::cell::RefCell;
#[path = "sparse_nn_static_inputs.rs"]
mod static_inputs;
pub(crate) use static_inputs::{PreparedBasisInput, PreparedCommonInput};

#[derive(Debug, Clone)]
pub(super) struct PreparedPlan {
    mappings: Vec<usize>,
    cfv_indices: Vec<u32>,
    // In original canonical hand / OOP-IP order, retaining only initial support.
    reach_indices: Vec<(usize, u32)>,
    context_indices: Vec<u32>,
    outputs: Vec<(usize, u32)>,
    output_indices: Vec<u32>,
    coefficients: Vec<f32>,
    // Same canonical OOP/IP order as reach_indices, pointing directly into
    // the two solver-local ranges instead of a zero-filled canonical buffer.
    compact_reaches: Vec<(usize, usize, u32)>,
    compact_reach_lengths: [usize; 2],
    rms_denominator: usize,
    static_inputs: Option<static_inputs::StaticPlan>,
}

#[derive(Default)]
struct Scratch {
    indices: Vec<u32>,
    inputs: Vec<f32>,
    outputs: Vec<f32>,
    normalized: Vec<f32>,
}
thread_local! { static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default()); }
pub(super) fn diagnostic_worker_scratch_bytes()->u64{SCRATCH.with_borrow(|x|((x.indices.capacity()+x.inputs.capacity()+x.outputs.capacity()+x.normalized.capacity())*4) as u64)+static_inputs::diagnostic_worker_scratch_bytes()}

impl PreparedPlan {
    pub(super) fn diagnostic_numeric_bytes(&self)->usize{
        self.mappings.capacity()*std::mem::size_of::<usize>()+self.cfv_indices.capacity()*4+self.reach_indices.capacity()*std::mem::size_of::<(usize,u32)>()+self.context_indices.capacity()*4+self.outputs.capacity()*std::mem::size_of::<(usize,u32)>()+self.output_indices.capacity()*4+self.coefficients.capacity()*4+self.compact_reaches.capacity()*std::mem::size_of::<(usize,usize,u32)>()+self.static_inputs.as_ref().map_or(0,|x|x.diagnostic_numeric_bytes())
    }
    pub(super) fn new(model: &SparseCpuModel, mappings: &[usize], all_mappings: &[Vec<usize>; 2]) -> Option<Self> {
        if !model.hand_major
            || model.folded_buckets
            || model.bucket_projection.is_some()
            || model.input_channels != model.basis_count + 2
            || model.basis_count == 0
            || model.input_row_remap.is_empty()
            || model.dense_compact_inputs
            || model.stream_compact_inputs
        {
            return None;
        }
        let remap = |index| model.input_row_remap[index];
        let mut cfv_indices = Vec::new();
        for basis in 0..model.basis_count {
            for &canonical in mappings {
                cfv_indices.push(remap(canonical * model.input_channels + basis));
            }
        }
        if cfv_indices.contains(&u32::MAX) {
            return None;
        }
        let mut reach_indices = Vec::new();
        for canonical in 0..CANONICAL_HANDS {
            for side in 0..2 {
                let index = remap(canonical * model.input_channels + model.basis_count + side);
                if index != u32::MAX {
                    reach_indices.push((canonical * 2 + side, index));
                }
            }
        }
        let context_indices = (0..model.context_count)
            .map(|i| remap(CANONICAL_HANDS * model.input_channels + i))
            .collect::<Vec<_>>();
        if context_indices.contains(&u32::MAX) {
            return None;
        }
        let mut outputs = Vec::new();
        let mut coefficients = Vec::new();
        for (local, &canonical) in mappings.iter().enumerate() {
            if model
                .legal_mask
                .as_ref()
                .map_or(true, |mask| mask[canonical] != 0)
            {
                let index = if model.output_row_remap.is_empty() {
                    canonical as u32
                } else {
                    model.output_row_remap[canonical]
                };
                if index == u32::MAX {
                    return None;
                }
                outputs.push((local, index));
                for basis in 0..model.basis_count {
                    coefficients
                        .push(1.0 + model.linear_weights[basis * model.output_count + canonical]);
                }
            }
        }
        let mut local_indices = [[usize::MAX; CANONICAL_HANDS]; 2];
        for side in 0..2 {
            for (local, &canonical) in all_mappings[side].iter().enumerate() {
                local_indices[side][canonical] = local;
            }
        }
        let mut compact_reaches = Vec::with_capacity(reach_indices.len());
        for &(original, index) in &reach_indices {
            let canonical = original / 2;
            let side = original & 1;
            let local = local_indices[side][canonical];
            if local == usize::MAX { return None; }
            compact_reaches.push((side, local, index));
        }
        let hand_count = if model.supported_hand_rms {
            super::supported_legal_hand_count(mappings, model.legal_mask.as_deref()).max(1)
        } else { model.legal_count };
        let static_inputs = match std::env::var("PFS_NN_STATIC_INPUTS").as_deref() {
            Err(_) | Ok("auto") => {
                if model.supported_hand_rms && matches!(model.hidden, 64 | 128)
                    && super::cpu_defaults::available() {
                    eprintln!("STATIC_NN_INPUTS mode=dense hidden={} basis={}",model.hidden,model.basis_count);
                    Some(static_inputs::StaticPlan::new(model, mappings, all_mappings, true))
                } else { None }
            }
            Ok("0") | Ok("off") => None,
            Ok(mode @ ("sparse" | "dense")) => {
                if mode == "dense" {
                    #[cfg(target_arch = "x86_64")]
                    assert!(std::is_x86_feature_detected!("avx512f"), "Static dense NN requires AVX512");
                    #[cfg(not(target_arch = "x86_64"))]
                    panic!("Static dense NN requires x86 AVX512");
                }
                eprintln!("STATIC_NN_INPUTS mode={mode} hidden={} basis={}", model.hidden, model.basis_count);
                Some(static_inputs::StaticPlan::new(model, mappings, all_mappings, mode == "dense"))
            }
            Ok(mode) => panic!("Unknown PFS_NN_STATIC_INPUTS={mode}"),
        };
        Some(Self {
            mappings: mappings.to_vec(),
            cfv_indices,
            reach_indices,
            context_indices,
            output_indices: outputs.iter().map(|&(_, i)| i).collect(),
            outputs,
            coefficients,
            compact_reaches,
            compact_reach_lengths: [all_mappings[0].len(), all_mappings[1].len()],
            rms_denominator: hand_count * model.basis_count,
            static_inputs,
        })
    }

    pub(super) fn supports_child_inputs(&self) -> bool {
        self.static_inputs.as_ref().is_some_and(|p|p.supports_child_inputs())
    }
    pub(super) fn prepare_basis(&self, model:&SparseCpuModel, basis:usize, raw:Vec<f32>) -> PreparedBasisInput {
        self.static_inputs.as_ref().expect("child inputs require static dense plan").prepare_basis(model,basis,raw)
    }
    pub(super) fn prepare_common(&self, context:&[f32], reaches:[&[f32];2]) -> PreparedCommonInput {
        self.static_inputs.as_ref().expect("child inputs require static dense plan").prepare_common(context,reaches)
    }
    pub(super) fn infer_children(&self, model:&SparseCpuModel, blocks:&[PreparedBasisInput],
        common:PreparedCommonInput, hands:usize) -> Vec<f64> {
        self.static_inputs.as_ref().expect("child inputs require static dense plan").infer_children(model,blocks,common,hands)
    }

    pub(super) fn matches(&self, mappings: &[usize]) -> bool {
        self.mappings == mappings
    }

    /// mode 1: precomputed plans, original division/order. mode 2: reciprocal
    /// normalization. mode 3: reciprocal plus SIMD RMS reduction. mode 4:
    /// original reduction and exact SIMD division, with no reciprocals.
    pub(super) fn infer(
        &self,
        model: &SparseCpuModel,
        values: &[f32],
        context: &[f32],
        reaches: &[f32],
        mode: u8,
    ) -> Vec<f32> {
        let hands = self.mappings.len();
        assert_eq!(values.len(), hands * model.basis_count);
        assert_eq!(context.len(), model.context_count);
        assert_eq!(reaches.len(), CANONICAL_HANDS * 2);
        let reciprocal = mode == 2 || mode == 3;
        let sum_squares = if mode == 3 {
            sum_squares_simd(values)
        } else {
            values.iter().map(|v| v * v).sum()
        };
        if sum_squares == 0.0 {
            return vec![0.0; hands];
        }
        let denominator = if model.supported_hand_rms {
            super::supported_legal_hand_count(&self.mappings,model.legal_mask.as_deref()).max(1)
        } else {
            model.legal_count
        };
        let scale = super::recover_cfv_rms(values,
            (sum_squares / (denominator * model.basis_count) as f32).sqrt(),
            denominator * model.basis_count);
        let inverse = if reciprocal { 1.0 / scale } else { 0.0 };
        let mut result = vec![0.0; hands];
        SCRATCH.with_borrow_mut(|scratch| {
            scratch.indices.clear();
            scratch.inputs.clear();
            let maximum = self.cfv_indices.len() + self.reach_indices.len() + context.len();
            scratch.indices.reserve(maximum);
            scratch.inputs.reserve(maximum);
            if mode == 4 {
                scratch.normalized.resize(values.len(), 0.0);
                divide_exact(values, scale, &mut scratch.normalized);
            }
            for (position, (&value, &index)) in values.iter().zip(&self.cfv_indices).enumerate() {
                if value != 0.0 {
                    scratch.indices.push(index);
                    scratch.inputs.push(if mode == 4 {
                        scratch.normalized[position]
                    } else if reciprocal {
                        value * inverse
                    } else {
                        value / scale
                    });
                }
            }
            let mut sums = [0.0f32; 2];
            for &(original, _) in &self.reach_indices {
                sums[original & 1] += reaches[original];
            }
            let denominators = [sums[0].max(1e-12), sums[1].max(1e-12)];
            let reciprocals = if reciprocal {
                [1.0 / denominators[0], 1.0 / denominators[1]]
            } else {
                [0.0; 2]
            };
            for &(original, index) in &self.reach_indices {
                let value = if reciprocal {
                    reaches[original] * reciprocals[original & 1]
                } else {
                    reaches[original] / denominators[original & 1]
                };
                if value != 0.0 {
                    scratch.indices.push(index);
                    scratch.inputs.push(value);
                }
            }
            for (&value, &index) in context.iter().zip(&self.context_indices) {
                if value != 0.0 {
                    scratch.indices.push(index);
                    scratch.inputs.push(value);
                }
            }
            scratch.outputs.resize(self.outputs.len(), 0.0);
            unsafe {
                sparse_nn_forward_f32(
                    model.w1_f32_input_major.as_ptr(),
                    model.b1.as_ptr(),
                    model.norm_weight.as_ptr(),
                    model.norm_bias.as_ptr(),
                    model.prelu,
                    model.w2_f32_output_major.as_ptr(),
                    model.b2.as_ptr(),
                    model.hidden as u32,
                    scratch.indices.as_ptr(),
                    scratch.inputs.as_ptr(),
                    scratch.indices.len() as u32,
                    self.output_indices.as_ptr(),
                    self.output_indices.len() as u32,
                    scratch.outputs.as_mut_ptr(),
                );
            }
            for (index, &(local, _)) in self.outputs.iter().enumerate() {
                let mut baseline = 0.0;
                for basis in 0..model.basis_count {
                    baseline += values[basis * hands + local]
                        * self.coefficients[index * model.basis_count + basis];
                }
                result[local] = baseline + scratch.outputs[index] * scale;
            }
        });
        result
    }

    /// Opt-in mode4 data-flow experiment. Preserve every FP32 arithmetic step
    /// and its order, returning recovered values directly in the caller's FP64
    /// units. Only the final caller-owned result is allocated after warm-up.
    pub(super) fn infer_compact_f64(
        &self, model: &SparseCpuModel, values: &[f32], context: &[f32],
        reaches: [&[f32]; 2],
    ) -> Vec<f64> {
        let hands = self.mappings.len();
        assert_eq!(values.len(), hands * model.basis_count);
        assert_eq!(context.len(), self.context_indices.len());
        for side in 0..2 { assert_eq!(reaches[side].len(), self.compact_reach_lengths[side]); }
        if let Some(plan) = &self.static_inputs {
            return plan.infer(model, values, context, reaches, hands);
        }
        let mut masked = None;
        if model.supported_hand_rms {
            if let Some(legal) = &model.legal_mask {
                if values.iter().enumerate().any(|(i, &v)| v != 0.0 && legal[self.mappings[i % hands]] == 0) {
                    let mut copy = values.to_vec();
                    for (i, v) in copy.iter_mut().enumerate() {
                        if legal[self.mappings[i % hands]] == 0 { *v = 0.0; }
                    }
                    masked = Some(copy);
                }
            }
        }
        let values = masked.as_deref().unwrap_or(values);
        let sum_squares: f32 = values.iter().map(|v| v * v).sum();
        if sum_squares == 0.0 { return vec![0.0; hands]; }
        let scale = super::recover_cfv_rms(values,
            (sum_squares / self.rms_denominator as f32).sqrt(), self.rms_denominator);
        let mut result = vec![0.0; hands];
        SCRATCH.with_borrow_mut(|scratch| {
            scratch.indices.clear();
            scratch.inputs.clear();
            let maximum = self.cfv_indices.len() + self.compact_reaches.len() + context.len();
            scratch.indices.reserve(maximum);
            scratch.inputs.reserve(maximum);
            scratch.normalized.resize(values.len(), 0.0);
            divide_exact(values, scale, &mut scratch.normalized);
            for (position, (&value, &index)) in values.iter().zip(&self.cfv_indices).enumerate() {
                if value != 0.0 {
                    scratch.indices.push(index);
                    scratch.inputs.push(scratch.normalized[position]);
                }
            }
            let reach_value = |position: usize| {
                let (side, local, _) = self.compact_reaches[position];
                let canonical = self.reach_indices[position].0 / 2;
                if model.supported_hand_rms && model.legal_mask.as_ref().is_some_and(|m| m[canonical] == 0) {
                    0.0
                } else { reaches[side][local] }
            };
            let mut sums = [0.0f32; 2];
            for (position, &(side, _, _)) in self.compact_reaches.iter().enumerate() {
                sums[side] += reach_value(position);
            }
            let denominators = [sums[0].max(1e-12), sums[1].max(1e-12)];
            for (position, &(side, _, index)) in self.compact_reaches.iter().enumerate() {
                let value = reach_value(position) / denominators[side];
                if value != 0.0 {
                    scratch.indices.push(index);
                    scratch.inputs.push(value);
                }
            }
            for (&value, &index) in context.iter().zip(&self.context_indices) {
                if value != 0.0 { scratch.indices.push(index); scratch.inputs.push(value); }
            }
            scratch.outputs.resize(self.outputs.len(), 0.0);
            unsafe {
                sparse_nn_forward_f32(model.w1_f32_input_major.as_ptr(), model.b1.as_ptr(),
                    model.norm_weight.as_ptr(), model.norm_bias.as_ptr(), model.prelu,
                    model.w2_f32_output_major.as_ptr(), model.b2.as_ptr(), model.hidden as u32,
                    scratch.indices.as_ptr(), scratch.inputs.as_ptr(), scratch.indices.len() as u32,
                    self.output_indices.as_ptr(), self.output_indices.len() as u32,
                    scratch.outputs.as_mut_ptr());
            }
            for (index, &(local, _)) in self.outputs.iter().enumerate() {
                let mut baseline = 0.0f32;
                for basis in 0..model.basis_count {
                    baseline += values[basis * hands + local] * self.coefficients[index * model.basis_count + basis];
                }
                // The FP32 rounding here is part of the model interface.
                let recovered = baseline + scratch.outputs[index] * scale;
                result[local] = recovered as f64 * 1.0;
            }
        });
        result
    }
}

fn sum_squares_simd(values: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return unsafe { squared_avx2(values) };
    }
    values.iter().map(|v| v * v).sum()
}

fn divide_exact(values: &[f32], scale: f32, output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { divide_avx2(values, scale, output) };
        return;
    }
    for (out, &value) in output.iter_mut().zip(values) {
        *out = value / scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn divide_avx2(values: &[f32], scale: f32, output: &mut [f32]) {
    use std::arch::x86_64::*;
    let divisor = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 8 <= values.len() {
        _mm256_storeu_ps(
            output.as_mut_ptr().add(i),
            _mm256_div_ps(_mm256_loadu_ps(values.as_ptr().add(i)), divisor),
        );
        i += 8;
    }
    for j in i..values.len() {
        output[j] = values[j] / scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn squared_avx2(values: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sums = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= values.len() {
        for k in 0..4 {
            let v = _mm256_loadu_ps(values.as_ptr().add(i + k * 8));
            sums[k] = _mm256_add_ps(sums[k], _mm256_mul_ps(v, v));
        }
        i += 32;
    }
    let total = _mm256_add_ps(
        _mm256_add_ps(sums[0], sums[1]),
        _mm256_add_ps(sums[2], sums[3]),
    );
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), total);
    let mut result = lanes.iter().copied().sum::<f32>();
    for &v in &values[i..] {
        result += v * v;
    }
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn simd_division_is_bit_identical() {
        let values: Vec<f32> = (0..1989).map(|i| (i as f32 - 1000.0) / 71.0).collect();
        for scale in [1e-12, 0.0000314159, 0.812345, 10000.123] {
            let mut output = vec![0.0; values.len()];
            super::divide_exact(&values, scale, &mut output);
            for (&v, &out) in values.iter().zip(&output) {
                assert_eq!((v / scale).to_bits(), out.to_bits());
            }
        }
    }
    #[test]
    fn rms_reduction_preserves_scale() {
        for n in [0, 1, 7, 32, 65, 1989, 8192] {
            let values: Vec<f32> = (0..n)
                .map(|i| {
                    if i % 3 == 0 {
                        0.0
                    } else {
                        (i as f32 - 100.0) / 731.0
                    }
                })
                .collect();
            let reference: f32 = values.iter().map(|v| v * v).sum();
            let actual = super::sum_squares_simd(&values);
            assert!((actual - reference).abs() <= reference.abs() * 3e-6 + 1e-12);
        }
    }
}
