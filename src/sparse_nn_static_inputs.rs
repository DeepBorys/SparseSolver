//! Fixed board/range FP32 input plans, guarded by hardware/model compatibility.
use super::super::buffer_pool;
use super::super::{recover_cfv_rms, sparse_nn_forward_f32, SparseCpuModel};
use std::cell::RefCell;
use std::sync::OnceLock;

fn child_partial_rms_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = std::env::var("PFS_CHILD_PARTIAL_RMS_EXPERIMENT").as_deref() == Ok("1");
        if enabled { eprintln!("NN_CHILD_PARTIAL_RMS enabled=true reduction=fp32_child_totals"); }
        enabled
    })
}
fn child_partial_hidden_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let enabled = std::env::var("PFS_CHILD_PARTIAL_HIDDEN_EXPERIMENT").as_deref() == Ok("1");
        if enabled { eprintln!("NN_CHILD_PARTIAL_HIDDEN enabled=true rms=established_order"); }
        enabled
    })
}
#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn sparse_nn_partial_first_f32(w1: *const f32, values: *const f32,
        inputs: u32, hidden: u32, out: *mut f32);
    fn sparse_nn_forward_f32_static_dense(
        w1: *const f32,
        b1: *const f32,
        nw: *const f32,
        nb: *const f32,
        prelu: f32,
        w2: *const f32,
        b2: *const f32,
        hidden: u32,
        input: *const f32,
        n: u32,
        out_n: u32,
        out: *mut f32,
    );
    fn sparse_nn_forward_f32_register_dense(
        w1: *const f32,
        b1: *const f32,
        nw: *const f32,
        nb: *const f32,
        prelu: f32,
        w2: *const f32,
        b2: *const f32,
        hidden: u32,
        input: *const f32,
        n: u32,
        out_n: u32,
        out: *mut f32,
    );
}
/// Owned per-card result prepared on that card's traversal worker.
#[derive(Debug)]
pub(crate) struct PreparedBasisInput {
    recycled: bool,
    linear: Option<Vec<f32>>,
    count: usize,
    pub(crate) raw_cfvs: Vec<f32>,
    values: Vec<f32>,
    squares: Vec<f32>,
    partial_square_sum: Option<f32>,
    partial_hidden: Vec<f32>,
    baseline_products: Vec<f32>,
}
impl Drop for PreparedBasisInput {
    fn drop(&mut self) {
        if self.recycled {
            buffer_pool::recycle(std::mem::take(&mut self.raw_cfvs));
            if let Some(data) = self.linear.take() {
                buffer_pool::recycle(data);
            }
        }
    }
}
impl PreparedBasisInput {
    fn values(&self) -> &[f32] {
        self.linear
            .as_ref()
            .map_or(self.values.as_slice(), |v| &v[..self.count])
    }
    fn squares(&self) -> &[f32] {
        self.linear
            .as_ref()
            .map_or(self.squares.as_slice(), |v| &v[self.count..2 * self.count])
    }
    fn products(&self) -> &[f32] {
        self.linear
            .as_ref()
            .map_or(self.baseline_products.as_slice(), |v| {
                let offset = if self.partial_square_sum.is_some() { self.count } else { 2 * self.count };
                &v[offset..]
            })
    }
}
#[derive(Debug, Clone)]
struct CopyRun {
    local: usize,
    compact: usize,
    len: usize,
}
#[derive(Debug, Clone)]
struct LinearLayout {
    runs: Vec<CopyRun>,
    coefficients: Vec<Vec<f32>>,
    range_sides: Vec<usize>,
}
/// Parent-node reaches/context, prepared independently of the card subtrees.
#[derive(Debug)]
pub(crate) struct PreparedCommonInput {
    features: Vec<f32>,
}
impl Drop for PreparedCommonInput {
    fn drop(&mut self) {
        buffer_pool::recycle(std::mem::take(&mut self.features));
    }
}
#[derive(Debug, Clone)]
pub(super) struct StaticPlan {
    dense: bool,
    register_kernel: bool,
    linear: Option<LinearLayout>,
    cfvs: Vec<(usize, u32)>,
    ranges: Vec<(usize, usize, u32)>,
    context: Vec<u32>,
    outputs: Vec<(usize, u32)>,
    output_ids: Vec<u32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    coefficients: Vec<f32>,
    denominator: usize,
    inputs: usize,
}
impl StaticPlan {
    pub(super) fn diagnostic_numeric_bytes(&self)->usize{
        let mut n=self.cfvs.capacity()*std::mem::size_of::<(usize,u32)>()+self.ranges.capacity()*std::mem::size_of::<(usize,usize,u32)>()+self.context.capacity()*4+self.outputs.capacity()*std::mem::size_of::<(usize,u32)>()+self.output_ids.capacity()*4;
        for v in [&self.w1,&self.w2,&self.b2,&self.coefficients]{n+=v.capacity()*4;}
        if let Some(l)=&self.linear{n+=l.runs.capacity()*std::mem::size_of::<CopyRun>()+l.range_sides.capacity()*std::mem::size_of::<usize>();for v in &l.coefficients{n+=v.capacity()*4;}}
        n
    }
    pub(super) fn new(
        m: &SparseCpuModel,
        mappings: &[usize],
        maps: &[Vec<usize>; 2],
        dense: bool,
    ) -> Self {
        assert!(
            m.supported_hand_rms,
            "Static input plans require supported-hand RMS"
        );
        let legal = |h: usize| m.legal_mask.as_ref().map_or(true, |v| v[h] != 0);
        let mut rows = Vec::new();
        let mut cfvs = Vec::new();
        let mut ranges = Vec::new();
        for b in 0..m.basis_count {
            for (local, &h) in mappings.iter().enumerate() {
                if legal(h) {
                    cfvs.push((b * mappings.len() + local, rows.len() as u32));
                    rows.push(m.input_row_remap[h * m.input_channels + b] as usize);
                }
            }
        }
        let mut reverse = [[usize::MAX; 1326]; 2];
        for p in 0..2 {
            for (local, &h) in maps[p].iter().enumerate() {
                reverse[p][h] = local;
            }
        }
        for h in 0..1326 {
            for p in 0..2 {
                if reverse[p][h] != usize::MAX && legal(h) {
                    ranges.push((p, reverse[p][h], rows.len() as u32));
                    rows.push(m.input_row_remap[h * m.input_channels + m.basis_count + p] as usize);
                }
            }
        }
        let mut context = Vec::new();
        for i in 0..m.context_count {
            context.push(rows.len() as u32);
            rows.push(m.input_row_remap[1326 * m.input_channels + i] as usize);
        }
        let mut w1 = Vec::new();
        for &old in &rows {
            assert_ne!(old, u32::MAX as usize);
            w1.extend_from_slice(&m.w1_f32_input_major[old * m.hidden..(old + 1) * m.hidden]);
        }
        let mut outputs = Vec::new();
        let mut w2 = Vec::new();
        let mut b2 = Vec::new();
        let mut coefficients = Vec::new();
        for (local, &h) in mappings.iter().enumerate() {
            if legal(h) {
                let old = m.output_row_remap[h];
                outputs.push((local, old));
                w2.extend_from_slice(
                    &m.w2_f32_output_major[old as usize * m.hidden..(old as usize + 1) * m.hidden],
                );
                b2.push(m.b2[old as usize]);
                for b in 0..m.basis_count {
                    coefficients.push(1. + m.linear_weights[b * 1326 + h]);
                }
            }
        }
        let denominator = cfvs.len().max(1);
        let inputs = rows.len();
        let output_ids = (0..outputs.len() as u32).collect();
        let linear = if dense && super::super::cpu_defaults::flag("PFS_NN_LINEAR_LAYOUT") {
            let mut runs: Vec<CopyRun> = Vec::new();
            for (compact, &(local, _)) in outputs.iter().enumerate() {
                if let Some(last) = runs.last_mut().filter(|r| r.local + r.len == local) {
                    last.len += 1;
                } else {
                    runs.push(CopyRun {
                        local,
                        compact,
                        len: 1,
                    });
                }
            }
            let by_basis = (0..m.basis_count)
                .map(|b| {
                    (0..outputs.len())
                        .map(|i| coefficients[i * m.basis_count + b])
                        .collect()
                })
                .collect();
            Some(LinearLayout {
                runs,
                coefficients: by_basis,
                range_sides: ranges.iter().map(|r| r.0).collect(),
            })
        } else {
            None
        };
        Self {
            register_kernel: super::super::cpu_defaults::flag("PFS_NN_REGISTER_KERNEL"),
            dense,
            linear,
            cfvs,
            ranges,
            context,
            outputs,
            output_ids,
            w1,
            w2,
            b2,
            coefficients,
            denominator,
            inputs,
        }
    }
    fn forward(
        &self,
        m: &SparseCpuModel,
        dense: bool,
        values: &[f32],
        indices: &[u32],
        out: &mut [f32],
    ) {
        unsafe {
            if dense {
                #[cfg(target_arch = "x86_64")]
                (if self.register_kernel {
                    sparse_nn_forward_f32_register_dense
                } else {
                    sparse_nn_forward_f32_static_dense
                })(
                    self.w1.as_ptr(),
                    m.b1.as_ptr(),
                    m.norm_weight.as_ptr(),
                    m.norm_bias.as_ptr(),
                    m.prelu,
                    self.w2.as_ptr(),
                    self.b2.as_ptr(),
                    m.hidden as u32,
                    values.as_ptr(),
                    values.len() as u32,
                    out.len() as u32,
                    out.as_mut_ptr(),
                );
                #[cfg(not(target_arch = "x86_64"))]
                unreachable!("Dense static inputs require AVX512");
            } else {
                sparse_nn_forward_f32(
                    self.w1.as_ptr(),
                    m.b1.as_ptr(),
                    m.norm_weight.as_ptr(),
                    m.norm_bias.as_ptr(),
                    m.prelu,
                    self.w2.as_ptr(),
                    self.b2.as_ptr(),
                    m.hidden as u32,
                    indices.as_ptr(),
                    values.as_ptr(),
                    values.len() as u32,
                    self.output_ids.as_ptr(),
                    out.len() as u32,
                    out.as_mut_ptr(),
                );
            }
        }
    }
    pub(super) fn supports_child_inputs(&self) -> bool {
        self.dense
    }

    pub(super) fn prepare_basis(
        &self,
        m: &SparseCpuModel,
        basis: usize,
        raw_cfvs: Vec<f32>,
    ) -> PreparedBasisInput {
        assert!(basis < m.basis_count);
        if let Some(layout) = &self.linear {
            let count = self.outputs.len();
            if child_partial_rms_enabled() {
                let mut data = buffer_pool::take(2 * count);
                let (values, products) = data.split_at_mut(count);
                for run in &layout.runs {
                    values[run.compact..run.compact + run.len]
                        .copy_from_slice(&raw_cfvs[run.local..run.local + run.len]);
                }
                let mut sum = 0.0f32;
                for ((&v, product), &coefficient) in values.iter().zip(products).zip(&layout.coefficients[basis]) {
                    sum += v * v;
                    *product = v * coefficient;
                }
                return self.with_partial_hidden(m, basis, PreparedBasisInput {
                    recycled: buffer_pool::enabled(), raw_cfvs,
                    values: Vec::new(), squares: Vec::new(), partial_square_sum: Some(sum),
                    partial_hidden: Vec::new(),
                    baseline_products: Vec::new(), linear: Some(data), count,
                });
            }
            let mut data = buffer_pool::take(3 * count);
            let (values, rest) = data.split_at_mut(count);
            let (squares, products) = rest.split_at_mut(count);
            for run in &layout.runs {
                values[run.compact..run.compact + run.len]
                    .copy_from_slice(&raw_cfvs[run.local..run.local + run.len]);
            }
            for (((&v, square), product), &coefficient) in values
                .iter()
                .zip(squares)
                .zip(products)
                .zip(&layout.coefficients[basis])
            {
                *square = v * v;
                *product = v * coefficient;
            }
            return self.with_partial_hidden(m, basis, PreparedBasisInput {
                recycled: buffer_pool::enabled(),
                raw_cfvs,
                values: Vec::new(),
                squares: Vec::new(),
                partial_square_sum: None,
                partial_hidden: Vec::new(),
                baseline_products: Vec::new(),
                linear: Some(data),
                count,
            });
        }
        let hands = raw_cfvs.len();
        let count = self.cfvs.len() / m.basis_count;
        let values: Vec<f32> = self.cfvs[basis * count..(basis + 1) * count]
            .iter()
            .map(|&(position, _)| raw_cfvs[position - basis * hands])
            .collect();
        let partial_square_sum = child_partial_rms_enabled()
            .then(|| values.iter().map(|v| v * v).sum());
        let squares = if partial_square_sum.is_some() { Vec::new() }
            else { values.iter().map(|v| v * v).collect() };
        let baseline_products = self
            .outputs
            .iter()
            .enumerate()
            .map(|(i, &(local, _))| raw_cfvs[local] * self.coefficients[i * m.basis_count + basis])
            .collect();
        self.with_partial_hidden(m, basis, PreparedBasisInput {
            recycled: false,
            linear: None,
            count: values.len(),
            raw_cfvs,
            values,
            squares,
            partial_square_sum,
            partial_hidden: Vec::new(),
            baseline_products,
        })
    }

    fn with_partial_hidden(&self, m: &SparseCpuModel, basis: usize,
        mut block: PreparedBasisInput) -> PreparedBasisInput {
        if child_partial_hidden_enabled() {
            assert!(self.dense);
            let count = block.count;
            let mut hidden = vec![0f32; m.hidden];
            unsafe { sparse_nn_partial_first_f32(
                self.w1[basis * count * m.hidden..].as_ptr(), block.values().as_ptr(),
                count as u32, m.hidden as u32, hidden.as_mut_ptr()); }
            block.partial_hidden = hidden;
        }
        block
    }

    pub(super) fn prepare_common(
        &self,
        context: &[f32],
        reaches: [&[f32]; 2],
    ) -> PreparedCommonInput {
        let mut features = buffer_pool::take(self.inputs);
        if self.linear.is_some() {
            self.fill_common_linear(&mut features, context, reaches);
            return PreparedCommonInput { features };
        }
        let mut sums = [0f32; 2];
        for &(side, local, _) in &self.ranges {
            sums[side] += reaches[side][local];
        }
        let denominators = [sums[0].max(1e-12), sums[1].max(1e-12)];
        for &(side, local, index) in &self.ranges {
            features[index as usize] = reaches[side][local] / denominators[side];
        }
        for (&index, &value) in self.context.iter().zip(context) {
            features[index as usize] = value;
        }
        PreparedCommonInput { features }
    }

    pub(super) fn infer_children(
        &self,
        m: &SparseCpuModel,
        blocks: &[PreparedBasisInput],
        mut common: PreparedCommonInput,
        hands: usize,
    ) -> Vec<f64> {
        assert!(self.dense);
        assert_eq!(blocks.len(), m.basis_count);
        // The optional experiment groups FP32 additions by child. The default
        // preserves the established basis/hand reduction order exactly.
        let sum: f32 = if child_partial_rms_enabled() {
            blocks.iter().map(|b| b.partial_square_sum.expect("missing child RMS sum")).sum()
        } else { blocks
            .iter()
            .flat_map(|b| b.squares().iter())
            .copied()
            .sum() };
        if sum == 0. {
            return vec![0.; hands];
        }
        let mut scale = (sum / self.denominator as f32).sqrt();
        if !scale.is_finite() || scale == 0. {
            let values: Vec<_> = blocks
                .iter()
                .flat_map(|b| b.values().iter())
                .copied()
                .collect();
            scale = recover_cfv_rms(&values, scale, self.denominator);
        }
        let count = self.cfvs.len() / m.basis_count;
        let prefix = &mut common.features[..self.cfvs.len()];
        if !child_partial_hidden_enabled() {
        #[cfg(feature = "rayon")]
        if blocks.len() > 1 {
            use rayon::prelude::*;
            prefix
                .par_chunks_mut(count)
                .zip(blocks.par_iter())
                .for_each(|(out, b)| {
                    for (dst, &v) in out.iter_mut().zip(b.values()) {
                        *dst = v / scale;
                    }
                });
        } else {
            for (dst, &v) in prefix.iter_mut().zip(blocks[0].values()) {
                *dst = v / scale;
            }
        }
        #[cfg(not(feature = "rayon"))]
        for (out, b) in prefix.chunks_mut(count).zip(blocks) {
            for (dst, &v) in out.iter_mut().zip(b.values()) {
                *dst = v / scale;
            }
        }
        }
        let mut result = vec![0f64; hands];
        SCRATCH.with_borrow_mut(|scratch| {
            scratch.out.resize(self.outputs.len(), 0.);
            if child_partial_hidden_enabled() {
                scratch.hidden.resize(m.hidden, 0.);
                scratch.hidden.fill(0.);
                for block in blocks {
                    assert_eq!(block.partial_hidden.len(), m.hidden);
                    for (dst, &v) in scratch.hidden.iter_mut().zip(&block.partial_hidden) { *dst += v; }
                }
                for (v, &bias) in scratch.hidden.iter_mut().zip(&m.b1) { *v = *v / scale + bias; }
                let offset = self.cfvs.len();
                unsafe {
                    (if self.register_kernel { sparse_nn_forward_f32_register_dense }
                     else { sparse_nn_forward_f32_static_dense })(
                        self.w1[offset * m.hidden..].as_ptr(), scratch.hidden.as_ptr(),
                        m.norm_weight.as_ptr(), m.norm_bias.as_ptr(), m.prelu,
                        self.w2.as_ptr(), self.b2.as_ptr(), m.hidden as u32,
                        common.features[offset..].as_ptr(), (self.inputs-offset) as u32,
                        self.outputs.len() as u32, scratch.out.as_mut_ptr());
                }
            } else {
                self.forward(m, true, &common.features, &[], &mut scratch.out);
            }
            if let Some(layout) = &self.linear {
                scratch.baseline.resize(self.outputs.len(), 0.);
                scratch.baseline.fill(0.);
                for b in blocks {
                    for (dst, &product) in scratch.baseline.iter_mut().zip(b.products()) {
                        *dst += product;
                    }
                }
                for run in &layout.runs {
                    for ((dst, &baseline), &residual) in result[run.local..run.local + run.len]
                        .iter_mut()
                        .zip(&scratch.baseline[run.compact..run.compact + run.len])
                        .zip(&scratch.out[run.compact..run.compact + run.len])
                    {
                        *dst = (baseline + residual * scale) as f64 * 1.;
                    }
                }
            } else {
                for (i, &(local, _)) in self.outputs.iter().enumerate() {
                    let mut baseline = 0f32;
                    for b in blocks {
                        baseline += b.products()[i];
                    }
                    result[local] = (baseline + scratch.out[i] * scale) as f64 * 1.;
                }
            }
        });
        result
    }

    fn fill_common_linear(&self, features: &mut [f32], context: &[f32], reaches: [&[f32]; 2]) {
        let layout = self.linear.as_ref().unwrap();
        let begin = self.cfvs.len();
        let end = begin + self.ranges.len();
        let tail = &mut features[begin..end];
        let mut sums = [0f32; 2];
        // One planned gather in the established canonical OOP/IP order.
        for (dst, &(side, local, _)) in tail.iter_mut().zip(&self.ranges) {
            let value = reaches[side][local];
            *dst = value;
            sums[side] += value;
        }
        let denom = [sums[0].max(1e-12), sums[1].max(1e-12)];
        for (value, &side) in tail.iter_mut().zip(&layout.range_sides) {
            *value /= denom[side];
        }
        features[end..].copy_from_slice(context);
    }

    fn infer_linear(
        &self,
        m: &SparseCpuModel,
        values: &[f32],
        context: &[f32],
        reaches: [&[f32]; 2],
        hands: usize,
    ) -> Vec<f64> {
        let layout = self.linear.as_ref().unwrap();
        let count = self.outputs.len();
        let mut sum = 0f32;
        for block in values.chunks_exact(hands) {
            for run in &layout.runs {
                for &v in &block[run.local..run.local + run.len] {
                    sum += v * v;
                }
            }
        }
        if sum == 0. {
            return vec![0.; hands];
        }
        let mut scale = (sum / self.denominator as f32).sqrt();
        if scale == 0. || !scale.is_finite() {
            let masked: Vec<_> = self.cfvs.iter().map(|&(p, _)| values[p]).collect();
            scale = recover_cfv_rms(&masked, scale, self.denominator);
        }
        let mut result = vec![0f64; hands];
        SCRATCH.with_borrow_mut(|scratch| {
            scratch.values.resize(self.inputs, 0.);
            scratch.baseline.resize(count, 0.);
            scratch.baseline.fill(0.);
            for (basis, block) in values.chunks_exact(hands).enumerate() {
                let normalized = &mut scratch.values[basis * count..(basis + 1) * count];
                for run in &layout.runs {
                    let raw = &block[run.local..run.local + run.len];
                    let coefficients =
                        &layout.coefficients[basis][run.compact..run.compact + run.len];
                    let baseline = &mut scratch.baseline[run.compact..run.compact + run.len];
                    for (((dst, &v), acc), &coefficient) in normalized
                        [run.compact..run.compact + run.len]
                        .iter_mut()
                        .zip(raw)
                        .zip(baseline)
                        .zip(coefficients)
                    {
                        *dst = v / scale;
                        *acc += v * coefficient;
                    }
                }
            }
            self.fill_common_linear(&mut scratch.values, context, reaches);
            scratch.out.resize(count, 0.);
            self.forward(m, true, &scratch.values, &[], &mut scratch.out);
            for run in &layout.runs {
                for ((dst, &baseline), &residual) in result[run.local..run.local + run.len]
                    .iter_mut()
                    .zip(&scratch.baseline[run.compact..run.compact + run.len])
                    .zip(&scratch.out[run.compact..run.compact + run.len])
                {
                    *dst = (baseline + residual * scale) as f64 * 1.;
                }
            }
        });
        result
    }

    pub(super) fn infer(
        &self,
        m: &SparseCpuModel,
        values: &[f32],
        context: &[f32],
        reaches: [&[f32]; 2],
        hands: usize,
    ) -> Vec<f64> {
        if self.linear.is_some() {
            return self.infer_linear(m, values, context, reaches, hands);
        }
        let dense = self.dense;
        let sum: f32 = self.cfvs.iter().map(|&(p, _)| values[p] * values[p]).sum();
        if sum == 0. {
            return vec![0.; hands];
        }
        let mut scale = (sum / self.denominator as f32).sqrt();
        if !scale.is_finite() || scale == 0. {
            let masked: Vec<_> = self.cfvs.iter().map(|&(p, _)| values[p]).collect();
            scale = recover_cfv_rms(&masked, scale, self.denominator);
        }
        let mut result = vec![0.; hands];
        SCRATCH.with_borrow_mut(|scratch| {
            scratch.values.clear();
            scratch.indices.clear();
            if dense {
                scratch.values.resize(self.inputs, 0.);
            }
            let mut push = |index: u32, value: f32| {
                if dense {
                    scratch.values[index as usize] = value;
                } else if value != 0. {
                    scratch.indices.push(index);
                    scratch.values.push(value);
                }
            };
            for &(p, index) in &self.cfvs {
                push(index, values[p] / scale);
            }
            let mut sums = [0f32; 2];
            for &(p, local, _) in &self.ranges {
                sums[p] += reaches[p][local];
            }
            let denominators = [sums[0].max(1e-12), sums[1].max(1e-12)];
            for &(p, local, index) in &self.ranges {
                push(index, reaches[p][local] / denominators[p]);
            }
            for (&index, &value) in self.context.iter().zip(context) {
                push(index, value);
            }
            scratch.out.resize(self.outputs.len(), 0.);
            self.forward(
                m,
                dense,
                &scratch.values,
                &scratch.indices,
                &mut scratch.out,
            );
            for (i, &(local, _)) in self.outputs.iter().enumerate() {
                let mut baseline = 0f32;
                for b in 0..m.basis_count {
                    baseline +=
                        values[b * hands + local] * self.coefficients[i * m.basis_count + b];
                }
                result[local] = (baseline + scratch.out[i] * scale) as f64 * 1.;
            }
        });
        result
    }
}

#[derive(Default)]
struct Scratch {
    hidden: Vec<f32>,
    baseline: Vec<f32>,
    values: Vec<f32>,
    indices: Vec<u32>,
    out: Vec<f32>,
}
thread_local! {static SCRATCH:RefCell<Scratch>=RefCell::new(Scratch::default());}
pub(super) fn diagnostic_worker_scratch_bytes()->u64{SCRATCH.with_borrow(|x|((x.hidden.capacity()+x.baseline.capacity()+x.values.capacity()+x.indices.capacity()+x.out.capacity())*4) as u64)}

#[cfg(test)]
mod partial_hidden_tests {
    use super::*;
    #[test]
    fn child_partial_first_layer_matches_scalar_fma() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        for hidden in [64usize, 128, 256] {
            let n = 37;
            let weights: Vec<f32> = (0..n * hidden)
                .map(|i| ((i * 17 % 101) as f32 - 50.) / 101.).collect();
            let values: Vec<f32> = (0..n).map(|i| ((i * 7 % 19) as f32 - 9.) / 11.).collect();
            let mut actual = vec![0.; hidden];
            unsafe { sparse_nn_partial_first_f32(weights.as_ptr(), values.as_ptr(), n as u32,
                hidden as u32, actual.as_mut_ptr()); }
            for j in 0..hidden {
                let expected = (0..n).fold(0f32, |s, i| values[i].mul_add(weights[i*hidden+j], s));
                assert_eq!(actual[j].to_bits(), expected.to_bits());
            }
        }
    }
    #[test]
    fn split_hidden_preserves_shared_scaling_and_common_features() {
        if !std::is_x86_feature_detected!("avx512f") { return; }
        let hidden=64usize; let count=23usize; let common=9usize; let outputs=13usize;
        for basis in [1usize,2,4,6] {
            let inputs=basis*count+common;
            let weights:Vec<f32>=(0..inputs*hidden).map(|i| ((i*17%113) as f32-56.)/137.).collect();
            let b1=vec![0.02;hidden]; let nw=vec![1.;hidden]; let nb=vec![0.01;hidden];
            let w2:Vec<f32>=(0..outputs*hidden).map(|i| ((i*11%37) as f32-18.)/89.).collect();
            let b2=vec![0.03;outputs];
            for scale in [1e-8f32,1.,1e8] {
                let mut values:Vec<f32>=(0..inputs).map(|i| ((i*13%31) as f32-15.)/19.).collect();
                let mut accum=vec![0.;hidden];
                for b in 0..basis {
                    let raw:Vec<f32>=values[b*count..(b+1)*count].iter().map(|v| v*scale).collect();
                    for (dst,&raw_value) in values[b*count..(b+1)*count].iter_mut().zip(&raw) { *dst=raw_value/scale; }
                    let mut partial=vec![0.;hidden];
                    unsafe { sparse_nn_partial_first_f32(weights[b*count*hidden..].as_ptr(),raw.as_ptr(),count as u32,hidden as u32,partial.as_mut_ptr()); }
                    for (dst,&v) in accum.iter_mut().zip(&partial) { *dst+=v; }
                }
                for (v,&bias) in accum.iter_mut().zip(&b1) { *v=*v/scale+bias; }
                let mut baseline=vec![0.;outputs]; let mut split=vec![0.;outputs];
                unsafe {
                    sparse_nn_forward_f32_register_dense(weights.as_ptr(),b1.as_ptr(),nw.as_ptr(),nb.as_ptr(),0.1,w2.as_ptr(),b2.as_ptr(),hidden as u32,values.as_ptr(),inputs as u32,outputs as u32,baseline.as_mut_ptr());
                    sparse_nn_forward_f32_register_dense(weights[basis*count*hidden..].as_ptr(),accum.as_ptr(),nw.as_ptr(),nb.as_ptr(),0.1,w2.as_ptr(),b2.as_ptr(),hidden as u32,values[basis*count..].as_ptr(),common as u32,outputs as u32,split.as_mut_ptr());
                }
                for (&a,&b) in baseline.iter().zip(&split) { assert!((a-b).abs()<1e-5,"basis={basis} scale={scale} baseline={a} split={b}"); }
            }
        }
    }
}
