use super::*;
use crate::sliceop::*;
#[cfg(target_arch = "x86_64")]
use std::arch::asm;
use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::path::Path;
use std::sync::atomic::Ordering;

const FLOP_ALLIN_CACHE_MAGIC: &[u8; 8] = b"PFAMAT01";
const TURN_ALLIN_CACHE_MAGIC: &[u8; 8] = b"PTAMAT01";
const CANONICAL_HANDS: usize = 52 * 51 / 2;

thread_local! {
    static EXACT_ALLIN_SPARSE_SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static EXACT_ALLIN_F32_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Dense mixed-precision kernel for the exact all-in experiment. Payouts and
/// accumulation remain f64; only the already-f32 reach input and final CFV are
/// f32. The caller performs runtime AVX-512 detection.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn exact_allin_dense_avx512(
    matrix: &[f64],
    x: &[f32],
    output: &mut [MaybeUninit<f32>],
    denominator: f64,
) {
    let cols = x.len();
    for (row, destination) in output.iter_mut().enumerate() {
        let weights = matrix.as_ptr().add(row * cols);
        let input = x.as_ptr();
        let groups = cols / 32;
        let remainder = cols % 32;
        let full_vectors = remainder / 8;
        let tail = remainder & 7;
        let mask = if tail == 0 { 0 } else { (1u32 << tail) - 1 };
        let sum: f64;
        asm!(
            "vxorpd zmm0, zmm0, zmm0", "vxorpd zmm1, zmm1, zmm1",
            "vxorpd zmm2, zmm2, zmm2", "vxorpd zmm3, zmm3, zmm3",
            "test {groups}, {groups}", "jz 3f", "2:",
            "vcvtps2pd zmm4, ymmword ptr [{input}]", "vfmadd231pd zmm0, zmm4, [{weights}]",
            "vcvtps2pd zmm5, ymmword ptr [{input} + 32]", "vfmadd231pd zmm1, zmm5, [{weights} + 64]",
            "vcvtps2pd zmm6, ymmword ptr [{input} + 64]", "vfmadd231pd zmm2, zmm6, [{weights} + 128]",
            "vcvtps2pd zmm7, ymmword ptr [{input} + 96]", "vfmadd231pd zmm3, zmm7, [{weights} + 192]",
            "add {input}, 128", "add {weights}, 256", "dec {groups}", "jnz 2b", "3:",
            "test {full_vectors}, {full_vectors}", "jz 5f", "4:",
            "vcvtps2pd zmm4, ymmword ptr [{input}]", "vfmadd231pd zmm0, zmm4, [{weights}]",
            "add {input}, 32", "add {weights}, 64", "dec {full_vectors}", "jnz 4b", "5:",
            "test {mask:e}, {mask:e}", "jz 6f", "kmovd k1, {mask:e}",
            "vmovups ymm4 {{k1}}{{z}}, [{input}]", "vcvtps2pd zmm4, ymm4",
            "vmovupd zmm5 {{k1}}{{z}}, [{weights}]", "vfmadd231pd zmm0, zmm4, zmm5", "6:",
            "vaddpd zmm0, zmm0, zmm1", "vaddpd zmm2, zmm2, zmm3", "vaddpd zmm0, zmm0, zmm2",
            "vextractf64x4 ymm1, zmm0, 1", "vaddpd ymm0, ymm0, ymm1",
            "vextractf128 xmm1, ymm0, 1", "vaddpd xmm0, xmm0, xmm1", "vhaddpd xmm0, xmm0, xmm0",
            input=inout(reg)input=>_, weights=inout(reg)weights=>_, groups=inout(reg)groups=>_,
            full_vectors=inout(reg)full_vectors=>_, mask=in(reg)mask, lateout("xmm0")sum,
            out("zmm1")_, out("zmm2")_, out("zmm3")_, out("zmm4")_, out("zmm5")_,
            out("zmm6")_, out("zmm7")_, out("k1")_, options(nostack)
        );
        destination.write((sum / denominator) as f32);
    }
}

/// All-f32 counterpart used only by the explicit accuracy experiment.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn exact_allin_dense_f32_avx512(
    matrix: &[f32],
    x: &[f32],
    output: &mut [MaybeUninit<f32>],
    denominator: f32,
) {
    let cols = x.len();
    for (row, destination) in output.iter_mut().enumerate() {
        let weights = matrix.as_ptr().add(row * cols);
        let input = x.as_ptr();
        let groups = cols / 64;
        let remainder = cols % 64;
        let full_vectors = remainder / 16;
        let tail = remainder & 15;
        let mask = if tail == 0 { 0 } else { (1u32 << tail) - 1 };
        let sum: f32;
        asm!(
            "vxorps zmm0, zmm0, zmm0", "vxorps zmm1, zmm1, zmm1",
            "vxorps zmm2, zmm2, zmm2", "vxorps zmm3, zmm3, zmm3",
            "test {groups}, {groups}", "jz 3f", "2:",
            "vmovups zmm4, [{input}]", "vfmadd231ps zmm0, zmm4, [{weights}]",
            "vmovups zmm5, [{input} + 64]", "vfmadd231ps zmm1, zmm5, [{weights} + 64]",
            "vmovups zmm6, [{input} + 128]", "vfmadd231ps zmm2, zmm6, [{weights} + 128]",
            "vmovups zmm7, [{input} + 192]", "vfmadd231ps zmm3, zmm7, [{weights} + 192]",
            "add {input}, 256", "add {weights}, 256", "dec {groups}", "jnz 2b", "3:",
            "test {full_vectors}, {full_vectors}", "jz 5f", "4:",
            "vmovups zmm4, [{input}]", "vfmadd231ps zmm0, zmm4, [{weights}]",
            "add {input}, 64", "add {weights}, 64", "dec {full_vectors}", "jnz 4b", "5:",
            "test {mask:e}, {mask:e}", "jz 6f", "kmovd k1, {mask:e}",
            "vmovups zmm4 {{k1}}{{z}}, [{input}]", "vmovups zmm5 {{k1}}{{z}}, [{weights}]",
            "vfmadd231ps zmm0, zmm4, zmm5", "6:",
            "vaddps zmm0, zmm0, zmm1", "vaddps zmm2, zmm2, zmm3", "vaddps zmm0, zmm0, zmm2",
            "vextractf32x8 ymm1, zmm0, 1", "vaddps ymm0, ymm0, ymm1",
            "vextractf128 xmm1, ymm0, 1", "vaddps xmm0, xmm0, xmm1",
            "vhaddps xmm0, xmm0, xmm0", "vhaddps xmm0, xmm0, xmm0",
            input=inout(reg)input=>_, weights=inout(reg)weights=>_, groups=inout(reg)groups=>_,
            full_vectors=inout(reg)full_vectors=>_, mask=in(reg)mask, lateout("xmm0")sum,
            out("zmm1")_, out("zmm2")_, out("zmm3")_, out("zmm4")_, out("zmm5")_,
            out("zmm6")_, out("zmm7")_, out("k1")_, options(nostack)
        );
        destination.write(sum / denominator);
    }
}

fn matrix_checksum(values: &[f64]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &value in values {
        for byte in value.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

#[inline]
fn min(x: f64, y: f64) -> f64 {
    if x < y {
        x
    } else {
        y
    }
}

impl PostFlopGame {
    pub(super) fn build_flop_allin_payout_matrix(&self) -> Vec<f64> {
        let canonical_hands: Vec<_> = (0..CANONICAL_HANDS).map(index_to_card_pair).collect();
        let hand_masks: Vec<u64> = canonical_hands
            .iter()
            .map(|&(c1, c2)| (1u64 << c1) | (1u64 << c2))
            .collect();
        let (runouts, strengths) =
            crate::card::flop_runout_showdown_strengths(self.card_config.flop, &canonical_hands);
        // The physical tree has 49*48 ordered runouts. We enumerate each
        // unordered public runout once, so each compatible private-hand pair
        // contributes through 990 unordered runouts. Each one represents two
        // physical orders, so normalize by 990 (equivalently, double every
        // unordered contribution and divide by 45*44 = 1980).
        assert_eq!(runouts.len(), 1176);
        let unordered_runout_count = 45.0 * 44.0 / 2.0;

        // Store the payout numerator before division by the current range-pair
        // combination mass. This keeps the full matrix reusable across ranges.
        let pot = (self.tree_config.starting_pot + 2 * self.tree_config.effective_stack) as f64;
        let half_pot = 0.5 * pot;
        let rake = min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);
        let amount_win = half_pot - rake;
        let amount_lose = -half_pot;
        let amount_tie = -0.5 * rake;
        let mut matrix = vec![0.0f64; CANONICAL_HANDS * CANONICAL_HANDS];
        let flop_mask = self
            .card_config
            .flop
            .iter()
            .fold(0u64, |mask, &card| mask | (1u64 << card));

        for (runout_index, &(turn, river)) in runouts.iter().enumerate() {
            let board_mask = flop_mask | (1u64 << turn) | (1u64 << river);
            let row =
                &strengths[runout_index * CANONICAL_HANDS..(runout_index + 1) * CANONICAL_HANDS];
            for hero in 0..CANONICAL_HANDS {
                let hero_strength = row[hero];
                if hero_strength == u16::MAX || hand_masks[hero] & board_mask != 0 {
                    continue;
                }
                let base = hero * CANONICAL_HANDS;
                for opponent in 0..CANONICAL_HANDS {
                    if hand_masks[hero] & hand_masks[opponent] != 0 {
                        continue;
                    }
                    let opponent_strength = row[opponent];
                    if opponent_strength == u16::MAX {
                        continue;
                    }
                    matrix[base + opponent] += if hero_strength > opponent_strength {
                        amount_win
                    } else if hero_strength < opponent_strength {
                        amount_lose
                    } else {
                        amount_tie
                    };
                }
            }
        }

        matrix
            .into_iter()
            .map(|value| value / unordered_runout_count)
            .collect()
    }

    pub(super) fn build_turn_allin_payout_matrix(&self, turn: Card) -> Result<Vec<f64>, String> {
        if self.card_config.flop.contains(&turn) || turn >= 52 {
            return Err(format!(
                "invalid turn card for all-in payout matrix: {turn}"
            ));
        }
        let canonical_hands: Vec<_> = (0..CANONICAL_HANDS).map(index_to_card_pair).collect();
        let hand_masks: Vec<u64> = canonical_hands
            .iter()
            .map(|&(c1, c2)| (1u64 << c1) | (1u64 << c2))
            .collect();
        let rivers: Vec<Card> = (0..52u8)
            .filter(|river| *river != turn && !self.card_config.flop.contains(river))
            .collect();
        assert_eq!(rivers.len(), 48);
        let strengths = crate::card::static_river_hand_strengths(
            self.card_config.flop,
            turn,
            &rivers,
            &canonical_hands,
        );

        let pot = (self.tree_config.starting_pot + 2 * self.tree_config.effective_stack) as f64;
        let half_pot = 0.5 * pot;
        let rake = min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);
        let amount_win = half_pot - rake;
        let amount_lose = -half_pot;
        let amount_tie = -0.5 * rake;
        let public_mask = self
            .card_config
            .flop
            .iter()
            .fold(1u64 << turn, |mask, &card| mask | (1u64 << card));
        let mut matrix = vec![0.0f64; CANONICAL_HANDS * CANONICAL_HANDS];

        for (river_index, &river) in rivers.iter().enumerate() {
            let board_mask = public_mask | (1u64 << river);
            let row =
                &strengths[river_index * CANONICAL_HANDS..(river_index + 1) * CANONICAL_HANDS];
            for hero in 0..CANONICAL_HANDS {
                let (_, hero_strength) = row[hero];
                if row[hero].0 == u8::MAX || hand_masks[hero] & board_mask != 0 {
                    continue;
                }
                let base = hero * CANONICAL_HANDS;
                for opponent in 0..CANONICAL_HANDS {
                    if hand_masks[hero] & hand_masks[opponent] != 0 {
                        continue;
                    }
                    let (_, opponent_strength) = row[opponent];
                    if row[opponent].0 == u8::MAX {
                        continue;
                    }
                    matrix[base + opponent] += if hero_strength > opponent_strength {
                        amount_win
                    } else if hero_strength < opponent_strength {
                        amount_lose
                    } else {
                        amount_tie
                    };
                }
            }
        }

        // Of the 48 board-legal rivers, four are blocked by a fixed pair of
        // disjoint private hands. This is the ordinary exact turn chance
        // factor used by the physical traversal.
        for value in &mut matrix {
            *value /= 44.0;
        }
        Ok(matrix)
    }

    fn flop_allin_cache_metadata(&self) -> ([Card; 3], i32, i32, u64, u64) {
        let mut flop = self.card_config.flop;
        flop.sort_unstable();
        (
            flop,
            self.tree_config.starting_pot,
            self.tree_config.effective_stack,
            self.tree_config.rake_rate.to_bits(),
            self.tree_config.rake_cap.to_bits(),
        )
    }

    pub(super) fn load_flop_allin_payout_matrix_cache(
        &self,
        path: &Path,
    ) -> Result<Vec<f64>, String> {
        const HEADER_LEN: usize = 8 + 3 + 4 + 4 + 8 + 8 + 8 + 8;
        let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
        let expected_len = HEADER_LEN + CANONICAL_HANDS * CANONICAL_HANDS * 8;
        if bytes.len() != expected_len || &bytes[..8] != FLOP_ALLIN_CACHE_MAGIC {
            return Err(format!(
                "invalid flop all-in matrix cache: {}",
                path.display()
            ));
        }
        let (flop, pot, stack, rake_rate, rake_cap) = self.flop_allin_cache_metadata();
        if bytes[8..11] != flop
            || i32::from_le_bytes(bytes[11..15].try_into().unwrap()) != pot
            || i32::from_le_bytes(bytes[15..19].try_into().unwrap()) != stack
            || u64::from_le_bytes(bytes[19..27].try_into().unwrap()) != rake_rate
            || u64::from_le_bytes(bytes[27..35].try_into().unwrap()) != rake_cap
            || u64::from_le_bytes(bytes[35..43].try_into().unwrap())
                != (CANONICAL_HANDS * CANONICAL_HANDS) as u64
        {
            return Err(format!(
                "flop all-in matrix cache metadata mismatch: {}",
                path.display()
            ));
        }
        let expected_checksum = u64::from_le_bytes(bytes[43..51].try_into().unwrap());
        let values: Vec<f64> = bytes[HEADER_LEN..]
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        if matrix_checksum(&values) != expected_checksum {
            return Err(format!(
                "flop all-in matrix cache checksum mismatch: {}",
                path.display()
            ));
        }
        Ok(values)
    }

    pub(super) fn save_flop_allin_payout_matrix_cache(
        &self,
        path: &Path,
        values: &[f64],
    ) -> Result<(), String> {
        use std::io::{BufWriter, Write};
        if values.len() != CANONICAL_HANDS * CANONICAL_HANDS {
            return Err("canonical flop all-in matrix has the wrong size".into());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
        let file = std::fs::File::create(&temporary).map_err(|error| error.to_string())?;
        let mut writer = BufWriter::new(file);
        let (flop, pot, stack, rake_rate, rake_cap) = self.flop_allin_cache_metadata();
        writer
            .write_all(FLOP_ALLIN_CACHE_MAGIC)
            .and_then(|_| writer.write_all(&flop))
            .and_then(|_| writer.write_all(&pot.to_le_bytes()))
            .and_then(|_| writer.write_all(&stack.to_le_bytes()))
            .and_then(|_| writer.write_all(&rake_rate.to_le_bytes()))
            .and_then(|_| writer.write_all(&rake_cap.to_le_bytes()))
            .and_then(|_| {
                writer.write_all(&((CANONICAL_HANDS * CANONICAL_HANDS) as u64).to_le_bytes())
            })
            .and_then(|_| writer.write_all(&matrix_checksum(values).to_le_bytes()))
            .map_err(|error| error.to_string())?;
        for &value in values {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|error| error.to_string())?;
        }
        writer.flush().map_err(|error| error.to_string())?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| error.to_string())?;
        std::fs::rename(&temporary, path).map_err(|error| error.to_string())?;
        Ok(())
    }

    pub(super) fn load_turn_allin_payout_matrix_cache(
        &self,
        path: &Path,
        turn: Card,
    ) -> Result<Vec<f64>, String> {
        const HEADER_LEN: usize = 8 + 3 + 1 + 4 + 4 + 8 + 8 + 8 + 8;
        let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
        let expected_len = HEADER_LEN + CANONICAL_HANDS * CANONICAL_HANDS * 8;
        if bytes.len() != expected_len || &bytes[..8] != TURN_ALLIN_CACHE_MAGIC {
            return Err(format!(
                "invalid turn all-in matrix cache: {}",
                path.display()
            ));
        }
        let (flop, pot, stack, rake_rate, rake_cap) = self.flop_allin_cache_metadata();
        if bytes[8..11] != flop
            || bytes[11] != turn
            || i32::from_le_bytes(bytes[12..16].try_into().unwrap()) != pot
            || i32::from_le_bytes(bytes[16..20].try_into().unwrap()) != stack
            || u64::from_le_bytes(bytes[20..28].try_into().unwrap()) != rake_rate
            || u64::from_le_bytes(bytes[28..36].try_into().unwrap()) != rake_cap
            || u64::from_le_bytes(bytes[36..44].try_into().unwrap())
                != (CANONICAL_HANDS * CANONICAL_HANDS) as u64
        {
            return Err(format!(
                "turn all-in matrix cache metadata mismatch: {}",
                path.display()
            ));
        }
        let expected_checksum = u64::from_le_bytes(bytes[44..52].try_into().unwrap());
        let values: Vec<f64> = bytes[HEADER_LEN..]
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        if matrix_checksum(&values) != expected_checksum {
            return Err(format!(
                "turn all-in matrix cache checksum mismatch: {}",
                path.display()
            ));
        }
        Ok(values)
    }

    pub(super) fn save_turn_allin_payout_matrix_cache(
        &self,
        path: &Path,
        turn: Card,
        values: &[f64],
    ) -> Result<(), String> {
        use std::io::{BufWriter, Write};
        if values.len() != CANONICAL_HANDS * CANONICAL_HANDS {
            return Err("canonical turn all-in matrix has the wrong size".into());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
        let file = std::fs::File::create(&temporary).map_err(|error| error.to_string())?;
        let mut writer = BufWriter::new(file);
        let (flop, pot, stack, rake_rate, rake_cap) = self.flop_allin_cache_metadata();
        writer
            .write_all(TURN_ALLIN_CACHE_MAGIC)
            .and_then(|_| writer.write_all(&flop))
            .and_then(|_| writer.write_all(&[turn]))
            .and_then(|_| writer.write_all(&pot.to_le_bytes()))
            .and_then(|_| writer.write_all(&stack.to_le_bytes()))
            .and_then(|_| writer.write_all(&rake_rate.to_le_bytes()))
            .and_then(|_| writer.write_all(&rake_cap.to_le_bytes()))
            .and_then(|_| {
                writer.write_all(&((CANONICAL_HANDS * CANONICAL_HANDS) as u64).to_le_bytes())
            })
            .and_then(|_| writer.write_all(&matrix_checksum(values).to_le_bytes()))
            .map_err(|error| error.to_string())?;
        for &value in values {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|error| error.to_string())?;
        }
        writer.flush().map_err(|error| error.to_string())?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| error.to_string())?;
        std::fs::rename(&temporary, path).map_err(|error| error.to_string())?;
        Ok(())
    }

    fn called_allin_payout_matrix(&self, node: &PostFlopNode) -> Option<&FlopAllinPayoutMatrix> {
        if node.river != NOT_DEALT || (node.amount != self.tree_config.effective_stack
            && !self.check_check_river_nodes.contains(&node.id)) {
            return None;
        }
        if node.turn == NOT_DEALT {
            self.flop_allin_payout_matrix.as_ref()
        } else {
            self.turn_allin_payout_matrices.get(&node.turn)
        }
    }

    pub(super) fn evaluate_called_allin_matrix(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) -> bool {
        let Some(matrix) = self.called_allin_payout_matrix(node) else {
            return false;
        };

        let denominator = self.showdown_matrix_denominator(node);
        let opponent_hands = matrix.local_hands[player ^ 1];
        assert_eq!(result.len(), matrix.local_hands[player]);
        assert_eq!(cfreach.len(), opponent_hands);
        if super::base::EXACT_ALLIN_MATRIX_F32_EVAL.load(Ordering::Relaxed) {
            let hero_hands = result.len();
            let nonzeros = cfreach.iter().filter(|&&reach| reach != 0.0).count();
            if 2 * nonzeros <= opponent_hands {
                let transposed = &matrix.values_transposed_f32[player];
                assert_eq!(transposed.len(), hero_hands * opponent_hands);
                EXACT_ALLIN_F32_SCRATCH.with(|scratch| {
                    let mut scratch = scratch.borrow_mut();
                    scratch.resize(hero_hands, 0.0);
                    scratch.fill(0.0);
                    for (opponent, &reach) in cfreach.iter().enumerate() {
                        if reach == 0.0 {
                            continue;
                        }
                        let weights =
                            &transposed[opponent * hero_hands..(opponent + 1) * hero_hands];
                        for (value, &payout) in scratch.iter_mut().zip(weights) {
                            *value += payout * reach;
                        }
                    }
                    let denominator = denominator as f32;
                    for (output, &value) in result.iter_mut().zip(scratch.iter()) {
                        output.write(value / denominator);
                    }
                });
                return true;
            }
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq")
            {
                unsafe {
                    exact_allin_dense_f32_avx512(
                        &matrix.values_f32[player],
                        cfreach,
                        result,
                        denominator as f32,
                    );
                }
                return true;
            }
            for (hero, output) in result.iter_mut().enumerate() {
                let row =
                    &matrix.values_f32[player][hero * opponent_hands..(hero + 1) * opponent_hands];
                let mut value = 0.0f32;
                for (&payout, &reach) in row.iter().zip(cfreach) {
                    value += payout * reach;
                }
                output.write(value / denominator as f32);
            }
            return true;
        }
        if super::base::EXACT_ALLIN_MATRIX_SPARSE_EVAL.load(Ordering::Relaxed) {
            let hero_hands = result.len();
            let transposed = &matrix.values_transposed[player];
            assert_eq!(transposed.len(), hero_hands * opponent_hands);
            // Output-major sparse evaluation wins only once enough reaches are
            // exactly zero. Keep the original row-major path above the measured
            // 50% density crossover, which is common in early CFR iterations.
            let nonzeros = cfreach.iter().filter(|&&reach| reach != 0.0).count();
            if 2 * nonzeros <= opponent_hands {
                EXACT_ALLIN_SPARSE_SCRATCH.with(|scratch| {
                    let mut scratch = scratch.borrow_mut();
                    scratch.resize(hero_hands, 0.0);
                    scratch.fill(0.0);
                    for (opponent, &reach) in cfreach.iter().enumerate() {
                        if reach == 0.0 {
                            continue;
                        }
                        let weights =
                            &transposed[opponent * hero_hands..(opponent + 1) * hero_hands];
                        let reach = reach as f64;
                        for (value, &payout) in scratch.iter_mut().zip(weights) {
                            *value += payout * reach;
                        }
                    }
                    for (output, &value) in result.iter_mut().zip(scratch.iter()) {
                        output.write((value / denominator) as f32);
                    }
                });
                return true;
            }
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq")
            {
                unsafe {
                    exact_allin_dense_avx512(&matrix.values[player], cfreach, result, denominator);
                }
                return true;
            }
        }
        for (hero, output) in result.iter_mut().enumerate() {
            let row = &matrix.values[player][hero * opponent_hands..(hero + 1) * opponent_hands];
            let mut value = 0.0f64;
            for (&payout, &reach) in row.iter().zip(cfreach) {
                value += payout * reach as f64;
            }
            output.write((value / denominator) as f32);
        }
        true
    }

    pub(super) fn compare_called_allin_matrix(
        &self,
        result: &[f64],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) -> Option<f64> {
        let matrix = self.called_allin_payout_matrix(node)?;
        let denominator = self.showdown_matrix_denominator(node);
        let opponent_hands = matrix.local_hands[player ^ 1];
        assert_eq!(result.len(), matrix.local_hands[player]);
        assert_eq!(cfreach.len(), opponent_hands);
        let mut max_diff = 0.0f64;
        for (hero, &physical) in result.iter().enumerate() {
            let row = &matrix.values[player][hero * opponent_hands..(hero + 1) * opponent_hands];
            let mut value = 0.0f64;
            for (&payout, &reach) in row.iter().zip(cfreach) {
                value += payout * reach as f64;
            }
            let matrix_value = value / denominator;
            max_diff = max_diff.max((physical - matrix_value).abs());
        }
        Some(max_diff)
    }

    pub(super) fn evaluate_internal(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);
        let amount_win = (half_pot - rake) / self.num_combinations;
        let amount_lose = -half_pot / self.num_combinations;

        static STREAM_FOLDS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG
            && *STREAM_FOLDS.get_or_init(|| std::env::var("PFS_STREAM_FOLD_EXPERIMENT").as_deref() == Ok("1"))
        {
            let valid = if node.river != NOT_DEALT {
                &self.valid_indices_river[card_pair_to_index(node.turn, node.river)]
            } else if node.turn != NOT_DEALT { &self.valid_indices_turn[node.turn as usize] }
            else { &self.valid_indices_flop };
            let mut total = 0.0f64;
            let mut minus = [0.0f64; 52];
            for &i in &valid[player ^ 1] {
                let value = cfreach[i as usize];
                if value != 0.0 {
                    let (a,b) = self.private_cards[player ^ 1][i as usize];
                    let value = value as f64;
                    total += value; minus[a as usize] += value; minus[b as usize] += value;
                }
            }
            if total == 0.0 { for out in result { out.write(0.0); } return; }
            let payoff = if (node.player & PLAYER_MASK) as usize != player { amount_win } else { amount_lose };
            let board_mask = (if node.turn == NOT_DEALT { 0 } else { 1u64 << node.turn })
                | (if node.river == NOT_DEALT { 0 } else { 1u64 << node.river });
            for (i, (out, &(a,b))) in result.iter_mut().zip(&self.private_cards[player]).enumerate() {
                if board_mask & ((1u64 << a) | (1u64 << b)) != 0 { out.write(0.0); continue; }
                let same = self.same_hand_index[player][i];
                let same_reach = if same == u16::MAX { 0.0 } else { cfreach[same as usize] as f64 };
                let compatible = total + same_reach - minus[a as usize] - minus[b as usize];
                out.write((payoff * compatible) as f32);
            }
            return;
        }

        let player_cards = &self.private_cards[player];
        let opponent_cards = &self.private_cards[player ^ 1];

        let mut cfreach_sum = 0.0;
        let mut cfreach_minus = [0.0; 52];

        result.iter_mut().for_each(|v| {
            v.write(0.0);
        });

        let result = unsafe { &mut *(result as *mut _ as *mut [f32]) };

        // someone folded
        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
            let folded_player = node.player & PLAYER_MASK;
            let payoff = if folded_player as usize != player {
                amount_win
            } else {
                amount_lose
            };

            let valid_indices = if node.river != NOT_DEALT {
                &self.valid_indices_river[card_pair_to_index(node.turn, node.river)]
            } else if node.turn != NOT_DEALT {
                &self.valid_indices_turn[node.turn as usize]
            } else {
                &self.valid_indices_flop
            };

            let opponent_indices = &valid_indices[player ^ 1];
            for &i in opponent_indices {
                unsafe {
                    let cfreach_i = *cfreach.get_unchecked(i as usize);
                    if cfreach_i != 0.0 {
                        let (c1, c2) = *opponent_cards.get_unchecked(i as usize);
                        let cfreach_i_f64 = cfreach_i as f64;
                        cfreach_sum += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                    }
                }
            }

            if cfreach_sum == 0.0 {
                return;
            }

            let player_indices = &valid_indices[player];
            let same_hand_index = &self.same_hand_index[player];
            for &i in player_indices {
                unsafe {
                    let (c1, c2) = *player_cards.get_unchecked(i as usize);
                    let same_i = *same_hand_index.get_unchecked(i as usize);
                    let cfreach_same = if same_i == u16::MAX {
                        0.0
                    } else {
                        *cfreach.get_unchecked(same_i as usize) as f64
                    };
                    // inclusion-exclusion principle
                    let cfreach = cfreach_sum + cfreach_same
                        - *cfreach_minus.get_unchecked(c1 as usize)
                        - *cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(i as usize) = (payoff * cfreach) as f32;
                }
            }
        }
        // showdown (optimized for no rake; 2-pass)
        else if rake == 0.0 {
            let pair_index = card_pair_to_index(node.turn, node.river);
            let hand_strength = &self.hand_strength[pair_index];
            let player_strength = &hand_strength[player];
            let opponent_strength = &hand_strength[player ^ 1];

            let valid_player_strength = &player_strength[1..player_strength.len() - 1];
            let mut i = 1;

            for &StrengthItem { strength, index } in valid_player_strength {
                unsafe {
                    while opponent_strength.get_unchecked(i).strength < strength {
                        let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                        let cfreach_i = *cfreach.get_unchecked(opponent_index);
                        if cfreach_i != 0.0 {
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i_f64 = cfreach_i as f64;
                            cfreach_sum += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                        }
                        i += 1;
                    }
                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(index as usize) = (amount_win * cfreach) as f32;
                }
            }

            cfreach_sum = 0.0;
            cfreach_minus.fill(0.0);
            i = opponent_strength.len() - 2;

            for &StrengthItem { strength, index } in valid_player_strength.iter().rev() {
                unsafe {
                    while opponent_strength.get_unchecked(i).strength > strength {
                        let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                        let cfreach_i = *cfreach.get_unchecked(opponent_index);
                        if cfreach_i != 0.0 {
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i_f64 = cfreach_i as f64;
                            cfreach_sum += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                        }
                        i -= 1;
                    }
                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(index as usize) += (amount_lose * cfreach) as f32;
                }
            }
        }
        // showdown (raked; 3-pass)
        else {
            let amount_tie = -0.5 * rake / self.num_combinations;
            let same_hand_index = &self.same_hand_index[player];

            let pair_index = card_pair_to_index(node.turn, node.river);
            let hand_strength = &self.hand_strength[pair_index];
            let player_strength = &hand_strength[player];
            let opponent_strength = &hand_strength[player ^ 1];

            let valid_player_strength = &player_strength[1..player_strength.len() - 1];
            let valid_opponent_strength = &opponent_strength[1..opponent_strength.len() - 1];

            for &StrengthItem { index, .. } in valid_opponent_strength {
                unsafe {
                    let cfreach_i = *cfreach.get_unchecked(index as usize);
                    if cfreach_i != 0.0 {
                        let (c1, c2) = *opponent_cards.get_unchecked(index as usize);
                        let cfreach_i_f64 = cfreach_i as f64;
                        cfreach_sum += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                    }
                }
            }

            if cfreach_sum == 0.0 {
                return;
            }

            let mut cfreach_sum_win = 0.0;
            let mut cfreach_sum_tie = 0.0;
            let mut cfreach_minus_win = [0.0; 52];
            let mut cfreach_minus_tie = [0.0; 52];

            let mut i = 1;
            let mut j = 1;
            let mut prev_strength = 0; // strength is always > 0

            for &StrengthItem { strength, index } in valid_player_strength {
                unsafe {
                    if strength > prev_strength {
                        prev_strength = strength;

                        if i < j {
                            cfreach_sum_win = cfreach_sum_tie;
                            cfreach_minus_win = cfreach_minus_tie;
                            i = j;
                        }

                        while opponent_strength.get_unchecked(i).strength < strength {
                            let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i = *cfreach.get_unchecked(opponent_index) as f64;
                            cfreach_sum_win += cfreach_i;
                            *cfreach_minus_win.get_unchecked_mut(c1 as usize) += cfreach_i;
                            *cfreach_minus_win.get_unchecked_mut(c2 as usize) += cfreach_i;
                            i += 1;
                        }

                        if j < i {
                            cfreach_sum_tie = cfreach_sum_win;
                            cfreach_minus_tie = cfreach_minus_win;
                            j = i;
                        }

                        while opponent_strength.get_unchecked(j).strength == strength {
                            let opponent_index = opponent_strength.get_unchecked(j).index as usize;
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_j = *cfreach.get_unchecked(opponent_index) as f64;
                            cfreach_sum_tie += cfreach_j;
                            *cfreach_minus_tie.get_unchecked_mut(c1 as usize) += cfreach_j;
                            *cfreach_minus_tie.get_unchecked_mut(c2 as usize) += cfreach_j;
                            j += 1;
                        }
                    }

                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach_total = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    let cfreach_win = cfreach_sum_win
                        - cfreach_minus_win.get_unchecked(c1 as usize)
                        - cfreach_minus_win.get_unchecked(c2 as usize);
                    let cfreach_tie = cfreach_sum_tie
                        - cfreach_minus_tie.get_unchecked(c1 as usize)
                        - cfreach_minus_tie.get_unchecked(c2 as usize);
                    let same_i = *same_hand_index.get_unchecked(index as usize);
                    let cfreach_same = if same_i == u16::MAX {
                        0.0
                    } else {
                        *cfreach.get_unchecked(same_i as usize) as f64
                    };

                    let cfvalue = amount_win * cfreach_win
                        + amount_tie * (cfreach_tie - cfreach_win + cfreach_same)
                        + amount_lose * (cfreach_total - cfreach_tie);
                    *result.get_unchecked_mut(index as usize) = cfvalue as f32;
                }
            }
        }
    }

    pub(super) fn evaluate_internal_bunching(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);
        let amount_win = ((half_pot - rake) / self.bunching_num_combinations) as f32;
        let amount_lose = (-half_pot / self.bunching_num_combinations) as f32;
        let amount_tie = (-0.5 * rake / self.bunching_num_combinations) as f32;
        let opponent_len = self.private_cards[player ^ 1].len();

        // someone folded
        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
            let folded_player = node.player & PLAYER_MASK;
            let payoff = if folded_player as usize != player {
                amount_win
            } else {
                amount_lose
            };

            let indices = if node.river != NOT_DEALT {
                &self.bunching_num_river[player][card_pair_to_index(node.turn, node.river)]
            } else if node.turn != NOT_DEALT {
                &self.bunching_num_turn[player][node.turn as usize]
            } else {
                &self.bunching_num_flop[player]
            };

            result.iter_mut().zip(indices).for_each(|(r, &index)| {
                if index != 0 {
                    let slice = &self.bunching_arena[index..index + opponent_len];
                    r.write(payoff * inner_product(cfreach, slice));
                } else {
                    r.write(0.0);
                }
            });
        }
        // showdown
        else {
            let pair_index = card_pair_to_index(node.turn, node.river);
            let indices = &self.bunching_num_river[player][pair_index];
            let player_strength = &self.bunching_strength[pair_index][player];
            let opponent_strength = &self.bunching_strength[pair_index][player ^ 1];

            result
                .iter_mut()
                .zip(indices)
                .zip(player_strength)
                .for_each(|((r, &index), &strength)| {
                    if index != 0 {
                        r.write(inner_product_cond(
                            cfreach,
                            &self.bunching_arena[index..index + opponent_len],
                            opponent_strength,
                            strength,
                            amount_win,
                            amount_lose,
                            amount_tie,
                        ));
                    } else {
                        r.write(0.0);
                    }
                });
        }
    }
}
