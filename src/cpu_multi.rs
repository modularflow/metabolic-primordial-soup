//! CPU-based multi-simulation backend
//!
//! This module provides a CPU implementation matching the CUDA backend's
//! `CudaMultiSimulation` API for feature parity.
//!
//! Key features:
//! - Multiple simulations in parallel (batched)
//! - Energy system with per-sim death_timer and reserve_duration
//! - Mega mode for cross-simulation interaction
//! - Tape-based energy model (@, $, !)
//! - Per-tape step limits
//! - Spontaneous generation

use crate::bff::{SINGLE_TAPE_SIZE, FULL_TAPE_SIZE};
use crate::energy::EnergyConfig;
use rand::prelude::*;
use rayon::prelude::*;

/// Generate simple pairs for simulation (adjacent programs)
fn generate_pairs(num_programs: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(num_programs / 2);
    for i in (0..num_programs).step_by(2) {
        if i + 1 < num_programs {
            pairs.push((i, i + 1));
        }
    }
    pairs
}

/// Hash function for per-sim source offsets (matches CUDA)
#[inline]
fn sim_hash(sim_idx: u32, src_idx: u32) -> u32 {
    let mut h = sim_idx.wrapping_mul(0x9E3779B9).wrapping_add(src_idx.wrapping_mul(0x85EBCA6B));
    h ^= h >> 16;
    h = h.wrapping_mul(0x21F0AAAD);
    h ^= h >> 15;
    h
}

#[inline]
fn source_offset_x(sim_idx: u32, src_idx: u32, base_x: u32, grid_width: u32) -> u32 {
    let h = sim_hash(sim_idx, src_idx * 2);
    let offset = (h % grid_width) as i32 - (grid_width / 2) as i32;
    let new_x = base_x as i32 + offset;
    ((new_x + grid_width as i32) % grid_width as i32) as u32
}

#[inline]
fn source_offset_y(sim_idx: u32, src_idx: u32, base_y: u32, grid_height: u32) -> u32 {
    let h = sim_hash(sim_idx, src_idx * 2 + 1);
    let offset = (h % grid_height) as i32 - (grid_height / 2) as i32;
    let new_y = base_y as i32 + offset;
    ((new_y + grid_height as i32) % grid_height as i32) as u32
}

/// Check if point is within a source (matches CUDA shapes)
#[inline]
fn in_source(x: i32, y: i32, sx: u32, sy: u32, shape: u32, radius: u32) -> bool {
    let dx = x as f32 - sx as f32;
    let dy = y as f32 - sy as f32;
    let r = radius as f32;
    let r_sq = r * r;
    let dist_sq = dx * dx + dy * dy;

    match shape {
        0 => dist_sq <= r_sq,                                    // Circle
        1 => dx.abs() <= r && dy.abs() <= r / 4.0,              // Strip Horizontal
        2 => dx.abs() <= r / 4.0 && dy.abs() <= r,              // Strip Vertical
        3 => dy <= 0.0 && dist_sq <= r_sq,                      // Half Circle Top
        4 => dy >= 0.0 && dist_sq <= r_sq,                      // Half Circle Bottom
        5 => dx <= 0.0 && dist_sq <= r_sq,                      // Half Circle Left
        6 => dx >= 0.0 && dist_sq <= r_sq,                      // Half Circle Right
        7 => (dx / r).powi(2) + (dy / (r / 2.0)).powi(2) <= 1.0, // Ellipse Horizontal
        8 => (dx / (r / 2.0)).powi(2) + (dy / r).powi(2) <= 1.0, // Ellipse Vertical
        _ => dist_sq <= r_sq,
    }
}

/// Compute energy zone membership bitmask (matches CUDA)
fn compute_energy_map(
    config: Option<&EnergyConfig>,
    num_programs: usize,
    num_sims: usize,
    grid_width: usize,
    grid_height: usize,
    border_thickness: usize,
) -> Vec<u32> {
    let total_programs = num_programs * num_sims;
    let num_words = (total_programs + 31) / 32;
    let mut map = vec![0u32; num_words];

    let config = match config {
        Some(cfg) if cfg.enabled && !cfg.sources.is_empty() => cfg,
        Some(cfg) if cfg.enabled && cfg.sources.is_empty() => {
            // Energy grid mode (death_only): no zones, all bits stay 0
            return map;
        }
        _ => {
            // Energy disabled: all programs are considered "in zone"
            for word in &mut map {
                *word = 0xFFFFFFFF;
            }
            return map;
        }
    };

    let sources: Vec<(u32, u32, u32, u32)> = config
        .sources
        .iter()
        .take(8)
        .map(|s| (s.x as u32, s.y as u32, s.shape.to_gpu_id(), s.radius as u32))
        .collect();

    for sim_idx in 0..num_sims {
        for prog_idx in 0..num_programs {
            let x = (prog_idx % grid_width) as i32;
            let y = (prog_idx / grid_width) as i32;

            // Check border (dead zone)
            let in_border = border_thickness > 0 && (
                x < border_thickness as i32 || x >= (grid_width - border_thickness) as i32 ||
                y < border_thickness as i32 || y >= (grid_height - border_thickness) as i32
            );

            if in_border {
                continue; // Not in any energy zone
            }

            // Check energy sources
            let mut in_zone = false;
            for (src_idx, (base_x, base_y, shape, radius)) in sources.iter().enumerate() {
                let offset_x = source_offset_x(sim_idx as u32, src_idx as u32, *base_x, grid_width as u32);
                let offset_y = source_offset_y(sim_idx as u32, src_idx as u32, *base_y, grid_height as u32);
                if in_source(x, y, offset_x, offset_y, *shape, *radius) {
                    in_zone = true;
                    break;
                }
            }

            if in_zone {
                let global_idx = sim_idx * num_programs + prog_idx;
                let word_idx = global_idx / 32;
                let bit_idx = global_idx % 32;
                map[word_idx] |= 1u32 << bit_idx;
            }
        }
    }

    map
}

/// Energy state helpers (matches CUDA packing: reserve(16) | timer(15) | dead(1))
#[inline]
fn get_reserve(state: u32) -> u32 { state & 0xFFFF }
#[inline]
fn get_timer(state: u32) -> u32 { (state >> 16) & 0x7FFF }
#[inline]
fn is_dead(state: u32) -> bool { (state >> 31) != 0 }
#[inline]
fn pack_state(reserve: u32, timer: u32, dead: bool) -> u32 {
    (reserve & 0xFFFF) | ((timer & 0x7FFF) << 16) | (if dead { 1u32 << 31 } else { 0 })
}

/// Check if a byte is a BFF operation (excluding @)
#[inline]
fn is_bff_op(c: u8) -> bool {
    matches!(c, b'<' | b'>' | b'{' | b'}' | b'+' | b'-' | b'.' | b',' | b'[' | b']' | b'!' | b'$')
}

/// LCG for fast mutations (matches CUDA)
#[inline]
fn lcg(s: u32) -> u32 {
    s.wrapping_mul(1664525).wrapping_add(1013904223)
}

/// CPU-based multi-simulation matching CudaMultiSimulation API
pub struct CpuMultiSimulation {
    /// All program tapes across all sims
    soup: Vec<u8>,
    /// Pair indices (local per sim, or absolute in mega mode)
    pairs: Vec<(u32, u32)>,
    /// Energy state per program: reserve(16) | timer(15) | dead(1)
    energy_state: Vec<u32>,
    /// Per-sim configs: (death_timer, reserve_duration)
    sim_configs: Vec<(u32, u32)>,
    /// Precomputed energy zone membership bitmask
    energy_map: Vec<u32>,
    /// Per-tape step limits
    tape_steps: Vec<u32>,
    // Configuration
    num_sims: usize,
    num_programs: usize,
    num_pairs: usize,
    grid_width: usize,
    grid_height: usize,
    steps_per_run: u32,
    mutation_prob: u32,
    seed: u64,
    epoch: u64,
    energy_enabled: bool,
    mega_mode: bool,
    spontaneous_rate: u32,
    border_thickness: usize,
    use_per_tape_steps: bool,
    zoneless_mode: bool,
}

impl CpuMultiSimulation {
    /// Create a new CPU multi-simulation
    pub fn new(
        num_sims: usize,
        num_programs: usize,
        grid_width: usize,
        grid_height: usize,
        seed: u64,
        mutation_prob: u32,
        steps_per_run: u32,
        energy_config: Option<&EnergyConfig>,
        per_sim_configs: Option<Vec<(u32, u32)>>,
        border_thickness: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let total_programs = num_sims * num_programs;
        let num_pairs = num_programs / 2;

        // Energy configuration
        let energy_enabled = energy_config.map(|c| c.enabled).unwrap_or(false);
        let zoneless_mode = energy_config.map(|c| c.enabled && c.sources.is_empty()).unwrap_or(false);
        let default_death = energy_config.map(|c| c.interaction_death).unwrap_or(10);
        let default_reserve = energy_config.map(|c| c.reserve_duration).unwrap_or(5);
        let spontaneous_rate = energy_config.map(|c| c.spontaneous_rate).unwrap_or(0);

        // Per-sim configs
        let sim_configs: Vec<(u32, u32)> = match per_sim_configs {
            Some(configs) if !configs.is_empty() => {
                (0..num_sims)
                    .map(|i| configs[i % configs.len()])
                    .collect()
            }
            _ => vec![(default_death, default_reserve); num_sims],
        };

        // Energy map
        let energy_map = compute_energy_map(
            energy_config,
            num_programs,
            num_sims,
            grid_width,
            grid_height,
            border_thickness,
        );

        // Initialize soup with random data, borders zeroed
        let mut rng = StdRng::seed_from_u64(seed);
        let mut soup: Vec<u8> = (0..total_programs * SINGLE_TAPE_SIZE)
            .map(|_| rng.random())
            .collect();

        // Zero out border regions
        if border_thickness > 0 {
            for sim_idx in 0..num_sims {
                let sim_offset = sim_idx * num_programs;
                for prog_idx in 0..num_programs {
                    let x = prog_idx % grid_width;
                    let y = prog_idx / grid_width;
                    let in_border = x < border_thickness || x >= grid_width - border_thickness ||
                                   y < border_thickness || y >= grid_height - border_thickness;
                    if in_border {
                        let byte_offset = (sim_offset + prog_idx) * SINGLE_TAPE_SIZE;
                        for i in 0..SINGLE_TAPE_SIZE {
                            soup[byte_offset + i] = 0;
                        }
                    }
                }
            }
        }

        // Energy states (all alive with no reserve initially)
        let energy_state = vec![0u32; total_programs];

        // Per-tape step limits
        let tape_steps = vec![steps_per_run; total_programs];

        // Default pairs
        let pairs: Vec<(u32, u32)> = generate_pairs(num_programs)
            .into_iter()
            .map(|(a, b)| (a as u32, b as u32))
            .collect();

        println!("CPU Multi-Simulation initialized:");
        println!("  Total programs: {} ({} sims × {} programs/sim)", total_programs, num_sims, num_programs);
        println!("  Memory: {:.2} MB", (total_programs * SINGLE_TAPE_SIZE) as f64 / 1e6);

        Ok(Self {
            soup,
            pairs,
            energy_state,
            sim_configs,
            energy_map,
            tape_steps,
            num_sims,
            num_programs,
            num_pairs,
            grid_width,
            grid_height,
            steps_per_run,
            mutation_prob,
            seed,
            epoch: 0,
            energy_enabled,
            mega_mode: false,
            spontaneous_rate,
            border_thickness,
            use_per_tape_steps: false,
            zoneless_mode,
        })
    }

    /// Run one epoch across all simulations
    pub fn step(&mut self) -> u64 {
        let total_pairs = if self.mega_mode {
            self.num_pairs
        } else {
            self.num_pairs * self.num_sims
        };

        // Build work items
        let work_items: Vec<_> = if self.mega_mode {
            (0..self.num_pairs)
                .map(|pair_idx| {
                    let (p1_abs, p2_abs) = self.pairs[pair_idx];
                    (pair_idx, p1_abs as usize, p2_abs as usize)
                })
                .collect()
        } else {
            let mut items = Vec::with_capacity(total_pairs);
            for sim_idx in 0..self.num_sims {
                let sim_offset = sim_idx * self.num_programs;
                for (pair_idx, &(p1_local, p2_local)) in self.pairs.iter().enumerate() {
                    items.push((
                        sim_idx * self.num_pairs + pair_idx,
                        sim_offset + p1_local as usize,
                        sim_offset + p2_local as usize,
                    ));
                }
            }
            items
        };

        // Process pairs in parallel and collect results
        let results: Vec<_> = work_items
            .par_iter()
            .map(|&(pair_idx, p1_abs, p2_abs)| {
                self.process_pair(pair_idx, p1_abs, p2_abs)
            })
            .collect();

        // Apply results sequentially
        let mut total_ops = 0u64;
        for result in results {
            total_ops += result.ops;
            
            // Update soup
            let p1_offset = result.p1_abs * SINGLE_TAPE_SIZE;
            let p2_offset = result.p2_abs * SINGLE_TAPE_SIZE;
            self.soup[p1_offset..p1_offset + SINGLE_TAPE_SIZE]
                .copy_from_slice(&result.tape[..SINGLE_TAPE_SIZE]);
            self.soup[p2_offset..p2_offset + SINGLE_TAPE_SIZE]
                .copy_from_slice(&result.tape[SINGLE_TAPE_SIZE..]);
            
            // Update energy states
            self.energy_state[result.p1_abs] = result.p1_energy;
            self.energy_state[result.p2_abs] = result.p2_energy;
        }

        self.epoch += 1;
        total_ops
    }

    /// Process a single pair (the core BFF evaluation with energy)
    fn process_pair(&self, pair_idx: usize, p1_abs: usize, p2_abs: usize) -> PairResult {
        let p1_sim = p1_abs / self.num_programs;
        let p2_sim = p2_abs / self.num_programs;
        let p1_local = p1_abs % self.num_programs;
        let p2_local = p2_abs % self.num_programs;

        // Get per-sim configs
        let (p1_death_timer, p1_reserve_duration) = self.sim_configs[p1_sim];
        let (p2_death_timer, p2_reserve_duration) = self.sim_configs[p2_sim];

        // Check energy zone membership
        let p1_in_zone = self.in_energy_zone(p1_abs);
        let p2_in_zone = self.in_energy_zone(p2_abs);
        let p1_in_border = self.in_border(p1_abs);
        let p2_in_border = self.in_border(p2_abs);

        // Load energy states
        let mut p1_state = self.energy_state[p1_abs];
        let mut p2_state = self.energy_state[p2_abs];
        let p1_was_dead = self.energy_enabled && is_dead(p1_state);
        let p2_was_dead = self.energy_enabled && is_dead(p2_state);

        // Create local tape
        let mut tape = [0u8; FULL_TAPE_SIZE];
        let p1_offset = p1_abs * SINGLE_TAPE_SIZE;
        let p2_offset = p2_abs * SINGLE_TAPE_SIZE;

        // ENERGY ZONE OVERWRITE: Tapes in energy zones become pure energy (@)
        if p1_in_zone {
            for i in 0..SINGLE_TAPE_SIZE {
                tape[i] = b'@';
            }
        } else {
            tape[..SINGLE_TAPE_SIZE].copy_from_slice(&self.soup[p1_offset..p1_offset + SINGLE_TAPE_SIZE]);
        }

        if p2_in_zone {
            for i in 0..SINGLE_TAPE_SIZE {
                tape[SINGLE_TAPE_SIZE + i] = b'@';
            }
        } else {
            tape[SINGLE_TAPE_SIZE..].copy_from_slice(&self.soup[p2_offset..p2_offset + SINGLE_TAPE_SIZE]);
        }

        // Can mutate check
        let p1_can_mutate = !self.energy_enabled || (!is_dead(p1_state) && (p1_in_zone || get_reserve(p1_state) > 0));
        let p2_can_mutate = !self.energy_enabled || (!is_dead(p2_state) && (p2_in_zone || get_reserve(p2_state) > 0));

        // Apply sparse mutations
        let mut rng = (self.seed as u32)
            ^ (self.epoch as u32)
            ^ (pair_idx as u32).wrapping_mul(0x9E3779B9)
            ^ (p1_sim as u32).wrapping_mul(0x85EBCA6B);

        if p1_can_mutate && self.mutation_prob > 0 {
            let inv_prob = (1u32 << 30) / self.mutation_prob.max(1);
            let mut byte_pos = 0usize;
            rng = lcg(rng);
            let skip = ((rng >> 8).wrapping_mul(inv_prob)) >> 22;
            byte_pos += skip as usize;

            while byte_pos < SINGLE_TAPE_SIZE {
                rng = lcg(rng);
                tape[byte_pos] = ((rng >> 8) & 0xFF) as u8;
                rng = lcg(rng);
                let skip = (((rng >> 8).wrapping_mul(inv_prob)) >> 22).max(1);
                byte_pos += skip as usize;
            }
        }

        if p2_can_mutate && self.mutation_prob > 0 {
            let inv_prob = (1u32 << 30) / self.mutation_prob.max(1);
            let mut byte_pos = SINGLE_TAPE_SIZE;
            rng = lcg(rng);
            let skip = ((rng >> 8).wrapping_mul(inv_prob)) >> 22;
            byte_pos += skip as usize;

            while byte_pos < FULL_TAPE_SIZE {
                rng = lcg(rng);
                tape[byte_pos] = ((rng >> 8) & 0xFF) as u8;
                rng = lcg(rng);
                let skip = (((rng >> 8).wrapping_mul(inv_prob)) >> 22).max(1);
                byte_pos += skip as usize;
            }
        }

        // Skip if both dead and not in energy zones
        if self.energy_enabled && p1_was_dead && p2_was_dead && !p1_in_zone && !p2_in_zone {
            return PairResult {
                p1_abs,
                p2_abs,
                tape,
                ops: 0,
                p1_energy: p1_state,
                p2_energy: p2_state,
            };
        }

        // Track copies
        let mut p1_received_copy = false;
        let mut p2_received_copy = false;

        // Get step budget
        let steps_per_run = if self.use_per_tape_steps {
            let p1_steps = self.tape_steps[p1_abs];
            let p2_steps = self.tape_steps[p2_abs];
            (p1_steps + p2_steps).min(self.steps_per_run * 2)
        } else {
            self.steps_per_run
        };

        // Count @ symbols for early termination
        let mut remaining_energy: u32 = tape.iter().filter(|&&c| c == b'@').count() as u32;
        let has_dollar = tape.iter().any(|&c| c == b'$');

        // BFF Evaluation with TAPE-BASED ENERGY MODEL
        let mut ops = 0u64;
        let mut pos: i32 = 2;
        let mut head0: i32 = (tape[0] & (FULL_TAPE_SIZE as u8 - 1)) as i32;
        let mut head1: i32 = (tape[1] & (FULL_TAPE_SIZE as u8 - 1)) as i32;

        // Skip execution if no energy and no way to get energy
        let tape_active = remaining_energy > 0 || has_dollar;

        if tape_active {
            for _step in 0..steps_per_run {
                if remaining_energy == 0 && !has_dollar {
                    break;
                }
                if pos < 0 || pos >= FULL_TAPE_SIZE as i32 {
                    break;
                }

                head0 &= (FULL_TAPE_SIZE as i32) - 1;
                head1 &= (FULL_TAPE_SIZE as i32) - 1;

                let cmd = tape[pos as usize];

                if cmd == b'@' {
                    // Consume the @
                    tape[pos as usize] = 0;
                    remaining_energy = remaining_energy.saturating_sub(1);

                    // Peek at next byte
                    let next_pos = pos + 1;
                    if next_pos < FULL_TAPE_SIZE as i32 {
                        let next_cmd = tape[next_pos as usize];
                        if is_bff_op(next_cmd) {
                            match next_cmd {
                                b'<' => { head0 -= 1; ops += 1; }
                                b'>' => { head0 += 1; ops += 1; }
                                b'{' => { head1 -= 1; ops += 1; }
                                b'}' => { head1 += 1; ops += 1; }
                                b'+' => {
                                    let idx = (head0 as usize) & (FULL_TAPE_SIZE - 1);
                                    tape[idx] = tape[idx].wrapping_add(1);
                                    ops += 1;
                                }
                                b'-' => {
                                    let idx = (head0 as usize) & (FULL_TAPE_SIZE - 1);
                                    tape[idx] = tape[idx].wrapping_sub(1);
                                    ops += 1;
                                }
                                b'.' => {
                                    let src = (head0 as usize) & (FULL_TAPE_SIZE - 1);
                                    let dst = (head1 as usize) & (FULL_TAPE_SIZE - 1);
                                    tape[dst] = tape[src];
                                    // Track cross-program copies
                                    if src < SINGLE_TAPE_SIZE && dst >= SINGLE_TAPE_SIZE {
                                        p2_received_copy = true;
                                    } else if src >= SINGLE_TAPE_SIZE && dst < SINGLE_TAPE_SIZE {
                                        p1_received_copy = true;
                                    }
                                    ops += 1;
                                }
                                b',' => {
                                    let src = (head1 as usize) & (FULL_TAPE_SIZE - 1);
                                    let dst = (head0 as usize) & (FULL_TAPE_SIZE - 1);
                                    tape[dst] = tape[src];
                                    if src < SINGLE_TAPE_SIZE && dst >= SINGLE_TAPE_SIZE {
                                        p2_received_copy = true;
                                    } else if src >= SINGLE_TAPE_SIZE && dst < SINGLE_TAPE_SIZE {
                                        p1_received_copy = true;
                                    }
                                    ops += 1;
                                }
                                b'[' => {
                                    let h0_idx = (head0 as usize) & (FULL_TAPE_SIZE - 1);
                                    if tape[h0_idx] == 0 {
                                        let mut depth = 1i32;
                                        let mut np = next_pos + 1;
                                        while np < FULL_TAPE_SIZE as i32 && depth > 0 {
                                            if tape[np as usize] == b']' { depth -= 1; }
                                            if tape[np as usize] == b'[' { depth += 1; }
                                            np += 1;
                                        }
                                        np -= 1;
                                        if depth != 0 { np = FULL_TAPE_SIZE as i32; }
                                        pos = np;
                                    } else {
                                        pos = next_pos;
                                    }
                                    ops += 1;
                                }
                                b']' => {
                                    let h0_idx = (head0 as usize) & (FULL_TAPE_SIZE - 1);
                                    if tape[h0_idx] != 0 {
                                        let mut depth = 1i32;
                                        let mut np = next_pos - 1;
                                        while np >= 0 && depth > 0 {
                                            if tape[np as usize] == b']' { depth += 1; }
                                            if tape[np as usize] == b'[' { depth -= 1; }
                                            np -= 1;
                                        }
                                        np += 1;
                                        if depth != 0 { np = -1; }
                                        pos = np;
                                    } else {
                                        pos = next_pos;
                                    }
                                    ops += 1;
                                }
                                b'!' => {
                                    ops += 1;
                                    break; // Halt
                                }
                                b'$' => {
                                    // Store-energy: only works if head points at @ on other half
                                    let h0 = (head0 as usize) & (FULL_TAPE_SIZE - 1);
                                    let h1 = (head1 as usize) & (FULL_TAPE_SIZE - 1);
                                    let current_half = if (next_pos as usize) < SINGLE_TAPE_SIZE { 0 } else { 1 };

                                    let can_harvest = if current_half == 0 {
                                        (h0 >= SINGLE_TAPE_SIZE && tape[h0] == b'@') ||
                                        (h1 >= SINGLE_TAPE_SIZE && tape[h1] == b'@')
                                    } else {
                                        (h0 < SINGLE_TAPE_SIZE && tape[h0] == b'@') ||
                                        (h1 < SINGLE_TAPE_SIZE && tape[h1] == b'@')
                                    };

                                    if can_harvest {
                                        tape[next_pos as usize] = b'@';
                                        // Consume the harvested @
                                        if current_half == 0 {
                                            if h0 >= SINGLE_TAPE_SIZE && tape[h0] == b'@' { tape[h0] = 0; }
                                            else if h1 >= SINGLE_TAPE_SIZE && tape[h1] == b'@' { tape[h1] = 0; }
                                        } else {
                                            if h0 < SINGLE_TAPE_SIZE && tape[h0] == b'@' { tape[h0] = 0; }
                                            else if h1 < SINGLE_TAPE_SIZE && tape[h1] == b'@' { tape[h1] = 0; }
                                        }
                                    }
                                    ops += 1;
                                    pos = next_pos;
                                }
                                _ => { pos = next_pos; }
                            }
                        }
                    }
                }
                // BFF ops without @ are NOPs
                pos += 1;
            }
        }

        // Update energy states
        let mut p1_stays_dead = false;
        let mut p2_stays_dead = false;

        if self.energy_enabled {
            // P1 energy update
            let mut p1_reserve = get_reserve(p1_state);
            let mut p1_timer = get_timer(p1_state);
            let mut p1_dead = p1_was_dead;

            if p1_in_zone {
                p1_reserve = p1_reserve_duration;
                p1_timer = 0;
            } else if p1_received_copy {
                p1_reserve = if p2_in_zone { p2_reserve_duration } else { get_reserve(p2_state) };
                p1_timer = 0;
                p1_dead = false;
            } else {
                if p1_reserve > 0 { p1_reserve -= 1; }
                if !p1_dead { p1_timer += 1; }
                if p1_death_timer > 0 && p1_timer > p1_death_timer && !p1_dead {
                    p1_dead = true;
                }
            }
            p1_stays_dead = p1_was_dead && p1_dead;
            p1_state = pack_state(p1_reserve, p1_timer, p1_dead);

            // P2 energy update
            let mut p2_reserve = get_reserve(p2_state);
            let mut p2_timer = get_timer(p2_state);
            let mut p2_dead = p2_was_dead;

            if p2_in_zone {
                p2_reserve = p2_reserve_duration;
                p2_timer = 0;
            } else if p2_received_copy {
                p2_reserve = if p1_in_zone { p1_reserve_duration } else { get_reserve(p1_state) };
                p2_timer = 0;
                p2_dead = false;
            } else {
                if p2_reserve > 0 { p2_reserve -= 1; }
                if !p2_dead { p2_timer += 1; }
                if p2_death_timer > 0 && p2_timer > p2_death_timer && !p2_dead {
                    p2_dead = true;
                }
            }
            p2_stays_dead = p2_was_dead && p2_dead;
            p2_state = pack_state(p2_reserve, p2_timer, p2_dead);
        }

        // Spontaneous generation
        if self.energy_enabled && self.spontaneous_rate > 0 {
            let p1_can_spawn = p1_stays_dead && (self.zoneless_mode || p1_in_zone);
            let p2_can_spawn = p2_stays_dead && (self.zoneless_mode || p2_in_zone);

            if p1_can_spawn {
                rng = lcg(rng);
                if rng % self.spontaneous_rate == 0 {
                    for i in 0..SINGLE_TAPE_SIZE {
                        rng = lcg(rng);
                        tape[i] = ((rng >> 8) & 0xFF) as u8;
                    }
                    p1_state = pack_state(p1_reserve_duration, 0, false);
                    p1_stays_dead = false;
                }
            }

            if p2_can_spawn {
                rng = lcg(rng);
                if rng % self.spontaneous_rate == 0 {
                    for i in 0..SINGLE_TAPE_SIZE {
                        rng = lcg(rng);
                        tape[SINGLE_TAPE_SIZE + i] = ((rng >> 8) & 0xFF) as u8;
                    }
                    p2_state = pack_state(p2_reserve_duration, 0, false);
                    p2_stays_dead = false;
                }
            }
        }

        // Handle final tape state
        // Border: always zero
        // In-zone: always @
        // Dead: zero
        // Alive: keep tape

        if p1_in_border {
            for i in 0..SINGLE_TAPE_SIZE { tape[i] = 0; }
        } else if p1_in_zone {
            for i in 0..SINGLE_TAPE_SIZE { tape[i] = b'@'; }
        } else if p1_stays_dead || (self.energy_enabled && is_dead(p1_state)) {
            for i in 0..SINGLE_TAPE_SIZE { tape[i] = 0; }
        }

        if p2_in_border {
            for i in 0..SINGLE_TAPE_SIZE { tape[SINGLE_TAPE_SIZE + i] = 0; }
        } else if p2_in_zone {
            for i in 0..SINGLE_TAPE_SIZE { tape[SINGLE_TAPE_SIZE + i] = b'@'; }
        } else if p2_stays_dead || (self.energy_enabled && is_dead(p2_state)) {
            for i in 0..SINGLE_TAPE_SIZE { tape[SINGLE_TAPE_SIZE + i] = 0; }
        }

        PairResult {
            p1_abs,
            p2_abs,
            tape,
            ops,
            p1_energy: p1_state,
            p2_energy: p2_state,
        }
    }

    /// Check if program is in energy zone
    #[inline]
    fn in_energy_zone(&self, prog_idx: usize) -> bool {
        let word_idx = prog_idx / 32;
        let bit_idx = prog_idx % 32;
        (self.energy_map[word_idx] & (1u32 << bit_idx)) != 0
    }

    /// Check if program is in border
    #[inline]
    fn in_border(&self, prog_idx: usize) -> bool {
        if self.border_thickness == 0 { return false; }
        let local_idx = prog_idx % self.num_programs;
        let x = local_idx % self.grid_width;
        let y = local_idx / self.grid_width;
        x < self.border_thickness || x >= self.grid_width - self.border_thickness ||
        y < self.border_thickness || y >= self.grid_height - self.border_thickness
    }

    // === Public API (matches CudaMultiSimulation) ===

    pub fn get_sim_soup(&self, sim_idx: usize) -> Vec<u8> {
        let offset = sim_idx * self.num_programs * SINGLE_TAPE_SIZE;
        let size = self.num_programs * SINGLE_TAPE_SIZE;
        self.soup[offset..offset + size].to_vec()
    }

    pub fn get_all_soup(&self) -> Vec<u8> {
        self.soup.clone()
    }

    pub fn get_all_energy_states(&self) -> Vec<u32> {
        self.energy_state.clone()
    }

    pub fn set_all_soup(&mut self, soup: &[u8]) {
        if soup.len() == self.soup.len() {
            self.soup.copy_from_slice(soup);
        }
    }

    pub fn set_all_energy_states(&mut self, states: &[u32]) {
        if states.len() == self.energy_state.len() {
            self.energy_state.copy_from_slice(states);
        }
    }

    pub fn set_pairs_all(&mut self, pairs: &[(u32, u32)]) {
        self.pairs = pairs.to_vec();
        self.num_pairs = pairs.len();
    }

    pub fn set_mega_mode(&mut self, enabled: bool) {
        self.mega_mode = enabled;
    }

    pub fn set_pairs_mega(&mut self, pairs: &[(u32, u32)]) {
        self.pairs = pairs.to_vec();
        self.num_pairs = pairs.len();
    }

    pub fn set_tape_steps(&mut self, sim_idx: usize, steps: &[u32]) {
        if steps.len() != self.num_programs { return; }
        let offset = sim_idx * self.num_programs;
        self.tape_steps[offset..offset + self.num_programs].copy_from_slice(steps);
    }

    pub fn set_all_tape_steps(&mut self, steps: &[u32]) {
        if steps.len() == self.tape_steps.len() {
            self.tape_steps.copy_from_slice(steps);
        }
    }

    pub fn set_use_per_tape_steps(&mut self, enabled: bool) {
        self.use_per_tape_steps = enabled;
    }

    pub fn use_per_tape_steps(&self) -> bool {
        self.use_per_tape_steps
    }

    pub fn update_energy_config(&mut self, config: &EnergyConfig) {
        self.energy_map = compute_energy_map(
            Some(config),
            self.num_programs,
            self.num_sims,
            self.grid_width,
            self.grid_height,
            self.border_thickness,
        );
        self.energy_enabled = config.enabled;
        self.zoneless_mode = config.enabled && config.sources.is_empty();
    }

    // Accessors
    pub fn num_sims(&self) -> usize { self.num_sims }
    pub fn num_programs(&self) -> usize { self.num_programs }
    pub fn grid_width(&self) -> usize { self.grid_width }
    pub fn grid_height(&self) -> usize { self.grid_height }
    pub fn epoch(&self) -> u64 { self.epoch }
    pub fn set_epoch(&mut self, epoch: u64) { self.epoch = epoch; }

    // Async readback stubs (CPU doesn't need async)
    pub fn begin_async_readback(&mut self) {}
    pub fn has_pending_readback(&self) -> bool { false }
    pub fn finish_async_readback(&mut self) -> Option<Vec<u8>> { None }
    pub fn get_all_soup_async(&mut self) -> Vec<u8> { self.get_all_soup() }
}

/// Result of processing a pair
struct PairResult {
    p1_abs: usize,
    p2_abs: usize,
    tape: [u8; FULL_TAPE_SIZE],
    ops: u64,
    p1_energy: u32,
    p2_energy: u32,
}
