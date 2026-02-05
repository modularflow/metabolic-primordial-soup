//! Energy Grid System
//!
//! A 2D energy field that simulates environmental energy distribution.
//! Replaces the binary "in zone / out of zone" model with a continuous
//! energy landscape that supports diffusion, convection, and dynamic sources.
//!
//! Key concepts:
//! - Energy values range from 0.0 to 1.0 per grid cell
//! - Energy sources emit energy into nearby cells
//! - Energy diffuses to neighboring cells over time
//! - Energy naturally decays each epoch
//! - Tapes acquire energy from the grid based on their position
//!
//! # Usage
//!
//! ```ignore
//! use energy_grid::{EnergyGridSystem, EnergyGridConfig};
//!
//! // Create from config
//! let config = EnergyGridConfig { enabled: true, ..Default::default() };
//! let mut system = EnergyGridSystem::new(config, grid_width, grid_height);
//!
//! // Each epoch:
//! system.update();  // Updates grid and tape energies
//!
//! // Get energy for a specific tape:
//! let steps = system.steps_per_run(tape_idx);
//! ```

use serde::{Deserialize, Serialize};

/// Type of energy source behavior
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EnergySourceType {
    /// Constant output (thermal vent)
    Constant,
    /// Oscillating output (geyser)
    Oscillating {
        /// Period in epochs
        period: usize,
        /// Phase offset (0.0 to 1.0)
        phase: f32,
    },
    /// Decaying output (lightning strike)
    Decaying {
        /// Half-life in epochs
        half_life: usize,
    },
    /// Moving source (convection cell)
    Moving {
        /// Velocity in grid cells per epoch
        velocity: (f32, f32),
    },
}

impl Default for EnergySourceType {
    fn default() -> Self {
        EnergySourceType::Constant
    }
}

/// An energy source in the grid
#[derive(Clone, Debug, PartialEq)]
pub struct GridEnergySource {
    /// X position (can be fractional for moving sources)
    pub x: f32,
    /// Y position (can be fractional for moving sources)
    pub y: f32,
    /// Intensity (0.0 to 1.0)
    pub intensity: f32,
    /// Radius of effect
    pub radius: f32,
    /// Source behavior type
    pub source_type: EnergySourceType,
    /// Age in epochs (for lifetime tracking and oscillation)
    pub age: usize,
    /// Initial intensity (for decaying sources)
    initial_intensity: f32,
}

impl GridEnergySource {
    /// Create a new constant energy source
    pub fn new(x: f32, y: f32, intensity: f32, radius: f32) -> Self {
        Self {
            x,
            y,
            intensity,
            radius,
            source_type: EnergySourceType::Constant,
            age: 0,
            initial_intensity: intensity,
        }
    }

    /// Create a new energy source with a specific type
    pub fn with_type(
        x: f32,
        y: f32,
        intensity: f32,
        radius: f32,
        source_type: EnergySourceType,
    ) -> Self {
        Self {
            x,
            y,
            intensity,
            radius,
            source_type,
            age: 0,
            initial_intensity: intensity,
        }
    }

    /// Get the current effective intensity based on source type and age
    pub fn effective_intensity(&self) -> f32 {
        match &self.source_type {
            EnergySourceType::Constant => self.intensity,
            EnergySourceType::Oscillating { period, phase } => {
                let t = (self.age as f32 / *period as f32) + phase;
                let oscillation = (t * 2.0 * std::f32::consts::PI).sin();
                // Map [-1, 1] to [0, intensity]
                self.intensity * (0.5 + 0.5 * oscillation)
            }
            EnergySourceType::Decaying { half_life } => {
                let decay = 0.5_f32.powf(self.age as f32 / *half_life as f32);
                self.initial_intensity * decay
            }
            EnergySourceType::Moving { .. } => self.intensity,
        }
    }

    /// Update the source for one epoch, returns true if source should be removed
    pub fn tick(&mut self, lifetime: Option<usize>) -> bool {
        self.age += 1;

        // Update position for moving sources
        if let EnergySourceType::Moving { velocity } = &self.source_type {
            self.x += velocity.0;
            self.y += velocity.1;
        }

        // Check lifetime expiration
        if let Some(max_age) = lifetime {
            if max_age > 0 && self.age >= max_age {
                return true;
            }
        }

        // Check if decaying source has effectively died
        if let EnergySourceType::Decaying { half_life } = &self.source_type {
            let effective = self.effective_intensity();
            // Remove if less than 1% of initial
            if effective < self.initial_intensity * 0.01 || self.age > half_life * 10 {
                return true;
            }
        }

        false
    }

    /// Calculate contribution to a grid cell at (gx, gy)
    pub fn contribution_at(&self, gx: usize, gy: usize) -> f32 {
        let dx = gx as f32 - self.x;
        let dy = gy as f32 - self.y;
        let dist_sq = dx * dx + dy * dy;
        let radius_sq = self.radius * self.radius;

        if dist_sq > radius_sq {
            return 0.0;
        }

        // Smooth falloff using squared distance ratio
        let falloff = 1.0 - (dist_sq / radius_sq);
        self.effective_intensity() * falloff * falloff
    }
}

/// 2D energy field underlying the tape grid
#[derive(Clone, Debug)]
pub struct EnergyGrid {
    /// Energy values at each grid cell (0.0 to 1.0)
    pub values: Vec<f32>,
    /// Grid dimensions
    pub width: usize,
    pub height: usize,
    /// Energy sources (heat vents, etc.)
    pub sources: Vec<GridEnergySource>,
    /// Diffusion rate (0.0 to 1.0) - how fast energy spreads
    pub diffusion_rate: f32,
    /// Convection velocity (global flow direction)
    pub convection_velocity: (f32, f32),
    /// Decay rate (0.0 to 1.0) - how fast energy dissipates
    pub decay_rate: f32,
    /// Scratch buffer for diffusion calculations
    scratch: Vec<f32>,
}

impl EnergyGrid {
    /// Create a new energy grid with given dimensions
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            values: vec![0.0; size],
            width,
            height,
            sources: Vec::new(),
            diffusion_rate: 0.1,
            convection_velocity: (0.0, 0.0),
            decay_rate: 0.01,
            scratch: vec![0.0; size],
        }
    }

    /// Create with specific parameters
    pub fn with_params(
        width: usize,
        height: usize,
        diffusion_rate: f32,
        convection_velocity: (f32, f32),
        decay_rate: f32,
    ) -> Self {
        let mut grid = Self::new(width, height);
        grid.diffusion_rate = diffusion_rate;
        grid.convection_velocity = convection_velocity;
        grid.decay_rate = decay_rate;
        grid
    }

    /// Add an energy source
    pub fn add_source(&mut self, source: GridEnergySource) {
        self.sources.push(source);
    }

    /// Add a constant source at position with given intensity and radius
    pub fn add_constant_source(&mut self, x: f32, y: f32, intensity: f32, radius: f32) {
        self.sources.push(GridEnergySource::new(x, y, intensity, radius));
    }

    /// Get energy at a specific grid position
    #[inline]
    pub fn energy_at(&self, x: usize, y: usize) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.values[y * self.width + x]
    }

    /// Set energy at a specific grid position
    #[inline]
    pub fn set_energy(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.values[y * self.width + x] = value.clamp(0.0, 1.0);
        }
    }

    /// Get the index for a grid position
    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Emit energy from all sources into the grid
    fn emit_from_sources(&mut self) {
        for source in &self.sources {
            // Calculate bounding box for source
            let min_x = ((source.x - source.radius).floor() as isize).max(0) as usize;
            let max_x = ((source.x + source.radius).ceil() as usize).min(self.width - 1);
            let min_y = ((source.y - source.radius).floor() as isize).max(0) as usize;
            let max_y = ((source.y + source.radius).ceil() as usize).min(self.height - 1);

            for gy in min_y..=max_y {
                for gx in min_x..=max_x {
                    let contribution = source.contribution_at(gx, gy);
                    if contribution > 0.0 {
                        let idx = self.idx(gx, gy);
                        self.values[idx] = (self.values[idx] + contribution).min(1.0);
                    }
                }
            }
        }
    }

    /// Diffuse energy to neighboring cells (simple 4-neighbor diffusion)
    fn diffuse(&mut self) {
        if self.diffusion_rate <= 0.0 {
            return;
        }

        // Copy current values to scratch buffer
        self.scratch.copy_from_slice(&self.values);

        let rate = self.diffusion_rate;

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = self.idx(x, y);
                let current = self.scratch[idx];

                // Get neighbor values (with boundary handling)
                let left = if x > 0 { self.scratch[idx - 1] } else { current };
                let right = if x < self.width - 1 { self.scratch[idx + 1] } else { current };
                let up = if y > 0 { self.scratch[idx - self.width] } else { current };
                let down = if y < self.height - 1 { self.scratch[idx + self.width] } else { current };

                // Laplacian diffusion
                let laplacian = (left + right + up + down) - 4.0 * current;
                self.values[idx] = (current + rate * laplacian).clamp(0.0, 1.0);
            }
        }
    }

    /// Apply convection (advection of energy by flow)
    fn convect(&mut self) {
        if self.convection_velocity == (0.0, 0.0) {
            return;
        }

        let (vx, vy) = self.convection_velocity;
        let width = self.width;
        self.scratch.copy_from_slice(&self.values);

        for y in 0..self.height {
            for x in 0..width {
                // Semi-Lagrangian advection: trace back to source
                let src_x = x as f32 - vx;
                let src_y = y as f32 - vy;

                // Bilinear interpolation from source position
                let value = self.sample_bilinear(&self.scratch, src_x, src_y);
                let idx = y * width + x;
                self.values[idx] = value;
            }
        }
    }

    /// Sample the grid with bilinear interpolation
    fn sample_bilinear(&self, buffer: &[f32], x: f32, y: f32) -> f32 {
        // Clamp to grid bounds
        let x = x.clamp(0.0, (self.width - 1) as f32);
        let y = y.clamp(0.0, (self.height - 1) as f32);

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let v00 = buffer[y0 * self.width + x0];
        let v10 = buffer[y0 * self.width + x1];
        let v01 = buffer[y1 * self.width + x0];
        let v11 = buffer[y1 * self.width + x1];

        let v0 = v00 * (1.0 - fx) + v10 * fx;
        let v1 = v01 * (1.0 - fx) + v11 * fx;

        v0 * (1.0 - fy) + v1 * fy
    }

    /// Decay energy across the grid
    fn decay(&mut self) {
        if self.decay_rate <= 0.0 {
            return;
        }

        let factor = 1.0 - self.decay_rate;
        for v in &mut self.values {
            *v *= factor;
        }
    }

    /// Update the energy grid for one epoch
    pub fn update(&mut self) {
        // 1. Emit from sources
        self.emit_from_sources();

        // 2. Diffuse energy
        self.diffuse();

        // 3. Apply convection (if enabled)
        self.convect();

        // 4. Decay
        self.decay();
    }

    /// Update sources (aging, lifetime expiration)
    /// Returns true if any sources were removed
    pub fn update_sources(&mut self, source_lifetime: Option<usize>) -> bool {
        let before = self.sources.len();
        self.sources.retain_mut(|s| !s.tick(source_lifetime));
        self.sources.len() != before
    }

    /// Get total energy in the grid
    pub fn total_energy(&self) -> f64 {
        self.values.iter().map(|&v| v as f64).sum()
    }

    /// Get average energy in the grid
    pub fn average_energy(&self) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.total_energy() as f32 / self.values.len() as f32
    }

    /// Get max energy in the grid
    pub fn max_energy(&self) -> f32 {
        self.values.iter().cloned().fold(0.0, f32::max)
    }

    /// Clear all energy
    pub fn clear(&mut self) {
        self.values.fill(0.0);
    }

    /// Clear all sources
    pub fn clear_sources(&mut self) {
        self.sources.clear();
    }

    /// Get a reference to the raw values (for rendering)
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Get number of active sources
    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }

    /// Render the energy grid as an RGB image
    /// Returns (width, height, RGB data)
    pub fn render_image(&self) -> (usize, usize, Vec<u8>) {
        let mut pixels = vec![0u8; self.width * self.height * 3];

        for y in 0..self.height {
            for x in 0..self.width {
                let energy = self.energy_at(x, y);
                let (r, g, b) = energy_to_color(energy);

                let idx = (y * self.width + x) * 3;
                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
            }
        }

        (self.width, self.height, pixels)
    }

    /// Render scaled energy grid (upscale to match tile-based tape rendering)
    /// Each grid cell becomes a scale x scale block of pixels
    pub fn render_scaled(&self, scale: usize) -> (usize, usize, Vec<u8>) {
        let img_w = self.width * scale;
        let img_h = self.height * scale;
        let mut pixels = vec![0u8; img_w * img_h * 3];

        for y in 0..self.height {
            for x in 0..self.width {
                let energy = self.energy_at(x, y);
                let (r, g, b) = energy_to_color(energy);

                // Fill scale x scale block
                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = x * scale + dx;
                        let py = y * scale + dy;
                        let idx = (py * img_w + px) * 3;
                        pixels[idx] = r;
                        pixels[idx + 1] = g;
                        pixels[idx + 2] = b;
                    }
                }
            }
        }

        (img_w, img_h, pixels)
    }

    /// Render as overlay (blend with existing image)
    /// The energy grid values are used as intensity for a color overlay
    pub fn render_overlay(&self, base_image: &mut [u8], img_width: usize, scale: usize, alpha: f32) {
        let alpha = alpha.clamp(0.0, 1.0);
        let inv_alpha = 1.0 - alpha;

        for y in 0..self.height {
            for x in 0..self.width {
                let energy = self.energy_at(x, y);
                let (er, eg, eb) = energy_to_color(energy);

                // Blend into each pixel of the scaled block
                for dy in 0..scale {
                    for dx in 0..scale {
                        let px = x * scale + dx;
                        let py = y * scale + dy;
                        let idx = (py * img_width + px) * 3;

                        if idx + 2 < base_image.len() {
                            let br = base_image[idx] as f32;
                            let bg = base_image[idx + 1] as f32;
                            let bb = base_image[idx + 2] as f32;

                            base_image[idx] = (br * inv_alpha + er as f32 * alpha) as u8;
                            base_image[idx + 1] = (bg * inv_alpha + eg as f32 * alpha) as u8;
                            base_image[idx + 2] = (bb * inv_alpha + eb as f32 * alpha) as u8;
                        }
                    }
                }
            }
        }
    }
}

/// Convert energy value (0.0-1.0) to RGB color
/// Uses a heat-map style gradient: black -> blue -> cyan -> green -> yellow -> red -> white
pub fn energy_to_color(energy: f32) -> (u8, u8, u8) {
    let energy = energy.clamp(0.0, 1.0);

    // 6-stop gradient
    if energy < 0.0001 {
        // Nearly zero - dark
        (10, 10, 15)
    } else if energy < 0.2 {
        // Black to blue
        let t = energy / 0.2;
        (10, 10, (15.0 + 200.0 * t) as u8)
    } else if energy < 0.4 {
        // Blue to cyan
        let t = (energy - 0.2) / 0.2;
        (10, (200.0 * t) as u8, 215)
    } else if energy < 0.6 {
        // Cyan to green
        let t = (energy - 0.4) / 0.2;
        ((10.0 + 100.0 * t) as u8, 200, (215.0 - 115.0 * t) as u8)
    } else if energy < 0.8 {
        // Green to yellow
        let t = (energy - 0.6) / 0.2;
        ((110.0 + 145.0 * t) as u8, 200, (100.0 - 100.0 * t) as u8)
    } else {
        // Yellow to red/white
        let t = (energy - 0.8) / 0.2;
        (255, (200.0 - 100.0 * t + 55.0 * t) as u8, (55.0 * t) as u8)
    }
}

/// Energy state for a single tape (per-tape energy reserve)
#[derive(Clone, Copy, Debug, Default)]
pub struct TapeEnergy {
    /// Current energy reserve (in "steps")
    pub reserve: u32,
    /// Maximum energy this tape can hold
    pub max_reserve: u32,
    /// Energy gained this epoch from environment
    pub gained_this_epoch: u32,
    /// Energy consumed this epoch during execution
    pub consumed_this_epoch: u32,
}

impl TapeEnergy {
    /// Default maximum steps per tape
    pub const DEFAULT_MAX_STEPS: u32 = 8192;

    /// Create a new tape energy state
    pub fn new() -> Self {
        Self {
            reserve: 0,
            max_reserve: Self::DEFAULT_MAX_STEPS,
            gained_this_epoch: 0,
            consumed_this_epoch: 0,
        }
    }

    /// Create with a specific max reserve
    pub fn with_max(max_reserve: u32) -> Self {
        Self {
            reserve: 0,
            max_reserve,
            gained_this_epoch: 0,
            consumed_this_epoch: 0,
        }
    }

    /// Calculate steps_per_run based on current reserve
    pub fn steps_per_run(&self) -> u32 {
        self.reserve.min(self.max_reserve)
    }

    /// Acquire energy from the environment
    pub fn acquire(&mut self, env_energy: f32, acquisition_rate: f32) {
        let gained = (env_energy * acquisition_rate) as u32;
        self.gained_this_epoch = gained;
        self.reserve = (self.reserve + gained).min(self.max_reserve);
    }

    /// Consume energy (called after execution)
    pub fn consume(&mut self, amount: u32) {
        self.consumed_this_epoch = amount;
        self.reserve = self.reserve.saturating_sub(amount);
    }

    /// Reset epoch counters
    pub fn reset_epoch(&mut self) {
        self.gained_this_epoch = 0;
        self.consumed_this_epoch = 0;
    }

    /// Check if tape has any energy
    pub fn has_energy(&self) -> bool {
        self.reserve > 0
    }
}

/// The complete energy grid system managing both the grid and per-tape energies
#[derive(Clone, Debug)]
pub struct EnergyGridSystem {
    /// The 2D energy grid
    pub grid: EnergyGrid,
    /// Per-tape energy reserves
    pub tape_energies: Vec<TapeEnergy>,
    /// Grid dimensions (tape grid, not energy grid)
    pub tape_grid_width: usize,
    pub tape_grid_height: usize,
    /// Configuration
    pub config: EnergyGridConfig,
    /// Current epoch
    pub epoch: usize,
}

impl EnergyGridSystem {
    /// Create a new energy grid system from configuration
    /// 
    /// For multi-simulation mode, set `num_sims` > 1. Each simulation gets
    /// its own set of tape energies, but they share the same energy grid.
    pub fn new(config: EnergyGridConfig, tape_grid_width: usize, tape_grid_height: usize) -> Self {
        Self::with_sims(config, tape_grid_width, tape_grid_height, 1)
    }
    
    /// Create a new energy grid system for multiple parallel simulations
    pub fn with_sims(config: EnergyGridConfig, tape_grid_width: usize, tape_grid_height: usize, num_sims: usize) -> Self {
        // Energy grid dimensions (may differ from tape grid based on resolution)
        let energy_width = tape_grid_width * config.resolution;
        let energy_height = tape_grid_height * config.resolution;

        let mut grid = EnergyGrid::with_params(
            energy_width,
            energy_height,
            config.diffusion_rate,
            if config.convection_enabled {
                (config.convection_x, config.convection_y)
            } else {
                (0.0, 0.0)
            },
            config.decay_rate,
        );

        // Add configured sources
        for source_config in &config.sources {
            grid.add_source(source_config.to_source());
        }

        // Initialize per-tape energies (one per tape per simulation)
        let num_tapes = tape_grid_width * tape_grid_height * num_sims;
        let tape_energies: Vec<TapeEnergy> = (0..num_tapes)
            .map(|_| TapeEnergy::with_max(config.max_reserve))
            .collect();

        Self {
            grid,
            tape_energies,
            tape_grid_width,
            tape_grid_height,
            config,
            epoch: 0,
        }
    }

    /// Create a disabled energy grid system (no effect on simulation)
    pub fn disabled(tape_grid_width: usize, tape_grid_height: usize) -> Self {
        Self::new(EnergyGridConfig::default(), tape_grid_width, tape_grid_height)
    }

    /// Check if the system is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Update the system for one epoch
    pub fn update(&mut self) {
        if !self.config.enabled {
            return;
        }

        // Update the energy grid (emission, diffusion, decay)
        self.grid.update();

        // Acquire energy for each tape based on its position
        self.acquire_energy_for_all_tapes();

        self.epoch += 1;
    }

    /// Acquire energy for all tapes from their grid positions
    fn acquire_energy_for_all_tapes(&mut self) {
        let resolution = self.config.resolution;
        let acquisition_rate = self.config.acquisition_rate;

        for tape_y in 0..self.tape_grid_height {
            for tape_x in 0..self.tape_grid_width {
                let tape_idx = tape_y * self.tape_grid_width + tape_x;

                // Map tape position to energy grid position
                // For resolution > 1, we sample the center of the corresponding region
                let energy_x = tape_x * resolution + resolution / 2;
                let energy_y = tape_y * resolution + resolution / 2;

                let energy = self.grid.energy_at(energy_x, energy_y);
                self.tape_energies[tape_idx].reset_epoch();
                self.tape_energies[tape_idx].acquire(energy, acquisition_rate);
            }
        }
    }

    /// Get the energy for a specific tape
    pub fn tape_energy(&self, tape_idx: usize) -> &TapeEnergy {
        &self.tape_energies[tape_idx]
    }

    /// Get steps_per_run for a specific tape
    pub fn steps_per_run(&self, tape_idx: usize) -> u32 {
        if !self.config.enabled {
            return self.config.max_reserve; // Default when disabled
        }
        self.tape_energies[tape_idx].steps_per_run()
    }

    /// Record energy consumption for a tape after execution
    pub fn consume_energy(&mut self, tape_idx: usize, amount: u32) {
        self.tape_energies[tape_idx].consume(amount);
    }

    /// Get the energy at a tape position (for visualization/debugging)
    pub fn energy_at_tape(&self, tape_x: usize, tape_y: usize) -> f32 {
        let resolution = self.config.resolution;
        let energy_x = tape_x * resolution + resolution / 2;
        let energy_y = tape_y * resolution + resolution / 2;
        self.grid.energy_at(energy_x, energy_y)
    }

    /// Render the energy grid scaled to match tape visualization (8x8 pixels per tape)
    pub fn render_tape_scale(&self) -> (usize, usize, Vec<u8>) {
        // Each tape is 8x8 pixels in standard visualization
        let tile_size = 8;
        let scale = tile_size / self.config.resolution.max(1);

        if scale > 1 {
            self.grid.render_scaled(scale)
        } else {
            self.grid.render_image()
        }
    }

    /// Create an overlay for the tape visualization
    pub fn apply_overlay(&self, base_image: &mut [u8], img_width: usize, alpha: f32) {
        let tile_size = 8;
        let scale = tile_size / self.config.resolution.max(1);
        self.grid.render_overlay(base_image, img_width, scale.max(1), alpha);
    }

    /// Get total energy in the system
    pub fn total_grid_energy(&self) -> f64 {
        self.grid.total_energy()
    }

    /// Get total energy in tape reserves
    pub fn total_tape_energy(&self) -> u64 {
        self.tape_energies.iter().map(|t| t.reserve as u64).sum()
    }

    /// Get metrics about the energy system
    pub fn metrics(&self) -> EnergyGridMetrics {
        EnergyGridMetrics {
            grid_total_energy: self.grid.total_energy(),
            tape_total_energy: self.total_tape_energy(),
            grid_average_energy: self.grid.average_energy(),
            grid_max_energy: self.grid.max_energy(),
            num_sources: self.grid.num_sources(),
            epoch: self.epoch,
            // These are populated by the simulation loop, not the system itself
            energy_acquired: 0,
            energy_consumed: 0,
            halt_count: 0,
            store_count: 0,
            skip_count: 0,
            cross_tape_energy_flow: 0,
        }
    }
}

/// Metrics from the energy grid system
#[derive(Clone, Debug, Default)]
pub struct EnergyGridMetrics {
    /// Total energy in the grid
    pub grid_total_energy: f64,
    /// Total energy in tape reserves
    pub tape_total_energy: u64,
    /// Average energy per grid cell
    pub grid_average_energy: f32,
    /// Maximum energy in any grid cell
    pub grid_max_energy: f32,
    /// Number of active energy sources
    pub num_sources: usize,
    /// Current epoch
    pub epoch: usize,
    /// Energy acquired this epoch (from grid to tapes)
    pub energy_acquired: u64,
    /// Energy consumed this epoch (during execution)
    pub energy_consumed: u64,
    /// Halt operations executed this epoch
    pub halt_count: u64,
    /// Store operations ($) executed this epoch
    pub store_count: u64,
    /// Skip operations (@) executed this epoch
    pub skip_count: u64,
    /// Cross-tape energy transfers (execution in partner's region)
    pub cross_tape_energy_flow: u64,
}

/// Accumulated statistics for cross-tape energy tracking
#[derive(Clone, Debug, Default)]
pub struct CrossTapeEnergyStats {
    /// Total energy consumed from tape 0 during paired execution
    pub total_consumed_tape0: u64,
    /// Total energy consumed from tape 1 during paired execution
    pub total_consumed_tape1: u64,
    /// Number of executions where tape 0 used tape 1's energy
    pub tape0_used_tape1_energy: u64,
    /// Number of executions where tape 1 used tape 0's energy
    pub tape1_used_tape0_energy: u64,
    /// Total halts executed
    pub total_halts: u64,
    /// Total store operations
    pub total_stores: u64,
    /// Total skip operations
    pub total_skips: u64,
}

impl CrossTapeEnergyStats {
    /// Update stats from a paired evaluation result
    pub fn update(&mut self, tape0_consumed: u32, tape1_consumed: u32, 
                  halted: bool, stores: u32, skips: u32,
                  steps_in_tape0: u32, steps_in_tape1: u32) {
        self.total_consumed_tape0 += tape0_consumed as u64;
        self.total_consumed_tape1 += tape1_consumed as u64;
        
        // Track cross-region execution (execution starts in one tape's region
        // but program originated from the other tape)
        if steps_in_tape1 > 0 {
            self.tape0_used_tape1_energy += 1;
        }
        if steps_in_tape0 > 0 {
            self.tape1_used_tape0_energy += 1;
        }
        
        if halted {
            self.total_halts += 1;
        }
        self.total_stores += stores as u64;
        self.total_skips += skips as u64;
    }
    
    /// Reset statistics for a new epoch
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Configuration for the energy grid system
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EnergyGridConfig {
    /// Enable the energy grid system
    pub enabled: bool,
    /// Grid resolution relative to tape grid (1 = same, 2 = 2x2 per tape, etc.)
    pub resolution: usize,
    /// Diffusion rate (0.0 to 1.0)
    pub diffusion_rate: f32,
    /// Enable convection
    pub convection_enabled: bool,
    /// Convection velocity X
    pub convection_x: f32,
    /// Convection velocity Y
    pub convection_y: f32,
    /// Energy decay per epoch (0.0 to 1.0)
    pub decay_rate: f32,
    /// Energy acquisition rate (steps per unit energy)
    pub acquisition_rate: f32,
    /// Initial energy sources
    pub sources: Vec<EnergyGridSourceConfig>,
    /// Maximum tape energy reserve
    pub max_reserve: u32,
    /// Base cost per execution step
    pub base_cost_per_step: u32,
    /// Epochs without receiving a copy until death (0 = infinite/never dies)
    /// When a program dies, its tape is zeroed out.
    #[serde(default)]
    pub death_epochs: u32,
    /// Energy reserve epochs after receiving a copy (grace period)
    /// Programs with reserve won't increment their death timer.
    #[serde(default = "default_reserve_epochs")]
    pub reserve_epochs: u32,
    /// Spontaneous generation rate (1 in N chance for dead tape to respawn, 0 = disabled)
    #[serde(default)]
    pub spontaneous_rate: u32,
}

fn default_reserve_epochs() -> u32 {
    10
}

impl Default for EnergyGridConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            resolution: 1,
            diffusion_rate: 0.1,
            convection_enabled: false,
            convection_x: 0.0,
            convection_y: 0.0,
            decay_rate: 0.01,
            acquisition_rate: 1000.0,
            sources: Vec::new(),
            max_reserve: TapeEnergy::DEFAULT_MAX_STEPS,
            base_cost_per_step: 1,
            death_epochs: 0,        // 0 = infinite (never dies)
            reserve_epochs: 10,     // Grace period after receiving copy
            spontaneous_rate: 0,    // Disabled by default
        }
    }
}

/// Configuration for a single energy source
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnergyGridSourceConfig {
    /// Source type: "constant", "oscillating", "decaying", "moving"
    #[serde(rename = "type")]
    pub source_type: String,
    /// X position
    pub x: f32,
    /// Y position
    pub y: f32,
    /// Intensity (0.0 to 1.0)
    pub intensity: f32,
    /// Radius of effect
    pub radius: f32,
    /// Period for oscillating sources
    #[serde(default)]
    pub period: Option<usize>,
    /// Phase for oscillating sources (0.0 to 1.0)
    #[serde(default)]
    pub phase: Option<f32>,
    /// Half-life for decaying sources
    #[serde(default)]
    pub half_life: Option<usize>,
    /// Velocity for moving sources
    #[serde(default)]
    pub velocity_x: Option<f32>,
    #[serde(default)]
    pub velocity_y: Option<f32>,
}

impl EnergyGridSourceConfig {
    /// Convert to a GridEnergySource
    pub fn to_source(&self) -> GridEnergySource {
        let source_type = match self.source_type.to_lowercase().as_str() {
            "oscillating" => EnergySourceType::Oscillating {
                period: self.period.unwrap_or(100),
                phase: self.phase.unwrap_or(0.0),
            },
            "decaying" => EnergySourceType::Decaying {
                half_life: self.half_life.unwrap_or(1000),
            },
            "moving" => EnergySourceType::Moving {
                velocity: (
                    self.velocity_x.unwrap_or(0.0),
                    self.velocity_y.unwrap_or(0.0),
                ),
            },
            _ => EnergySourceType::Constant,
        };

        GridEnergySource::with_type(self.x, self.y, self.intensity, self.radius, source_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_grid_creation() {
        let grid = EnergyGrid::new(100, 100);
        assert_eq!(grid.width, 100);
        assert_eq!(grid.height, 100);
        assert_eq!(grid.values.len(), 10000);
        assert_eq!(grid.total_energy(), 0.0);
    }

    #[test]
    fn test_source_emission() {
        let mut grid = EnergyGrid::new(100, 100);
        grid.add_constant_source(50.0, 50.0, 1.0, 10.0);

        // Before update, grid should be empty
        assert_eq!(grid.energy_at(50, 50), 0.0);

        // After update, center should have energy
        grid.update();
        assert!(grid.energy_at(50, 50) > 0.5);

        // Edge of radius should have less energy
        assert!(grid.energy_at(55, 50) < grid.energy_at(50, 50));

        // Outside radius should have minimal energy (from diffusion)
        assert!(grid.energy_at(70, 50) < 0.1);
    }

    #[test]
    fn test_diffusion() {
        let mut grid = EnergyGrid::with_params(10, 10, 0.2, (0.0, 0.0), 0.0);

        // Set center cell to max energy
        grid.set_energy(5, 5, 1.0);

        // Run a few diffusion steps
        for _ in 0..5 {
            grid.diffuse();
        }

        // Center should have decreased
        assert!(grid.energy_at(5, 5) < 1.0);

        // Neighbors should have some energy now
        assert!(grid.energy_at(4, 5) > 0.0);
        assert!(grid.energy_at(6, 5) > 0.0);
    }

    #[test]
    fn test_decay() {
        let mut grid = EnergyGrid::with_params(10, 10, 0.0, (0.0, 0.0), 0.1);

        // Set all cells to max
        for v in &mut grid.values {
            *v = 1.0;
        }

        // After decay, values should be 0.9
        grid.decay();
        assert!((grid.energy_at(5, 5) - 0.9).abs() < 0.001);

        // After another decay, should be 0.81
        grid.decay();
        assert!((grid.energy_at(5, 5) - 0.81).abs() < 0.001);
    }

    #[test]
    fn test_oscillating_source() {
        let source = GridEnergySource::with_type(
            50.0,
            50.0,
            1.0,
            10.0,
            EnergySourceType::Oscillating {
                period: 4,
                phase: 0.0,
            },
        );

        // At age 0, sin(0) = 0, so intensity should be 0.5 * 1.0 = 0.5
        assert!((source.effective_intensity() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_decaying_source() {
        let mut source = GridEnergySource::with_type(
            50.0,
            50.0,
            1.0,
            10.0,
            EnergySourceType::Decaying { half_life: 100 },
        );

        // At age 0, intensity should be 1.0
        assert!((source.effective_intensity() - 1.0).abs() < 0.01);

        // At half-life, intensity should be 0.5
        source.age = 100;
        assert!((source.effective_intensity() - 0.5).abs() < 0.01);

        // At 2x half-life, intensity should be 0.25
        source.age = 200;
        assert!((source.effective_intensity() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_tape_energy() {
        let mut tape = TapeEnergy::with_max(1000);

        // Acquire energy
        tape.acquire(0.5, 1000.0); // 0.5 * 1000 = 500 steps
        assert_eq!(tape.reserve, 500);
        assert_eq!(tape.gained_this_epoch, 500);

        // Steps per run should match reserve
        assert_eq!(tape.steps_per_run(), 500);

        // Consume energy
        tape.consume(200);
        assert_eq!(tape.reserve, 300);
        assert_eq!(tape.consumed_this_epoch, 200);

        // Reset epoch
        tape.reset_epoch();
        assert_eq!(tape.gained_this_epoch, 0);
        assert_eq!(tape.consumed_this_epoch, 0);
    }

    #[test]
    fn test_tape_energy_cap() {
        let mut tape = TapeEnergy::with_max(100);

        // Acquiring more than max should cap at max
        tape.acquire(1.0, 1000.0); // Would give 1000, but capped at 100
        assert_eq!(tape.reserve, 100);
    }

    #[test]
    fn test_energy_to_color() {
        // Zero energy should be dark
        let (r, g, b) = super::energy_to_color(0.0);
        assert!(r < 20 && g < 20 && b < 20);

        // Max energy should be bright
        let (r, g, b) = super::energy_to_color(1.0);
        assert!(r > 200);

        // Mid energy should have some green
        let (r, g, b) = super::energy_to_color(0.5);
        assert!(g > 100);
    }

    #[test]
    fn test_render_image() {
        let mut grid = EnergyGrid::new(10, 10);
        grid.set_energy(5, 5, 0.8);

        let (w, h, pixels) = grid.render_image();
        assert_eq!(w, 10);
        assert_eq!(h, 10);
        assert_eq!(pixels.len(), 10 * 10 * 3);

        // Center pixel should be brighter than corner
        let center_idx = (5 * 10 + 5) * 3;
        let corner_idx = 0;
        let center_brightness = pixels[center_idx] as u32 + pixels[center_idx + 1] as u32 + pixels[center_idx + 2] as u32;
        let corner_brightness = pixels[corner_idx] as u32 + pixels[corner_idx + 1] as u32 + pixels[corner_idx + 2] as u32;
        assert!(center_brightness > corner_brightness);
    }

    #[test]
    fn test_render_scaled() {
        let mut grid = EnergyGrid::new(5, 5);
        grid.set_energy(2, 2, 0.5);

        let (w, h, pixels) = grid.render_scaled(4);
        assert_eq!(w, 20);
        assert_eq!(h, 20);
        assert_eq!(pixels.len(), 20 * 20 * 3);
    }

    #[test]
    fn test_energy_grid_system_creation() {
        let config = EnergyGridConfig {
            enabled: true,
            resolution: 1,
            acquisition_rate: 1000.0,
            max_reserve: 1000,
            ..Default::default()
        };

        let system = super::EnergyGridSystem::new(config, 10, 10);
        assert!(system.is_enabled());
        assert_eq!(system.tape_energies.len(), 100);
        assert_eq!(system.grid.width, 10);
        assert_eq!(system.grid.height, 10);
    }

    #[test]
    fn test_energy_grid_system_disabled() {
        let system = super::EnergyGridSystem::disabled(10, 10);
        assert!(!system.is_enabled());

        // Disabled system should return max_reserve for all tapes
        assert_eq!(system.steps_per_run(0), system.config.max_reserve);
    }

    #[test]
    fn test_energy_acquisition() {
        let mut config = EnergyGridConfig {
            enabled: true,
            resolution: 1,
            acquisition_rate: 100.0,
            max_reserve: 1000,
            diffusion_rate: 0.0,
            decay_rate: 0.0,
            ..Default::default()
        };

        // Add a source at position (5, 5)
        config.sources.push(super::EnergyGridSourceConfig {
            source_type: "constant".to_string(),
            x: 5.0,
            y: 5.0,
            intensity: 1.0,
            radius: 3.0,
            period: None,
            phase: None,
            half_life: None,
            velocity_x: None,
            velocity_y: None,
        });

        let mut system = super::EnergyGridSystem::new(config, 10, 10);

        // Run one update
        system.update();

        // Tape at (5, 5) should have energy
        let center_energy = system.tape_energy(55).reserve; // 5*10 + 5 = 55
        assert!(center_energy > 0, "Center tape should have energy");

        // Tape at (0, 0) should have no energy (far from source)
        let corner_energy = system.tape_energy(0).reserve;
        assert!(corner_energy < center_energy, "Corner should have less energy than center");
    }

    #[test]
    fn test_energy_system_metrics() {
        let config = EnergyGridConfig {
            enabled: true,
            ..Default::default()
        };

        let system = super::EnergyGridSystem::new(config, 10, 10);
        let metrics = system.metrics();

        assert_eq!(metrics.epoch, 0);
        assert_eq!(metrics.num_sources, 0);
    }
}
