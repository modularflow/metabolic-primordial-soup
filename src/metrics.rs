//! Metrics for tracking simulation phase transitions
//!
//! Uses Brotli compression ratio as a key metric for detecting when
//! self-replicators emerge from the primordial soup.
//!
//! Based on: "Computational Life: How Well-formed, Self-replicating Programs
//! Emerge from Simple Interaction" (Agüera y Arcas et al., 2024)
//! https://arxiv.org/pdf/2406.19108

#![allow(dead_code)] // Metrics are conditionally used based on config

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::time::Instant;

use crate::bff::SINGLE_TAPE_SIZE;

/// Configuration for metrics collection
#[derive(Clone, Debug)]
pub struct MetricsConfig {
    /// Whether metrics collection is enabled
    pub enabled: bool,
    /// Interval (in epochs) between metric calculations
    pub interval: usize,
    /// Path to CSV output file (None = stdout only)
    pub output_path: Option<String>,
    /// Brotli compression quality (1-11, lower = faster)
    pub brotli_quality: u32,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: 1000,
            output_path: None,
            brotli_quality: 4, // Balance between speed and compression
        }
    }
}

/// Collected metrics for a single epoch
#[derive(Clone, Debug)]
pub struct EpochMetrics {
    pub epoch: usize,
    /// Compression ratio of live (non-zero) programs only
    pub compression_ratio: f64,
    pub compressed_size: usize,
    /// Size of live programs only (excludes dead/zeroed programs)
    pub original_size: usize,
    pub unique_bytes: usize,
    pub zero_byte_fraction: f64,
    pub command_fraction: f64,
    pub computation_time_ms: f64,
    /// Number of live (non-zero) programs
    pub live_program_count: usize,
    /// Total number of programs
    pub total_program_count: usize,
}

impl EpochMetrics {
    /// Format as CSV row
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{:.4},{},{},{},{:.4},{:.4},{:.2},{},{}",
            self.epoch,
            self.compression_ratio,
            self.compressed_size,
            self.original_size,
            self.unique_bytes,
            self.zero_byte_fraction,
            self.command_fraction,
            self.computation_time_ms,
            self.live_program_count,
            self.total_program_count,
        )
    }

    /// CSV header
    pub fn csv_header() -> &'static str {
        "epoch,compression_ratio,compressed_size,original_size,unique_bytes,zero_byte_fraction,command_fraction,computation_time_ms,live_program_count,total_program_count"
    }

    /// Fraction of programs that are alive
    pub fn live_fraction(&self) -> f64 {
        if self.total_program_count == 0 {
            0.0
        } else {
            self.live_program_count as f64 / self.total_program_count as f64
        }
    }
}

/// Metrics tracker that collects and logs simulation metrics
pub struct MetricsTracker {
    config: MetricsConfig,
    csv_writer: Option<BufWriter<File>>,
    history: Vec<EpochMetrics>,
    last_ratio: f64,
    phase_transition_detected: bool,
    phase_transition_epoch: Option<usize>,
    /// Baseline compression ratio from early measurements (before deaths)
    baseline_ratio: Option<f64>,
    /// Count of consecutive elevated measurements
    elevated_count: usize,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new(config: MetricsConfig) -> std::io::Result<Self> {
        let csv_writer = if let Some(ref path) = config.output_path {
            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;
            let mut writer = BufWriter::new(file);
            writeln!(writer, "{}", EpochMetrics::csv_header())?;
            Some(writer)
        } else {
            None
        };

        Ok(Self {
            config,
            csv_writer,
            history: Vec::new(),
            last_ratio: 1.0,
            phase_transition_detected: false,
            phase_transition_epoch: None,
            baseline_ratio: None,
            elevated_count: 0,
        })
    }

    /// Check if we should collect metrics this epoch
    pub fn should_collect(&self, epoch: usize) -> bool {
        self.config.enabled && epoch % self.config.interval == 0
    }

    /// Collect metrics for the current soup state
    pub fn collect(&mut self, epoch: usize, soup: &[u8]) -> EpochMetrics {
        let start = Instant::now();
        
        // Count total programs and filter out dead (all-zero) programs
        let total_program_count = soup.len() / SINGLE_TAPE_SIZE;
        let live_programs = filter_live_programs(soup);
        let live_program_count = live_programs.len() / SINGLE_TAPE_SIZE;
        
        // Calculate Brotli compression ratio on LIVE programs only
        // This prevents dead (zeroed) programs from skewing the metric
        let (compressed_size, compression_ratio) = if live_programs.is_empty() {
            (0, 1.0) // No live programs, default ratio
        } else {
            self.calculate_compression_ratio(&live_programs)
        };
        
        // Count unique bytes (in entire soup for general stats)
        let mut byte_counts = [0u32; 256];
        for &b in soup {
            byte_counts[b as usize] += 1;
        }
        let unique_bytes = byte_counts.iter().filter(|&&c| c > 0).count();
        
        // Zero byte fraction (dead/empty programs)
        let zero_count = byte_counts[0] as f64;
        let zero_byte_fraction = zero_count / soup.len() as f64;
        
        // BFF command fraction (in live programs only for meaningful metric)
        let command_count = live_programs.iter().filter(|&&b| is_bff_command(b)).count();
        let command_fraction = if live_programs.is_empty() {
            0.0
        } else {
            command_count as f64 / live_programs.len() as f64
        };
        
        let computation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        let metrics = EpochMetrics {
            epoch,
            compression_ratio,
            compressed_size,
            original_size: live_programs.len(),
            unique_bytes,
            zero_byte_fraction,
            command_fraction,
            computation_time_ms,
            live_program_count,
            total_program_count,
        };
        
        // Phase transition detection - improved to avoid false positives from first death wave
        // 
        // Strategy: Compare against BASELINE (early measurements), not just last measurement.
        // Require SUSTAINED elevation over multiple consecutive measurements.
        let live_fraction = metrics.live_fraction();
        
        // Establish baseline from first few measurements (before significant deaths)
        if self.baseline_ratio.is_none() && live_fraction > 0.9 {
            // Still in early phase with most programs alive - this is our baseline
            self.baseline_ratio = Some(compression_ratio);
        }
        
        // Phase transition detection criteria:
        // 1. Not already detected
        // 2. At least 10% of programs still alive (meaningful sample)
        // 3. Compression ratio significantly above baseline (true replicators compress MUCH better)
        // 4. Absolute compression ratio above threshold (replicators should hit 3x+)
        // 5. Sustained for multiple consecutive measurements
        let min_live_fraction = 0.1;
        let baseline = self.baseline_ratio.unwrap_or(1.0);
        let ratio_above_baseline = compression_ratio > baseline * 2.0;  // Must be 2x baseline
        let ratio_absolute_high = compression_ratio > 2.5;  // True replicators compress well
        
        if !self.phase_transition_detected && live_fraction >= min_live_fraction {
            if ratio_above_baseline && ratio_absolute_high {
                self.elevated_count += 1;
                
                // Require 5 consecutive elevated measurements to confirm phase transition
                if self.elevated_count >= 5 {
                    self.phase_transition_detected = true;
                    self.phase_transition_epoch = Some(epoch);
                    eprintln!("\n╔══════════════════════════════════════════════════════════╗");
                    eprintln!("║  🧬 PHASE TRANSITION DETECTED at epoch {}!", epoch);
                    eprintln!("║  Compression ratio: {:.2} (baseline: {:.2})", compression_ratio, baseline);
                    eprintln!("║  Live programs: {} / {} ({:.1}%)", live_program_count, total_program_count, live_fraction * 100.0);
                    eprintln!("║  Sustained elevation for {} measurements", self.elevated_count);
                    eprintln!("║  Self-replicators have likely emerged!");
                    eprintln!("╚══════════════════════════════════════════════════════════╝\n");
                }
            } else {
                // Reset counter if ratio drops
                self.elevated_count = 0;
            }
        }
        
        self.last_ratio = compression_ratio;
        
        // Log to CSV
        if let Some(ref mut writer) = self.csv_writer {
            let _ = writeln!(writer, "{}", metrics.to_csv_row());
            let _ = writer.flush();
        }
        
        self.history.push(metrics.clone());
        
        metrics
    }

    /// Calculate Brotli compression ratio
    fn calculate_compression_ratio(&self, data: &[u8]) -> (usize, f64) {
        use brotli::enc::BrotliEncoderParams;
        
        let mut compressed = Vec::new();
        let mut params = BrotliEncoderParams::default();
        params.quality = self.config.brotli_quality as i32;
        
        let result = brotli::BrotliCompress(
            &mut std::io::Cursor::new(data),
            &mut compressed,
            &params
        );
        
        match result {
            Ok(_) => {
                let ratio = data.len() as f64 / compressed.len() as f64;
                (compressed.len(), ratio)
            }
            Err(_) => (data.len(), 1.0) // Fallback if compression fails
        }
    }

    /// Get the phase transition epoch if detected
    pub fn phase_transition_epoch(&self) -> Option<usize> {
        self.phase_transition_epoch
    }

    /// Get all collected metrics history
    pub fn history(&self) -> &[EpochMetrics] {
        &self.history
    }

    /// Print a summary of metrics
    pub fn print_summary(&self) {
        if self.history.is_empty() {
            return;
        }

        let first = self.history.first().unwrap();
        let last = self.history.last().unwrap();
        
        println!("\n┌─────────────────────────────────────────────────────────────┐");
        println!("│                    METRICS SUMMARY                          │");
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│  Epochs tracked:        {:>8} → {:>8}                  │", first.epoch, last.epoch);
        println!("│  Compression ratio:     {:>8.2} → {:>8.2}  (live only)     │", first.compression_ratio, last.compression_ratio);
        println!("│  Live programs:         {:>8} → {:>8}                  │", first.live_program_count, last.live_program_count);
        println!("│  Live fraction:         {:>7.1}% → {:>7.1}%                  │",
            first.live_fraction() * 100.0, last.live_fraction() * 100.0);
        println!("│  Unique bytes:          {:>8} → {:>8}                  │", first.unique_bytes, last.unique_bytes);
        println!("│  Zero byte fraction:    {:>7.1}% → {:>7.1}%                  │", 
            first.zero_byte_fraction * 100.0, last.zero_byte_fraction * 100.0);
        println!("│  Command fraction:      {:>7.1}% → {:>7.1}%  (live only)     │",
            first.command_fraction * 100.0, last.command_fraction * 100.0);
        
        if let Some(epoch) = self.phase_transition_epoch {
            println!("├─────────────────────────────────────────────────────────────┤");
            println!("│  🧬 Phase transition detected at epoch {:>8}            │", epoch);
        }
        
        println!("└─────────────────────────────────────────────────────────────┘");
        
        if let Some(ref path) = self.config.output_path {
            println!("  Metrics saved to: {}", path);
        }
    }
}

/// Check if a byte is a BFF command
fn is_bff_command(b: u8) -> bool {
    matches!(b, b'+' | b'-' | b'>' | b'<' | b'{' | b'}' | b'[' | b']' | b'.' | b',')
}

/// Check if a program (tape) is dead (all zeros)
fn is_program_dead(program: &[u8]) -> bool {
    program.iter().all(|&b| b == 0)
}

/// Filter out dead (all-zero) programs from the soup
/// Returns a new Vec containing only the bytes from live programs
fn filter_live_programs(soup: &[u8]) -> Vec<u8> {
    soup.chunks(SINGLE_TAPE_SIZE)
        .filter(|program| !is_program_dead(program))
        .flatten()
        .copied()
        .collect()
}

/// Quick compression ratio calculation (for one-off checks)
/// Filters out dead (all-zero) programs before calculating
pub fn quick_compression_ratio(data: &[u8]) -> f64 {
    use brotli::enc::BrotliEncoderParams;
    
    // Filter out dead programs
    let live_data = filter_live_programs(data);
    if live_data.is_empty() {
        return 1.0;
    }
    
    let mut compressed = Vec::new();
    let mut params = BrotliEncoderParams::default();
    params.quality = 4; // Fast but reasonable compression
    
    match brotli::BrotliCompress(
        &mut std::io::Cursor::new(&live_data),
        &mut compressed,
        &params
    ) {
        Ok(_) => live_data.len() as f64 / compressed.len() as f64,
        Err(_) => 1.0
    }
}

/// Per-simulation metrics for multi-sim mode
#[derive(Clone, Debug)]
pub struct SimMetrics {
    pub sim_idx: usize,
    /// Compression ratio of live programs only
    pub compression_ratio: f64,
    pub zero_byte_fraction: f64,
    /// Command fraction of live programs only
    pub command_fraction: f64,
    pub live_program_count: usize,
    pub total_program_count: usize,
}

/// Calculate metrics for each simulation in a multi-sim setup
pub fn calculate_per_sim_metrics(
    all_soup: &[u8],
    num_sims: usize,
    programs_per_sim: usize,
    bytes_per_program: usize,
) -> Vec<SimMetrics> {
    let bytes_per_sim = programs_per_sim * bytes_per_program;
    
    (0..num_sims)
        .map(|sim_idx| {
            let start = sim_idx * bytes_per_sim;
            let end = start + bytes_per_sim;
            let sim_soup = &all_soup[start..end];
            
            // Filter out dead programs for compression ratio
            let live_soup = filter_live_programs(sim_soup);
            let live_program_count = live_soup.len() / bytes_per_program;
            let total_program_count = programs_per_sim;
            
            let compression_ratio = if live_soup.is_empty() {
                1.0
            } else {
                quick_compression_ratio_raw(&live_soup)
            };
            
            let zero_count = sim_soup.iter().filter(|&&b| b == 0).count();
            let zero_byte_fraction = zero_count as f64 / sim_soup.len() as f64;
            
            // Command fraction on live programs only
            let command_count = live_soup.iter().filter(|&&b| is_bff_command(b)).count();
            let command_fraction = if live_soup.is_empty() {
                0.0
            } else {
                command_count as f64 / live_soup.len() as f64
            };
            
            SimMetrics {
                sim_idx,
                compression_ratio,
                zero_byte_fraction,
                command_fraction,
                live_program_count,
                total_program_count,
            }
        })
        .collect()
}

/// Raw compression ratio calculation (no filtering, for pre-filtered data)
fn quick_compression_ratio_raw(data: &[u8]) -> f64 {
    use brotli::enc::BrotliEncoderParams;
    
    if data.is_empty() {
        return 1.0;
    }
    
    let mut compressed = Vec::new();
    let mut params = BrotliEncoderParams::default();
    params.quality = 4;
    
    match brotli::BrotliCompress(
        &mut std::io::Cursor::new(data),
        &mut compressed,
        &params
    ) {
        Ok(_) => data.len() as f64 / compressed.len() as f64,
        Err(_) => 1.0
    }
}

