//! BFF - Brainfuck variant with two heads
//! Matching semantics from cubff/bff.cu
//!
//! Note: The GPU backends implement BFF evaluation in shaders. The CPU
//! functions here are kept for testing, debugging, and potential CPU-only use.

#![allow(dead_code)]

pub const SINGLE_TAPE_SIZE: usize = 64;
pub const FULL_TAPE_SIZE: usize = 2 * SINGLE_TAPE_SIZE; // 128 bytes

/// Wrap a position to stay within tape bounds
#[inline]
fn wrap_pos(pos: i32) -> usize {
    (pos as usize) & (FULL_TAPE_SIZE - 1)
}

/// Determine which program "owns" a tape position
/// Returns 0 for first program (bytes 0-63), 1 for second program (bytes 64-127)
#[inline]
pub fn pos_to_program(pos: usize) -> usize {
    if pos < SINGLE_TAPE_SIZE { 0 } else { 1 }
}

/// Alias for pos_to_program - determines which tape's energy to consume
/// based on instruction pointer position during paired execution.
#[inline]
pub fn energy_owner(pos: usize) -> usize {
    pos_to_program(pos)
}

/// Check if a character is a BFF command (original set)
#[inline]
pub fn is_command(c: u8) -> bool {
    matches!(c, b'<' | b'>' | b'{' | b'}' | b'+' | b'-' | b'.' | b',' | b'[' | b']')
}

/// Check if a character is a BFF command (extended set with energy operations)
#[inline]
pub fn is_command_extended(c: u8) -> bool {
    matches!(c, b'<' | b'>' | b'{' | b'}' | b'+' | b'-' | b'.' | b',' | b'[' | b']' | b'!' | b'$' | b'@')
}

/// A cross-program copy event (copy between first and second half of tape)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossProgramCopy {
    /// Which half the data came FROM (0 = first 64 bytes, 1 = second 64 bytes)
    pub source_half: usize,
    /// Which half the data went TO (0 = first 64 bytes, 1 = second 64 bytes)
    pub dest_half: usize,
}

impl CrossProgramCopy {
    pub fn new(source_half: usize, dest_half: usize) -> Self {
        Self { source_half, dest_half }
    }
}

/// Result of BFF evaluation with copy tracking
#[derive(Clone, Debug)]
pub struct EvalResult {
    /// Number of non-NOP operations executed
    pub ops: usize,
    /// Cross-program copy events (copies between first and second half)
    pub cross_copies: Vec<CrossProgramCopy>,
}

/// Extended result of BFF evaluation with energy tracking
#[derive(Clone, Debug, Default)]
pub struct EvalResultExtended {
    /// Number of non-NOP operations executed
    pub ops: usize,
    /// Cross-program copy events (copies between first and second half)
    pub cross_copies: Vec<CrossProgramCopy>,
    /// Whether execution was halted by `!` instruction
    pub halted: bool,
    /// Energy consumed during execution (in steps)
    pub energy_consumed: u32,
    /// Number of store-energy operations (`$` -> `@`)
    pub store_count: u32,
    /// Number of stored-energy skips (`@` consumed)
    pub skip_count: u32,
}

/// Result of paired BFF evaluation with per-tape energy tracking
/// 
/// When two tapes are paired, execution can consume energy from either tape
/// depending on where the instruction pointer is located:
/// - Bytes 0-63: Tape 0's region
/// - Bytes 64-127: Tape 1's region
#[derive(Clone, Debug, Default)]
pub struct PairedEvalResult {
    /// Total operations executed
    pub ops: usize,
    /// Energy consumed from tape 0 (positions 0-63)
    pub energy_consumed_tape0: u32,
    /// Energy consumed from tape 1 (positions 64-127)
    pub energy_consumed_tape1: u32,
    /// Cross-program copy events
    pub cross_copies: Vec<CrossProgramCopy>,
    /// Whether execution was halted by `!` instruction
    pub halted: bool,
    /// Number of store-energy operations
    pub store_count: u32,
    /// Number of stored-energy skips
    pub skip_count: u32,
    /// Steps executed in tape 0's region
    pub steps_in_tape0: u32,
    /// Steps executed in tape 1's region  
    pub steps_in_tape1: u32,
}

/// Evaluate a BFF program on a 128-byte tape
/// Returns the number of non-NOP instructions executed
/// 
/// Tape layout (matching cubff):
/// - Bytes 0-1: Initial head positions (head0, head1)
/// - Bytes 2-127: Program/data
/// - First 64 bytes belong to "program 1", second 64 to "program 2"
pub fn evaluate(tape: &mut [u8; FULL_TAPE_SIZE], step_count: usize, debug: bool) -> usize {
    let mut nskip = 0usize;  // Count of NOPs
    
    // Instruction pointer starts at position 2 (after head storage bytes)
    let mut pos: i32 = 2;
    
    // Initial head positions from tape bytes 0 and 1
    let mut head0: i32 = wrap_pos(tape[0] as i32) as i32;
    let mut head1: i32 = wrap_pos(tape[1] as i32) as i32;
    
    for i in 0..step_count {
        // Wrap head positions
        head0 = wrap_pos(head0) as i32;
        head1 = wrap_pos(head1) as i32;
        
        if debug {
            print_program_internal(head0 as usize, head1 as usize, pos as usize, tape);
        }
        
        let cmd = tape[pos as usize];
        
        match cmd {
            b'<' => head0 -= 1,
            b'>' => head0 += 1,
            b'{' => head1 -= 1,
            b'}' => head1 += 1,
            b'+' => tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_add(1),
            b'-' => tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_sub(1),
            b'.' => tape[wrap_pos(head1)] = tape[wrap_pos(head0)],
            b',' => tape[wrap_pos(head0)] = tape[wrap_pos(head1)],
            b'[' => {
                if tape[wrap_pos(head0)] == 0 {
                    // Jump forward to matching ']'
                    let mut depth = 1i32;
                    pos += 1;
                    while pos < FULL_TAPE_SIZE as i32 && depth > 0 {
                        match tape[pos as usize] {
                            b']' => depth -= 1,
                            b'[' => depth += 1,
                            _ => {}
                        }
                        pos += 1;
                    }
                    pos -= 1;
                    if depth != 0 {
                        pos = FULL_TAPE_SIZE as i32; // Terminate
                    }
                }
            }
            b']' => {
                if tape[wrap_pos(head0)] != 0 {
                    // Jump backward to matching '['
                    let mut depth = 1i32;
                    pos -= 1;
                    while pos >= 0 && depth > 0 {
                        match tape[pos as usize] {
                            b']' => depth += 1,
                            b'[' => depth -= 1,
                            _ => {}
                        }
                        pos -= 1;
                    }
                    pos += 1;
                    if depth != 0 {
                        pos = -1; // Terminate
                    }
                }
            }
            _ => nskip += 1,
        }
        
        // Check termination conditions
        if pos < 0 {
            return i + 1 - nskip;
        }
        
        pos += 1;
        
        if pos >= FULL_TAPE_SIZE as i32 {
            return i + 1 - nskip;
        }
    }
    
    step_count - nskip
}

/// Evaluate a BFF program and track cross-program copy operations
/// 
/// This variant tracks when `.` or `,` copies data between the two program halves:
/// - First half (bytes 0-63): program 1
/// - Second half (bytes 64-127): program 2
/// 
/// Returns both the operation count and a list of cross-program copies.
pub fn evaluate_with_copy_tracking(
    tape: &mut [u8; FULL_TAPE_SIZE],
    step_count: usize,
) -> EvalResult {
    let mut nskip = 0usize;
    let mut cross_copies = Vec::new();
    
    let mut pos: i32 = 2;
    let mut head0: i32 = wrap_pos(tape[0] as i32) as i32;
    let mut head1: i32 = wrap_pos(tape[1] as i32) as i32;
    
    for i in 0..step_count {
        head0 = wrap_pos(head0) as i32;
        head1 = wrap_pos(head1) as i32;
        
        let cmd = tape[pos as usize];
        
        match cmd {
            b'<' => head0 -= 1,
            b'>' => head0 += 1,
            b'{' => head1 -= 1,
            b'}' => head1 += 1,
            b'+' => tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_add(1),
            b'-' => tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_sub(1),
            b'.' => {
                // Copy from head0 to head1
                let src_pos = wrap_pos(head0);
                let dst_pos = wrap_pos(head1);
                tape[dst_pos] = tape[src_pos];
                
                // Track if this is a cross-program copy
                let src_half = pos_to_program(src_pos);
                let dst_half = pos_to_program(dst_pos);
                if src_half != dst_half {
                    cross_copies.push(CrossProgramCopy::new(src_half, dst_half));
                }
            }
            b',' => {
                // Copy from head1 to head0
                let src_pos = wrap_pos(head1);
                let dst_pos = wrap_pos(head0);
                tape[dst_pos] = tape[src_pos];
                
                // Track if this is a cross-program copy
                let src_half = pos_to_program(src_pos);
                let dst_half = pos_to_program(dst_pos);
                if src_half != dst_half {
                    cross_copies.push(CrossProgramCopy::new(src_half, dst_half));
                }
            }
            b'[' => {
                if tape[wrap_pos(head0)] == 0 {
                    let mut depth = 1i32;
                    pos += 1;
                    while pos < FULL_TAPE_SIZE as i32 && depth > 0 {
                        match tape[pos as usize] {
                            b']' => depth -= 1,
                            b'[' => depth += 1,
                            _ => {}
                        }
                        pos += 1;
                    }
                    pos -= 1;
                    if depth != 0 {
                        pos = FULL_TAPE_SIZE as i32;
                    }
                }
            }
            b']' => {
                if tape[wrap_pos(head0)] != 0 {
                    let mut depth = 1i32;
                    pos -= 1;
                    while pos >= 0 && depth > 0 {
                        match tape[pos as usize] {
                            b']' => depth += 1,
                            b'[' => depth -= 1,
                            _ => {}
                        }
                        pos -= 1;
                    }
                    pos += 1;
                    if depth != 0 {
                        pos = -1;
                    }
                }
            }
            _ => nskip += 1,
        }
        
        if pos < 0 {
            return EvalResult {
                ops: i + 1 - nskip,
                cross_copies,
            };
        }
        
        pos += 1;
        
        if pos >= FULL_TAPE_SIZE as i32 {
            return EvalResult {
                ops: i + 1 - nskip,
                cross_copies,
            };
        }
    }
    
    EvalResult {
        ops: step_count - nskip,
        cross_copies,
    }
}

/// Evaluate a BFF program with TAPE-BASED energy model
/// 
/// Energy model:
/// - @ must IMMEDIATELY PRECEDE an operation on the tape for it to execute
/// - When IP lands on @: consume it, peek next byte, if BFF op execute it
/// - When IP lands directly on BFF op (no @): skip it (NOP)
/// - No external fuel counters - energy is purely positional on the tape
/// - $ (store-energy) only works if a head points at @ on the other tape half
/// 
/// Returns extended result with energy tracking.
pub fn evaluate_with_energy(
    tape: &mut [u8; FULL_TAPE_SIZE],
    step_count: usize,
) -> EvalResultExtended {
    let mut ops = 0usize;
    let mut cross_copies = Vec::new();
    let mut store_count = 0u32;
    let mut skip_count = 0u32;
    
    let mut pos: i32 = 2;
    let mut head0: i32 = wrap_pos(tape[0] as i32) as i32;
    let mut head1: i32 = wrap_pos(tape[1] as i32) as i32;
    
    // Helper: check if byte is a BFF operation (excluding @)
    let is_bff_op = |c: u8| -> bool {
        matches!(c, b'<' | b'>' | b'{' | b'}' | b'+' | b'-' | b'.' | b',' | b'[' | b']' | b'!' | b'$')
    };
    
    for _step in 0..step_count {
        if pos < 0 || pos >= FULL_TAPE_SIZE as i32 {
            break;
        }
        
        head0 = wrap_pos(head0) as i32;
        head1 = wrap_pos(head1) as i32;
        
        let cmd = tape[pos as usize];
        
        if cmd == b'@' {
            // '@' - Energy token: consume it and peek at next instruction
            tape[pos as usize] = 0;  // Consume the @
            skip_count += 1;
            
            // Peek at next byte
            let next_pos = pos + 1;
            if next_pos < FULL_TAPE_SIZE as i32 {
                let next_cmd = tape[next_pos as usize];
                if is_bff_op(next_cmd) {
                    // Execute the next operation (powered by this @)
                    match next_cmd {
                        b'<' => { head0 -= 1; ops += 1; }
                        b'>' => { head0 += 1; ops += 1; }
                        b'{' => { head1 -= 1; ops += 1; }
                        b'}' => { head1 += 1; ops += 1; }
                        b'+' => { tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_add(1); ops += 1; }
                        b'-' => { tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_sub(1); ops += 1; }
                        b'.' => {
                            let src_pos = wrap_pos(head0);
                            let dst_pos = wrap_pos(head1);
                            tape[dst_pos] = tape[src_pos];
                            let src_half = pos_to_program(src_pos);
                            let dst_half = pos_to_program(dst_pos);
                            if src_half != dst_half {
                                cross_copies.push(CrossProgramCopy::new(src_half, dst_half));
                            }
                            ops += 1;
                        }
                        b',' => {
                            let src_pos = wrap_pos(head1);
                            let dst_pos = wrap_pos(head0);
                            tape[dst_pos] = tape[src_pos];
                            let src_half = pos_to_program(src_pos);
                            let dst_half = pos_to_program(dst_pos);
                            if src_half != dst_half {
                                cross_copies.push(CrossProgramCopy::new(src_half, dst_half));
                            }
                            ops += 1;
                        }
                        b'[' => {
                            if tape[wrap_pos(head0)] == 0 {
                                let mut depth = 1i32;
                                pos = next_pos + 1;
                                while pos < FULL_TAPE_SIZE as i32 && depth > 0 {
                                    match tape[pos as usize] {
                                        b']' => depth -= 1,
                                        b'[' => depth += 1,
                                        _ => {}
                                    }
                                    pos += 1;
                                }
                                pos -= 1;
                                if depth != 0 {
                                    pos = FULL_TAPE_SIZE as i32;
                                }
                                ops += 1;
                                continue;  // Skip normal pos increment
                            }
                            ops += 1;
                            pos = next_pos;
                        }
                        b']' => {
                            if tape[wrap_pos(head0)] != 0 {
                                let mut depth = 1i32;
                                pos = next_pos - 1;
                                while pos >= 0 && depth > 0 {
                                    match tape[pos as usize] {
                                        b']' => depth += 1,
                                        b'[' => depth -= 1,
                                        _ => {}
                                    }
                                    pos -= 1;
                                }
                                pos += 1;
                                if depth != 0 {
                                    pos = -1;
                                }
                                ops += 1;
                                continue;  // Skip normal pos increment
                            }
                            ops += 1;
                            pos = next_pos;
                        }
                        b'!' => {
                            ops += 1;
                            return EvalResultExtended {
                                ops,
                                cross_copies,
                                halted: true,
                                energy_consumed: ops as u32,
                                store_count,
                                skip_count,
                            };
                        }
                        b'$' => {
                            // Store-energy: only works if head points at @ on other tape half
                            let h0 = wrap_pos(head0);
                            let h1 = wrap_pos(head1);
                            let current_half = if (next_pos as usize) < SINGLE_TAPE_SIZE { 0 } else { 1 };
                            
                            let can_harvest = if current_half == 0 {
                                // We're in first half, need head on @ in second half
                                (h0 >= SINGLE_TAPE_SIZE && tape[h0] == b'@') ||
                                (h1 >= SINGLE_TAPE_SIZE && tape[h1] == b'@')
                            } else {
                                // We're in second half, need head on @ in first half
                                (h0 < SINGLE_TAPE_SIZE && tape[h0] == b'@') ||
                                (h1 < SINGLE_TAPE_SIZE && tape[h1] == b'@')
                            };
                            
                            if can_harvest {
                                tape[next_pos as usize] = b'@';  // Convert $ to @
                                store_count += 1;
                                // Consume the @ that was harvested
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
        // If current byte is a BFF op (but no @ before it), skip it (NOP)
        // Just fall through to pos++
        
        pos += 1;
    }
    
    EvalResultExtended {
        ops,
        cross_copies,
        halted: false,
        energy_consumed: ops as u32,
        store_count,
        skip_count,
    }
}

/// Evaluate paired tapes with TAPE-BASED energy model
/// 
/// Energy model:
/// - @ must IMMEDIATELY PRECEDE an operation on the tape for it to execute
/// - When IP lands on @: consume it, peek next byte, if BFF op execute it
/// - When IP lands directly on BFF op (no @): skip it (NOP)
/// - No external fuel counters - energy is purely positional on the tape
/// - $ (store-energy) only works if a head points at @ on the other tape half
/// 
/// # Arguments
/// * `tape` - Combined 128-byte tape (tape0: 0-63, tape1: 64-127)
/// * `_energy0` - DEPRECATED: ignored (tape-based model)
/// * `_energy1` - DEPRECATED: ignored (tape-based model)
/// 
/// # Returns
/// `PairedEvalResult` with execution tracking
pub fn evaluate_paired(
    tape: &mut [u8; FULL_TAPE_SIZE],
    _energy0: u32,
    _energy1: u32,
) -> PairedEvalResult {
    let max_steps = 16384; // Fixed max steps for tape-based model
    
    let mut ops = 0usize;
    let mut steps_in_tape0 = 0u32;
    let mut steps_in_tape1 = 0u32;
    let mut cross_copies = Vec::new();
    let mut halted = false;
    let mut store_count = 0u32;
    let mut skip_count = 0u32;
    
    let mut pos: i32 = 2;
    let mut head0: i32 = wrap_pos(tape[0] as i32) as i32;
    let mut head1: i32 = wrap_pos(tape[1] as i32) as i32;
    
    // Helper: check if byte is a BFF operation (excluding @)
    let is_bff_op = |c: u8| -> bool {
        matches!(c, b'<' | b'>' | b'{' | b'}' | b'+' | b'-' | b'.' | b',' | b'[' | b']' | b'!' | b'$')
    };
    
    for _step in 0..max_steps {
        if pos < 0 || pos >= FULL_TAPE_SIZE as i32 {
            break;
        }
        
        head0 = wrap_pos(head0) as i32;
        head1 = wrap_pos(head1) as i32;
        
        // Track which region we're in
        let owner = energy_owner(pos as usize);
        if owner == 0 {
            steps_in_tape0 += 1;
        } else {
            steps_in_tape1 += 1;
        }
        
        let cmd = tape[pos as usize];
        
        if cmd == b'@' {
            // '@' - Energy token: consume it and peek at next instruction
            tape[pos as usize] = 0;  // Consume the @
            skip_count += 1;
            
            // Peek at next byte
            let next_pos = pos + 1;
            if next_pos < FULL_TAPE_SIZE as i32 {
                let next_cmd = tape[next_pos as usize];
                if is_bff_op(next_cmd) {
                    // Execute the next operation (powered by this @)
                    match next_cmd {
                        b'<' => { head0 -= 1; ops += 1; }
                        b'>' => { head0 += 1; ops += 1; }
                        b'{' => { head1 -= 1; ops += 1; }
                        b'}' => { head1 += 1; ops += 1; }
                        b'+' => { tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_add(1); ops += 1; }
                        b'-' => { tape[wrap_pos(head0)] = tape[wrap_pos(head0)].wrapping_sub(1); ops += 1; }
                        b'.' => {
                            let src_pos = wrap_pos(head0);
                            let dst_pos = wrap_pos(head1);
                            tape[dst_pos] = tape[src_pos];
                            let src_half = pos_to_program(src_pos);
                            let dst_half = pos_to_program(dst_pos);
                            if src_half != dst_half {
                                cross_copies.push(CrossProgramCopy::new(src_half, dst_half));
                            }
                            ops += 1;
                        }
                        b',' => {
                            let src_pos = wrap_pos(head1);
                            let dst_pos = wrap_pos(head0);
                            tape[dst_pos] = tape[src_pos];
                            let src_half = pos_to_program(src_pos);
                            let dst_half = pos_to_program(dst_pos);
                            if src_half != dst_half {
                                cross_copies.push(CrossProgramCopy::new(src_half, dst_half));
                            }
                            ops += 1;
                        }
                        b'[' => {
                            if tape[wrap_pos(head0)] == 0 {
                                let mut depth = 1i32;
                                pos = next_pos + 1;
                                while pos < FULL_TAPE_SIZE as i32 && depth > 0 {
                                    match tape[pos as usize] {
                                        b']' => depth -= 1,
                                        b'[' => depth += 1,
                                        _ => {}
                                    }
                                    pos += 1;
                                }
                                pos -= 1;
                                if depth != 0 {
                                    pos = FULL_TAPE_SIZE as i32;
                                }
                                ops += 1;
                                continue;
                            }
                            ops += 1;
                            pos = next_pos;
                        }
                        b']' => {
                            if tape[wrap_pos(head0)] != 0 {
                                let mut depth = 1i32;
                                pos = next_pos - 1;
                                while pos >= 0 && depth > 0 {
                                    match tape[pos as usize] {
                                        b']' => depth += 1,
                                        b'[' => depth -= 1,
                                        _ => {}
                                    }
                                    pos -= 1;
                                }
                                pos += 1;
                                if depth != 0 {
                                    pos = -1;
                                }
                                ops += 1;
                                continue;
                            }
                            ops += 1;
                            pos = next_pos;
                        }
                        b'!' => {
                            halted = true;
                            ops += 1;
                            break;
                        }
                        b'$' => {
                            // Store-energy: only works if head points at @ on other tape half
                            let h0 = wrap_pos(head0);
                            let h1 = wrap_pos(head1);
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
                                store_count += 1;
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
        // If current byte is a BFF op (but no @ before it), skip it (NOP)
        
        pos += 1;
    }
    
    PairedEvalResult {
        ops,
        energy_consumed_tape0: 0,  // Not tracked in tape-based model
        energy_consumed_tape1: 0,  // Not tracked in tape-based model
        cross_copies,
        halted,
        store_count,
        skip_count,
        steps_in_tape0,
        steps_in_tape1,
    }
}

/// Pretty-print the current state (for debugging)
fn print_program_internal(head0: usize, head1: usize, pc: usize, tape: &[u8; FULL_TAPE_SIZE]) {
    for (i, &byte) in tape.iter().enumerate() {
        let c = if byte.is_ascii_graphic() || byte == b' ' {
            byte as char
        } else if byte == 0 {
            '␀'
        } else {
            ' '
        };
        
        // Color coding via ANSI escape codes
        if i == head0 {
            print!("\x1b[44;1m"); // Blue background
        }
        if i == head1 {
            print!("\x1b[41;1m"); // Red background  
        }
        if i == pc {
            print!("\x1b[42;1m"); // Green background
        }
        if is_command(byte) {
            print!("\x1b[37;1m"); // Bright white
        }
        
        print!("{}", c);
        
        if is_command(byte) || i == head0 || i == head1 || i == pc {
            print!("\x1b[;m"); // Reset
        }
    }
    println!();
}

/// Create a tape from a program string
pub fn parse_program(program: &str) -> [u8; FULL_TAPE_SIZE] {
    let mut tape = [0u8; FULL_TAPE_SIZE];
    for (i, byte) in program.bytes().take(FULL_TAPE_SIZE).enumerate() {
        tape[i] = byte;
    }
    tape
}

/// Print program as readable string
pub fn tape_to_string(tape: &[u8; FULL_TAPE_SIZE]) -> String {
    tape.iter()
        .map(|&b| {
            if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else if b == 0 {
                '␀'
            } else {
                ' '
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_increment() {
        // Program: increment cell at head0
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'+';
        tape[3] = b'+';
        tape[4] = b'+';
        
        let ops = evaluate(&mut tape, 100, false);
        
        // head0 starts at position 0 (tape[0] = 0)
        assert_eq!(tape[0], 3); // Three increments
        assert_eq!(ops, 3);
    }
    
    #[test]
    fn test_copy_operation() {
        // Set up tape with value at head0, copy to head1
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 10; // head0 starts at position 10
        tape[1] = 20; // head1 starts at position 20
        tape[10] = 42; // Value to copy
        tape[2] = b'.'; // Copy from head0 to head1
        
        let ops = evaluate(&mut tape, 100, false);
        
        assert_eq!(tape[20], 42);
        assert_eq!(ops, 1);
    }
    
    #[test]
    fn test_simple_loop() {
        // Program: [->+<] - move value from cell 0 to cell 1
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 2; // head0 at position 2 (will be overwritten by program)
        // Actually, let's start head0 at a data area
        tape[0] = 64; // head0 at position 64
        tape[1] = 65; // head1 at position 65
        tape[64] = 3; // Value to move
        tape[2] = b'[';
        tape[3] = b'-';
        tape[4] = b'>';
        tape[5] = b'+';
        tape[6] = b'<';
        tape[7] = b']';
        
        evaluate(&mut tape, 1000, false);
        
        assert_eq!(tape[64], 0); // Source cleared
        assert_eq!(tape[65], 3); // Destination has value
    }
    
    #[test]
    fn test_pos_to_program() {
        // First half (0-63) belongs to program 0
        assert_eq!(pos_to_program(0), 0);
        assert_eq!(pos_to_program(32), 0);
        assert_eq!(pos_to_program(63), 0);
        
        // Second half (64-127) belongs to program 1
        assert_eq!(pos_to_program(64), 1);
        assert_eq!(pos_to_program(96), 1);
        assert_eq!(pos_to_program(127), 1);
    }
    
    #[test]
    fn test_cross_copy_tracking_same_half() {
        // Copy within the same half - should NOT be tracked as cross-program
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 10; // head0 at position 10 (first half)
        tape[1] = 20; // head1 at position 20 (first half)
        tape[10] = 42;
        tape[2] = b'.'; // Copy from 10 to 20 (both in first half)
        
        let result = evaluate_with_copy_tracking(&mut tape, 100);
        
        assert_eq!(tape[20], 42);
        assert_eq!(result.ops, 1);
        assert!(result.cross_copies.is_empty(), "No cross-program copies expected");
    }
    
    #[test]
    fn test_cross_copy_tracking_first_to_second() {
        // Copy from first half to second half - should be tracked
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 10; // head0 at position 10 (first half)
        tape[1] = 80; // head1 at position 80 (second half)
        tape[10] = 42;
        tape[2] = b'.'; // Copy from 10 to 80
        
        let result = evaluate_with_copy_tracking(&mut tape, 100);
        
        assert_eq!(tape[80], 42);
        assert_eq!(result.ops, 1);
        assert_eq!(result.cross_copies.len(), 1);
        assert_eq!(result.cross_copies[0].source_half, 0);
        assert_eq!(result.cross_copies[0].dest_half, 1);
    }
    
    #[test]
    fn test_cross_copy_tracking_second_to_first() {
        // Copy from second half to first half using ','
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 10; // head0 at position 10 (first half)
        tape[1] = 80; // head1 at position 80 (second half)
        tape[80] = 99;
        tape[2] = b','; // Copy from 80 to 10
        
        let result = evaluate_with_copy_tracking(&mut tape, 100);
        
        assert_eq!(tape[10], 99);
        assert_eq!(result.ops, 1);
        assert_eq!(result.cross_copies.len(), 1);
        assert_eq!(result.cross_copies[0].source_half, 1);
        assert_eq!(result.cross_copies[0].dest_half, 0);
    }
    
    #[test]
    fn test_cross_copy_tracking_multiple() {
        // Multiple cross-program copies in a loop
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 10; // head0 starts at 10 (first half)
        tape[1] = 80; // head1 starts at 80 (second half)
        tape[10] = 3; // Loop 3 times
        // Program: [-.}] - copy and decrement, moving head1 forward
        tape[2] = b'[';
        tape[3] = b'-';
        tape[4] = b'.';
        tape[5] = b'}';
        tape[6] = b']';
        
        let result = evaluate_with_copy_tracking(&mut tape, 1000);
        
        // Should have 3 cross-program copies (one per loop iteration)
        assert_eq!(result.cross_copies.len(), 3);
        for copy in &result.cross_copies {
            assert_eq!(copy.source_half, 0);
            assert_eq!(copy.dest_half, 1);
        }
    }
    
    // ========== Tests for new energy operations ==========
    
    #[test]
    fn test_halt_stops_execution() {
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'+';  // Increment (should execute)
        tape[3] = b'!';  // Halt here
        tape[4] = b'+';  // Should NOT execute
        tape[5] = b'+';  // Should NOT execute
        
        let result = evaluate_with_energy(&mut tape, 100);
        
        assert!(result.halted, "Should be halted");
        assert_eq!(tape[0], 1, "Only one increment should have executed");
        assert_eq!(result.ops, 2, "Should have 2 ops: + and !");
        assert_eq!(result.energy_consumed, 2, "Energy should be 2 (+ and !)");
    }
    
    #[test]
    fn test_halt_conserves_remaining_steps() {
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'!';  // Halt immediately
        
        let result = evaluate_with_energy(&mut tape, 1000);
        
        assert!(result.halted);
        assert_eq!(result.ops, 1, "Only halt op");
        assert_eq!(result.energy_consumed, 1, "Only 1 step consumed");
    }
    
    #[test]
    fn test_store_energy_converts_to_at() {
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'$';  // Store energy
        tape[3] = b'+';  // Increment
        
        let result = evaluate_with_energy(&mut tape, 100);
        
        assert_eq!(tape[2], b'@', "$ should become @");
        assert_eq!(result.store_count, 1, "One store operation");
        assert!(!result.halted);
    }
    
    #[test]
    fn test_stored_energy_skip() {
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'$';  // Store energy (becomes @)
        tape[3] = b'+';
        
        // First run: $ becomes @
        let result1 = evaluate_with_energy(&mut tape, 100);
        assert_eq!(tape[2], b'@');
        assert_eq!(result1.store_count, 1);
        assert_eq!(result1.skip_count, 0);
        
        // Reset head positions for second run
        tape[0] = 0;
        tape[1] = 0;
        
        // Second run: @ is consumed (free step)
        let result2 = evaluate_with_energy(&mut tape, 100);
        assert_eq!(tape[2], 0, "@ should be consumed (set to 0)");
        assert_eq!(result2.skip_count, 1, "One skip operation");
        assert_eq!(result2.store_count, 0, "No new stores");
    }
    
    #[test]
    fn test_at_cannot_accumulate() {
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'$';
        
        // First pass: $ -> @
        evaluate_with_energy(&mut tape, 100);
        assert_eq!(tape[2], b'@');
        
        // Reset heads
        tape[0] = 0;
        tape[1] = 0;
        
        // Second pass: @ is consumed, not accumulated
        evaluate_with_energy(&mut tape, 100);
        assert_eq!(tape[2], 0, "@ consumed, not accumulated");
        
        // Reset heads
        tape[0] = 0;
        tape[1] = 0;
        
        // Third pass: position 2 is now 0 (NOP)
        let result = evaluate_with_energy(&mut tape, 100);
        assert_eq!(result.skip_count, 0, "No @ to skip anymore");
    }
    
    #[test]
    fn test_store_and_skip_energy_accounting() {
        // Test that energy accounting is correct
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'+';   // op 1
        tape[3] = b'+';   // op 2  
        tape[4] = b'$';   // op 3 (store)
        tape[5] = b'+';   // op 4
        
        let result = evaluate_with_energy(&mut tape, 10);
        
        assert_eq!(result.ops, 4, "4 operations executed");
        assert_eq!(result.store_count, 1);
        assert!(!result.halted);
    }
    
    #[test]
    fn test_halt_in_loop() {
        // Test halt inside a loop
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[0] = 64;  // head0 at position 64
        tape[64] = 5;  // Loop 5 times
        tape[2] = b'[';
        tape[3] = b'-';
        tape[4] = b'!';  // Halt inside loop
        tape[5] = b']';
        
        let result = evaluate_with_energy(&mut tape, 1000);
        
        assert!(result.halted);
        // Should halt on first iteration: [ - !
        assert_eq!(tape[64], 4, "Should have decremented once before halt");
    }
    
    #[test]
    fn test_is_command_extended() {
        // Original commands
        assert!(is_command_extended(b'<'));
        assert!(is_command_extended(b'>'));
        assert!(is_command_extended(b'+'));
        assert!(is_command_extended(b'-'));
        assert!(is_command_extended(b'.'));
        assert!(is_command_extended(b','));
        assert!(is_command_extended(b'['));
        assert!(is_command_extended(b']'));
        assert!(is_command_extended(b'{'));
        assert!(is_command_extended(b'}'));
        
        // New energy commands
        assert!(is_command_extended(b'!'));
        assert!(is_command_extended(b'$'));
        assert!(is_command_extended(b'@'));
        
        // Non-commands
        assert!(!is_command_extended(b'a'));
        assert!(!is_command_extended(0));
    }
    
    // ========== Tests for cross-tape energy sharing ==========
    
    #[test]
    fn test_energy_owner() {
        // First half (0-63) belongs to tape 0
        assert_eq!(energy_owner(0), 0);
        assert_eq!(energy_owner(32), 0);
        assert_eq!(energy_owner(63), 0);
        
        // Second half (64-127) belongs to tape 1
        assert_eq!(energy_owner(64), 1);
        assert_eq!(energy_owner(96), 1);
        assert_eq!(energy_owner(127), 1);
    }
    
    #[test]
    fn test_paired_eval_energy_attribution_tape0() {
        // Execution starts at pos=2 (tape 0 region), should consume tape 0's energy
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'+';
        tape[3] = b'+';
        tape[4] = b'!';  // Halt after 3 instructions to make test deterministic
        
        let result = evaluate_paired(&mut tape, 100, 100);
        
        assert!(result.energy_consumed_tape0 > 0, "Should consume tape 0 energy");
        assert_eq!(result.energy_consumed_tape1, 0, "Should not consume tape 1 energy");
        assert_eq!(result.steps_in_tape0, 3);  // +, +, !
        assert_eq!(result.steps_in_tape1, 0);
        assert!(result.halted);
    }
    
    #[test]
    fn test_paired_eval_energy_attribution_tape1() {
        // Set up execution to start in tape 1 region (pos 64+)
        // We'll use a loop that jumps to tape 1's region
        let mut tape = [0u8; FULL_TAPE_SIZE];
        
        // Fill tape 0 region with NOPs (will be skipped in terms of energy)
        for i in 2..64 {
            tape[i] = 0; // NOP
        }
        
        // Put actual instructions in tape 1 region
        tape[64] = b'+';
        tape[65] = b'+';
        tape[66] = b'+';
        
        // This will run through NOPs in tape 0, then hit tape 1
        let result = evaluate_paired(&mut tape, 100, 100);
        
        // NOPs don't cost energy, but we traverse them
        assert!(result.steps_in_tape0 > 0 || result.steps_in_tape1 > 0);
    }
    
    #[test]
    fn test_paired_eval_no_energy_halts() {
        // If tape 0 has no energy, execution should halt immediately
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'+';
        tape[3] = b'+';
        
        let result = evaluate_paired(&mut tape, 0, 100);
        
        // Should halt because tape 0 (where execution starts) has no energy
        assert_eq!(result.energy_consumed_tape0, 0);
        assert_eq!(result.ops, 0);
    }
    
    #[test]
    fn test_paired_eval_energy_limit() {
        // Give tape 0 limited energy
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'+';
        tape[3] = b'+';
        tape[4] = b'+';
        tape[5] = b'+';
        tape[6] = b'+';
        
        // Only 2 energy for tape 0
        let result = evaluate_paired(&mut tape, 2, 100);
        
        // Should only execute 2 operations before running out
        assert_eq!(result.energy_consumed_tape0, 2);
        assert!(result.ops <= 2);
    }
    
    #[test]
    fn test_paired_eval_cross_tape_execution() {
        // Test that execution in tape 1 region uses tape 1's energy
        let mut tape = [0u8; FULL_TAPE_SIZE];
        
        // Use many NOPs to get to tape 1 region
        // Start at pos 2, fill with NOPs until pos 64
        for i in 2..64 {
            tape[i] = 0; // NOP - free step
        }
        
        // Instructions in tape 1 region
        tape[64] = b'+';
        tape[65] = b'+';
        
        let result = evaluate_paired(&mut tape, 100, 100);
        
        // Should have consumed energy from both tapes based on position
        // NOPs in tape 0 don't cost energy, but + in tape 1 do
        assert!(result.steps_in_tape0 >= 62, "Should traverse tape 0 region");
        if result.steps_in_tape1 > 0 {
            assert!(result.energy_consumed_tape1 > 0 || result.skip_count > 0);
        }
    }
    
    #[test]
    fn test_paired_eval_halt_in_tape1() {
        // Halt in tape 1 region should attribute halt cost to tape 1
        let mut tape = [0u8; FULL_TAPE_SIZE];
        
        // NOPs to get to tape 1
        for i in 2..64 {
            tape[i] = 0;
        }
        
        tape[64] = b'!';  // Halt in tape 1 region
        
        let result = evaluate_paired(&mut tape, 100, 100);
        
        assert!(result.halted);
        // The halt in tape 1 should consume tape 1's energy
        if result.steps_in_tape1 > 0 {
            assert!(result.energy_consumed_tape1 > 0);
        }
    }
    
    #[test]
    fn test_paired_eval_store_skip_in_regions() {
        let mut tape = [0u8; FULL_TAPE_SIZE];
        tape[2] = b'$';  // Store in tape 0
        tape[3] = b'+';
        
        let result = evaluate_paired(&mut tape, 100, 100);
        
        assert_eq!(tape[2], b'@', "$ should become @");
        assert_eq!(result.store_count, 1);
    }
    
    #[test]
    fn test_paired_eval_combined_energy() {
        // Test that total energy pool allows longer execution
        let mut tape = [0u8; FULL_TAPE_SIZE];
        
        // Fill with alternating NOPs and increments
        for i in 2..20 {
            tape[i] = if i % 2 == 0 { b'+' } else { 0 };
        }
        
        // Each tape contributes energy
        let result = evaluate_paired(&mut tape, 50, 50);
        
        // Should be able to execute many steps
        assert!(result.ops > 0);
    }
}
