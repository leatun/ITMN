# ITMN_FASTMAMBA Phase 6 Design Baseline

**Date:** 2026-07-29
**Target:** KV260 (xck26-sfvc784-2LV-c)
**Model:** Mamba2-130M (1 layer profiled, 24-layer decode extrapolated)
**Clock:** 100 MHz (10ns period)

## Architecture Summary

Two 16-lane compute clusters (`u_mc`, `u_mc2`) run the full M6 SSM chain in parallel:
- **Cluster1**: even groups g = 0, 2, ..., 94 (48 groups)
- **Cluster2**: odd groups g = 1, 3, ..., 95 (48 groups)

Each cluster is fully independent: own H_RegFile, own weight bank, own LUT bank, own memory read port.

### Ping-pong stages (M1A, M1B, M4, M8)
Ping-pong dispatch on MAC2 stages: c1 handles evens, c2 handles odds.
Each stage runs at ~100% MAC utilization.

### Parallel M6 (SSM scan)
Each cluster runs the full LOAD -> DAB -> SSM -> WRITE_H -> Y_MAC -> DU -> YSUM -> GATE -> WRITE_YG chain independently.
No FIFO handoff (Phase 3 substep model scrapped).
Y_MAC prefetches next (s+1) LOAD in parallel via unused BRAM cycles.

## Memory Architecture

### ram_main (URAM, mirrored)
Two SDP URAM copies with fanout writes:
- `ram_main_c1`: port_a = c1/DMA write, port_b = c1 read
- `ram_main_c2`: port_a = c1/DMA write (mirror), port_b = c2 read

Both mirrors always contain identical data. Cost: 16 URAM (2x8).

**Why mirror:** TDP RW pattern on single URAM instance causes Vivado to fall back to 36 BRAM. Mirror keeps URAM inference clean.

### ram_weight_A / ram_weight_B (BRAM, split by parity)
- `ram_weight_A`: SMALLS (W_DW, W_A) + even-group PP weights (W_INPROJ_X/Z, W_XPROJ, W_OUTPROJ evens)
- `ram_weight_B`: odd-group PP weights only (no SMALLS)
- Each: 57 BRAM36, TDP mode (1W + 2R same-cycle for MAC2 dual-weight fetch)

### ram_const (LUTRAM)
128 x 256b, distributed LUTRAM (1024 LUTRAM cells + 512 FF).
Dual read: c1 via port_b, c2 via port_a (mux with DMA writes at preload).

### H_RegFile (URAM)
- `u_mc/HAS_H_GEN.u_hrf`: 16 URAM (H depth 16K x 256b)
- `u_mc2/HAS_H_GEN.u_hrf`: 16 URAM
- Independent H state per cluster; SSM state persists across tokens.

## Post-Route Numbers

### Timing @ 100 MHz
| Metric | Value |
|---|---|
| WNS | +0.566 ns |
| TNS | 0.000 (all 33,007 endpoints met) |
| WHS | +0.042 ns |
| Fmax cap | ~106 MHz |

### Resources (KV260)
| Resource | Used | Total | Util% |
|---|---|---|---|
| CLB LUT | 26,226 | 117,120 | 22% |
| CLB Register | 11,624 | 234,240 | 5% |
| BRAM36 | 121 | 144 | 84% |
| URAM | 48 | 64 | 75% |
| DSP48E2 | 99 | 1,248 | 8% |
| CARRY8 | 944 | 14,640 | 6% |

### Power
| Metric | Value |
|---|---|
| Total | 0.688 W |
| Dynamic | 0.394 W |
| Static | 0.294 W |
| Junction Temp | 26.6 C |

## Per-Token Performance (Mamba2-130M, 1 layer, T=3 tokens sim)

### Compute cycles
| Stage | Cyc/token | Formula |
|---|---|---|
| H_INIT | 12,289 | Once per model init (amortized over N tokens) |
| RN | 294 | 48-iter SQ + 48-iter AP |
| M1A | 18,625 | 48 grp/cluster x 388 cyc (PP MAC2) |
| M1B | 18,625 | same as M1A |
| M2 | 865 | 96 grp x 9 cyc conv+bias |
| M3 | 98 | 96 grp SiLU stream |
| M4 | 6,949 | 9 grp/cluster x 772 cyc (PP MAC2) |
| M6 | **70,514** | 48 grp x 1469 cyc + 7 gate |
| M7 | 0 | Merged into c1 M6 chain |
| M8 | 18,529 | 24 grp/cluster x 772 cyc (PP MAC2) |
| **Total** | **134,500** | Excl H_INIT |

### Decode latency (24 layers)
| Mode | Latency/token | Throughput |
|---|---|---|
| Compute only | 32.3 ms | 31 tok/s |
| + DMA serial (235k cyc/layer) | 88.8 ms | **11.3 tok/s** |
| + DMA prefetch (future) | 56.5 ms | 17.7 tok/s |

## Optimization History

| Phase | M6 cyc | Total cyc | tok/s (24L+DMA) | Key change |
|---|---|---|---|---|
| Phase 0 (single cluster) | ~242k | ~369k | ~7.0 | Baseline |
| Phase 3 (FIFO substep 2x16) | 197k | 273k | 8.2 | Cluster2 as Y_MAC aux |
| Phase 6.1 (dual-port + slot 2 merge) | 102.7k | 166.8k | 10.0 | Scrap mutex, mirror ram_main |
| Phase 6.2b (Y_MAC prefetch) | 75.9k | 139.9k | 10.75 | Overlap Y_MAC with next LOAD |
| **Phase 6.2c (fold LATCH_C)** | **70.5k** | **134.5k** | **11.3** | Latch C at DAB cycle |

## Key Design Decisions

### Why mirror ram_main instead of arbitration
URAM primitives don't support TDP RW patterns (write + read on same port). Original Phase 6 attempted this and fell back to BRAM36 (36 tiles), pushing total BRAM over KV260's 144 limit.
Mirror uses SDP mode on both URAM copies, keeps URAM inference, costs only 8 extra URAM tiles.

### Why dual LUT_Bank instead of arbitration
LUT_Bank is combinational (0-cyc latency). Sharing between clusters would require mux on input port + stall logic when both need output same cycle. Dup cost: ~50 LUT primitives + 2 BRAM (RSqrt duplicate). Simpler, no perf loss.

### Why prefetch only within-s (not across l-boundary)
L-boundary transition goes through 5-cyc DU tail before next (l+1, s=0). Prefetching would require holding B/C/A across DU tail without them getting clobbered. Complexity not worth the extra 3k cyc save.

### Why c2 write serialization (not mirror write bus)
Both mirrors need same write. Write mux at Mamba_Top level (c1 wins tie, c2 stalls at S_C2_WRITE). Cost: ~48 stall cycles/token (48 c1 writes). Negligible.

## Files

**RTL sources:**
- `common/Memory_System.v` - dual-port with mirrored ram_main
- `common/BRAM_256b.v` - unchanged TDP wrapper
- `common/LUT_Bank.v` - unchanged, instantiated 2x in top
- `common/_parameter.v` - Mamba2 dims + weight bank layout
- `mamba/M_Cluster.v` - HAS_H param for optional H_RegFile
- `mamba/Mamba_PE.v` - 16-lane PE with MAC2 mode
- `mamba/H_RegFile.v` - URAM-inferred H state storage
- `mamba/Reduce16.v` + `Reduce16Wide.v` - y-reduction adders
- `top/Mamba_Top.v` - main FSM (~1900 lines, Phase 6)

**Dead code (safe to delete):**
- `mamba/Handshake_FIFO.v` - Phase 3 FIFO, no longer instantiated

**Testbench:**
- `tb/tb_Mamba_Top_MAMBA2.v` - T_TEST=3 multi-token profile

**Scripts:**
- `run_synth_impl_mamba2.tcl` - full flow (synth + impl + reports)
- `run_synth_only.tcl` - synth only for fast BRAM diagnosis
