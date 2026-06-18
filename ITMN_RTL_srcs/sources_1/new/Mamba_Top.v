// ============================================================================
// Mamba_Top — D1 Mamba-standalone synth wrapper.
//
// Instantiates ITM_Top_v2 with same port signature. Intended for OOC synth
// run with `+define+MAMBA_ONLY` so the controller's P1/BR/FIN/CASCADE state
// arms compile out; Vivado opt_design dead-code-eliminates the unused logic.
// Reports give standalone Mamba-block resource (LUT/BRAM/URAM/DSP/Fmax) for
// the paper's fair-compare table.
//
// NOTE: same I/O ports as the full ITM_Top. The cascade/pool inputs and
// done_phase1/done_inception outputs are still present; they are simply not
// exercised in this build mode (FSM forces done_phase1=done_inception=1 at
// start and skips directly to S_NORM_M1A_SQ_READ).
// ============================================================================
module Mamba_Top (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output wire        done_phase1,
    output wire        done_inception,
    output wire        done_mamba,
    output wire        done_all,

    input  wire [9:0]  T_MAX,
    input  wire [3:0]  CH_IN,
    input  wire [3:0]  CH_OUT,
    input  wire [3:0]  CH_M,
    input  wire [3:0]  DT_RANK,

    input  wire        need_pool,
    input  wire        cascade_mode,

    input  wire        dma_write_en,
    input  wire [1:0]  dma_target,
    input  wire [14:0] dma_addr,
    input  wire [255:0] dma_wdata,
    output wire        dma_ready,
    input  wire        dma_read_en,
    input  wire [1:0]  dma_rtarget,
    input  wire [14:0] dma_raddr,
    output wire [255:0] dma_rdata
);

    ITM_Top_v2 u_core (
        .clk(clk), .rst(rst), .start(start),
        .done_phase1(done_phase1), .done_inception(done_inception),
        .done_mamba(done_mamba),   .done_all(done_all),
        .T_MAX(T_MAX), .CH_IN(CH_IN), .CH_OUT(CH_OUT),
        .CH_M(CH_M),   .DT_RANK(DT_RANK),
        .need_pool(need_pool), .cascade_mode(cascade_mode),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr),         .dma_wdata(dma_wdata),
        .dma_ready(dma_ready),
        .dma_read_en(dma_read_en),   .dma_rtarget(dma_rtarget),
        .dma_raddr(dma_raddr),       .dma_rdata(dma_rdata)
    );

endmodule
