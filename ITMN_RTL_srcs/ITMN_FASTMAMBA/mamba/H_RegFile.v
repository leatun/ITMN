`include "_parameter.v"

// ============================================================================
// H_RegFile — on-chip h_state storage for Mamba SSM scan (16-lane cluster).
//
// Layout:
//   - 1 word = 16 lanes × 16-bit = 256-bit
//   - Each word holds h[c, s=0..15] for one channel c.
//   - Addressing convention (FSM_M drives):
//        addr = c       (channel index, 0..d_inner-1)
//   - With d_state=16 fitting one word, no state-pass folding needed.
//
// Capacity (per block):
//   B0..B3: d_inner=128 → 128 words   (32 Kbit)
//   B4    : d_inner=256 → 256 words   (64 Kbit)
//
// → Default DEPTH=256 covers all blocks. Width=256-bit matches Memory_System
//   word size, so dump/restore via DMA reuses the same datapath.
//   Vivado infers BRAM18K cascade (4 × 18Kb = 72-bit, so 4× wide for 256-bit
//   ≈ 4 BRAM_18K per row × 1 row for B0–B3, × 2 row for B4 → ~4–8 BRAM18K).
//
// Ports:
//   - 1 write port (sync, 1-cycle write)
//   - 1 read  port (sync, 1-cycle latency: rd_addr on cycle N → rd_data on N+1)
//
// Reset clears rd_data register only; memory contents undefined post-reset
// (FSM_M initialises h_state to 0 via explicit zero writes before SSM scan).
// ============================================================================
module H_RegFile #(
    parameter ADDR_W = `H_ADDR_W,   // default from _parameter.v
    parameter DEPTH  = `H_DEPTH     // default from _parameter.v
) (
    input  wire                clk,
    input  wire                rst,

    // Write port
    input  wire                wr_en,
    input  wire [ADDR_W-1:0]   wr_addr,
    input  wire [16*`DATA_W-1:0] wr_data,

    // Read port (1-cycle latency)
    input  wire [ADDR_W-1:0]   rd_addr,
    output reg  [16*`DATA_W-1:0] rd_data
);

    // Mamba2: DEPTH=16384 → 4 Mb. Force URAM cascade (KV260 has 64× URAM288).
    // BRAM would need 32 tiles vs URAM ~16 tiles.
    (* ram_style = "ultra" *)
    reg [16*`DATA_W-1:0] mem [0:DEPTH-1];

    // Synchronous write
    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
    end

    // Synchronous read (1-cycle latency)
    always @(posedge clk or posedge rst) begin
        if (rst) rd_data <= {16*`DATA_W{1'b0}};
        else     rd_data <= mem[rd_addr];
    end

endmodule
