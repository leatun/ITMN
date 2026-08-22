`timescale 1ns / 1ps

// ============================================================================
// Handshake_FIFO — small synchronous FIFO for cluster1 → cluster2 M6 substep
// pipeline (see plan file fluttering-stirring-alpaca).
//
// Producer: cluster1 pushes h_new_word + (g, l, s) IDs at end of SSM update
//           per (l, s) iteration in M6.
// Consumer: cluster2 pops → Y_MAC(h_new · C[s]) + external accumulator per s;
//           after all s per lane, does D·u + YSUM; per group does M7 gate + write.
//
// Depth 2 is sufficient because cluster1's per-(l,s) throughput is ~10 cyc
// while cluster2's Y_MAC is ~3 cyc — cluster2 stays ahead of cluster1, FIFO
// rarely fills. Depth 2 buffers 1-cyc slip; expand to 4 if D·u/YSUM tail
// causes back-pressure (rare, once per lane).
//
// Ports:
//   push       : write strobe (assert with wdata valid)
//   pop        : read strobe (advance rd_ptr; rdata reflects head)
//   wdata      : data to push
//   rdata      : head data (valid when !empty)
//   full/empty : flow-control flags
// ============================================================================
module Handshake_FIFO #(
    parameter WIDTH = 271,
    parameter DEPTH = 2
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              push,
    input  wire [WIDTH-1:0]  wdata,
    input  wire              pop,
    output wire [WIDTH-1:0]  rdata,
    output wire              full,
    output wire              empty
);

    localparam PTR_W = (DEPTH <= 2) ? 1 : $clog2(DEPTH);
    localparam CNT_W = $clog2(DEPTH + 1);

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [PTR_W-1:0] wr_ptr, rd_ptr;
    reg [CNT_W-1:0] count;

    assign rdata = mem[rd_ptr];
    assign full  = (count == DEPTH);
    assign empty = (count == 0);

    wire do_push = push && !full;
    wire do_pop  = pop  && !empty;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            wr_ptr <= {PTR_W{1'b0}};
            rd_ptr <= {PTR_W{1'b0}};
            count  <= {CNT_W{1'b0}};
        end else begin
            if (do_push) begin
                mem[wr_ptr] <= wdata;
                wr_ptr      <= (wr_ptr == DEPTH - 1) ? {PTR_W{1'b0}} : wr_ptr + 1'b1;
            end
            if (do_pop) begin
                rd_ptr <= (rd_ptr == DEPTH - 1) ? {PTR_W{1'b0}} : rd_ptr + 1'b1;
            end
            case ({do_push, do_pop})
                2'b10:   count <= count + 1'b1;
                2'b01:   count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

    // synthesis translate_off
    always @(posedge clk) begin
        if (!rst && push && full && !pop)
            $display("[%0t] Handshake_FIFO WARN: push while full (dropped)", $time);
        if (!rst && pop && empty)
            $display("[%0t] Handshake_FIFO WARN: pop while empty", $time);
    end
    // synthesis translate_on

endmodule
