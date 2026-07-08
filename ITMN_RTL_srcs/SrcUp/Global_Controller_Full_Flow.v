`include "_parameter.v"

module Global_Controller_Full_Flow
(
    input clk,
    input reset,

    // System
    input  start_system,
    output reg done_system,

    // Master Memory
    output reg [14:0] core_read_addr,
    input  [255:0] core_read_data, 

    output reg [14:0] weight_read_addr,
    input  [255:0] weight_read_data,

    output reg [14:0] const_read_addr,
    input  [255:0] const_read_data,

    output reg core_write_en,
    output reg [14:0] core_write_addr,
    output reg [255:0] core_write_data,
    
    output reg bank_sel,
    // 0: A -> Read , B -> Write 
    // 1: B -> Read , A -> Write 

    // Mamba Top
    output reg [2:0] mode_select,
    
    // Linear Interface
    output reg lin_start,
    output reg [15:0] lin_len,
    input  lin_done,
    output reg signed [`DATA_WIDTH-1:0] lin_x_val,
    output reg signed [255:0] lin_W_vals,
    output reg [16 * `DATA_WIDTH - 1 : 0] lin_bias_vals,
    output reg lin_en,
    input signed [255:0] lin_y_out_in, 
    
    // Conv1D Interface
    output reg conv_start, 
    output reg conv_valid_in,
    output reg conv_en,
    input      conv_ready_in, 
    input      conv_valid_out,
    output reg signed [255:0] conv_x_vec,      
    output reg signed [1023:0] conv_w_vec,     
    output reg signed [255:0] conv_b_vec,      
    input signed [255:0] conv_y_vec,
    
    // Scan Interface
    output reg scan_start,
    output reg scan_en,
    output reg scan_clear_h,
    input      scan_done,
    output reg signed [`DATA_WIDTH-1:0] scan_delta_val, scan_x_val, scan_D_val, scan_gate_val,
    output reg signed [255:0] scan_A_vec, scan_B_vec, scan_C_vec,
    input signed [`DATA_WIDTH-1:0] scan_y_out,
    
    // Softplus Interface
    output reg signed [`DATA_WIDTH-1:0] softplus_in,
    input  signed [`DATA_WIDTH-1:0] softplus_out
);

    // --- FSM STATES ---
    localparam S_IDLE         = 0;
    
    // PHASE 1: LINEAR PROJECTIONS 1 2
    localparam S_PHASE1_SETUP = 1;
    localparam S_LIN_SETUP    = 2; 
    localparam S_LIN_READ_XW  = 3; 
    localparam S_LIN_RAM_WAIT = 4;
    localparam S_LIN_PRE_FEED = 5;
    localparam S_LIN_FEED_X   = 6;
    localparam S_LIN_WAIT     = 7;
    localparam S_LIN_WRITE    = 8;
    
    // PHASE 2: CONV1D
    localparam S_CONV_SETUP        = 10;
    localparam S_CONV_LOAD_WEIGHT  = 11;
    localparam S_CONV_WAIT_WEIGHT  = 12;
    localparam S_CONV_LATCH_WEIGHT = 13;
    localparam S_CONV_READ_X       = 14;
    localparam S_CONV_WAIT_X       = 15;
    localparam S_CONV_PRE_FEED     = 16;
    localparam S_CONV_FEED         = 17;
    localparam S_CONV_EXEC_WAIT    = 18;
    localparam S_CONV_WRITE_MEM    = 19;
    
    // PHASE 3.1: X_PROJ (Calc B, C, dt_raw)
    localparam S_PHASE3_SETUP   = 30;
    localparam S_XPROJ_SETUP    = 31;
    localparam S_XPROJ_READ     = 32;
    localparam S_XPROJ_WAIT_RAM = 33;
    localparam S_XPROJ_FEED     = 34;
    localparam S_XPROJ_WAIT_LIN = 35;
    localparam S_XPROJ_WRITE    = 36;
    
    // PHASE 3.2: DT_PROJ (Read dt_raw -> Linear -> Softplus -> Delta)
    localparam S_DT_READ_INPUT     = 39;
    localparam S_DT_WAIT_INPUT_RAM = 38;
    
    localparam S_DT_SETUP          = 40;
    localparam S_DT_WAIT_RAM       = 41;
    localparam S_DT_FEED           = 42;
    localparam S_DT_WAIT_LIN       = 43;
    localparam S_DT_SOFTPLUS       = 44;
    localparam S_DT_LATCH_SOFTPLUS = 45;
    localparam S_DT_WRITE          = 46;
    localparam S_DT_CHECK_LOOP     = 47;
    localparam S_DT_WAIT_SOFTPLUS  = 49; 
    
    // PHASE 4: SCAN
    localparam S_SCAN_SETUP        = 50; 
    localparam S_SCAN_LOAD_STATIC  = 51; 
    
    localparam S_SCAN_LOAD_DYN_1   = 54;    // Delta
    localparam S_SCAN_LOAD_DYN_2   = 55;    // X
    localparam S_SCAN_LOAD_DYN_3   = 56;    // Gate
    localparam S_SCAN_LOAD_SHARED_1 = 57;   // B
    localparam S_SCAN_LOAD_SHARED_2 = 58;   // C
    
    localparam S_SCAN_RUN          = 59; 
    localparam S_SCAN_WAIT         = 60; 
    localparam S_SCAN_WRITE        = 61;
    
    localparam S_SCAN_RAM_WAIT     = 65;
    
    // Phase 5: LINEAR OUTPROJECTION
    localparam S_PHASE5_SETUP   = 80;
    localparam S_OUTPROJ_SETUP  = 81;
    localparam S_OUTPROJ_READ   = 82;
    localparam S_OUTPROJ_WAIT   = 83;
    localparam S_OUTPROJ_FEED   = 84;
    localparam S_OUTPROJ_WAIT_L = 85;
    localparam S_OUTPROJ_WRITE  = 86;
    
    // DEBUG ---
    localparam S_DEBUG_SETUP  = 90;
    localparam S_DEBUG_READ   = 91;
    localparam S_DEBUG_FEED   = 92;
    localparam S_DEBUG_WAIT   = 93;
    localparam S_DEBUG_WRITE  = 94;

    localparam ADDR_DEBUG_IN  = 15'd0;     // RAM A 
    localparam ADDR_DEBUG_OUT = 15'd20000; // RAM B
    
    localparam S_DONE           = 99;

    reg [6:0] state;
    reg [6:0] next_state_after_wait;
    
    // Internal Counters 
    reg [3:0]  chunk_cnt;        
    reg [15:0] token_cnt;
    
    reg [15:0] feed_x_idx;       
    reg [14:0] base_weight_addr;  
    
    reg [16 * `DATA_WIDTH - 1 : 0] x_cache; 
    reg [3:0] x_cache_idx;
    
    reg lin_done_flag;
    reg lin_job_sel; 
    
    reg [2:0] w_load_cnt;
    reg [1023:0] w_conv_cache;
    
    reg [15:0] stride_addr;         
    
    reg [2:0] dt_idx;      
    reg [4:0] delta_buf_idx; 
    reg signed [16*`DATA_WIDTH-1:0] delta_buffer; 
    
    // SCAN REGS
    reg [7:0] scan_ch_cnt;
    reg [3:0] scan_out_idx;
    reg [255:0] scan_out_buffer;
    reg [3:0] sub_idx;
    
    // --- MEMORY MAP ---
    localparam ADDR_X_INPUT     = 15'd0;         // A[0..3999]
    localparam ADDR_X_PRIM      = 15'd16384;     // A[16384..24383]
    localparam ADDR_CONV_OUT    = 15'd24576;     // A[24576..32575] 
    localparam ADDR_SCAN_Y_BASE = 15'd8192;      // 8192..16191 
    
    
    // RAM B (Target 1)
    localparam ADDR_B_BASE      = 15'd0;         // 0..999
    localparam ADDR_C_BASE      = 15'd1500;      // 1500..2499
    localparam ADDR_DT_RAW_BASE = 15'd3000;      // 3000..3999
    localparam ADDR_GATE        = 15'd8192;      // 8192..16191
    localparam ADDR_DELTA_BASE  = 15'd17000;     // 17000..24999
    localparam ADDR_FINAL_OUT   = 15'd20000;    

    // Weight RAM
    localparam W_BASE_INPROJ1 = 15'd0;
    localparam W_BASE_INPROJ2 = 15'd512;
    localparam W_BASE_CONV    = 15'd1024;
    localparam W_BASE_XPROJ   = 15'd1536;
    localparam W_BASE_DTPROJ  = 15'd1920;
    localparam W_BASE_OUTPROJ = 15'd2432;       // 1920 + 512 (DT_Proj size)
    
    // Const RAM
    localparam CONST_CONV_BIAS    = 15'd0;       // 0..7
    localparam CONST_DT_BIAS_BASE = 15'd128;     // 128..135
    localparam ADDR_A_BASE        = 15'd1024;    // 1024..1151 (A)
    localparam ADDR_D_BASE        = 15'd1152;    // 1152..1159 (D)


    // ============================================================
    // MAIN FSM
    // ============================================================
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= S_IDLE;
            done_system <= 0;
            bank_sel <= 0;

            core_read_addr <= 0; 
            weight_read_addr <= 0; 
            const_read_addr <= 0;

            core_write_en <= 0; 
            core_write_addr <= 0; 
            core_write_data <= 0;

            mode_select <= 0;

            lin_start <= 0; 
            lin_len <= 0; 
            lin_bias_vals <= 0; 
            lin_en <= 0;

            conv_start <= 0; 
            conv_en <= 0; 
            conv_valid_in <= 0;

            chunk_cnt <= 0; 
            token_cnt <= 0; 
            feed_x_idx <= 0;

            base_weight_addr <= 0; 
            x_cache <= 0; 
            x_cache_idx <= 0;

            lin_done_flag <= 0; 
            lin_job_sel <= 0; 
            w_load_cnt <= 0;

            scan_clear_h <= 0; 
            scan_en <= 0; 
            scan_start <= 0;
        end else begin
            
            if (lin_done) lin_done_flag <= 1;
            else if (state == S_LIN_WRITE || state == S_XPROJ_WRITE || state == S_IDLE) lin_done_flag <= 0;

            case (state)
                S_IDLE: begin
                    done_system <= 0;
                    if (start_system) begin
                        //state <= S_PHASE3_SETUP; 
                        //state <= S_PHASE1_SETUP; 
                        //state <= S_SCAN_SETUP;
                        //state <= S_PHASE5_SETUP;
                        state <= S_DEBUG_SETUP;
                    end
                end

                // ============================================================
                // PHASE 1: LINEAR PROJECTIONS (Token-First Processing)
                // Loop 1: Job 0 (Primary) -> Loop 2: Job 1 (Gate)
                // ============================================================
                S_PHASE1_SETUP: begin
                    mode_select <= 3'd1; // Linear
                    lin_job_sel <= 0;    // Job 0 (Primary)
                    //lin_job_sel <= 1;  //test
                    chunk_cnt <= 0;
                    token_cnt <= 0;
                    bank_sel <= 0;       // Input X RAM A
                    state <= S_LIN_SETUP;
                end

                S_LIN_SETUP: begin
                    lin_len <= 64; // Input D_MODEL = 64
                    lin_start <= 1; 
                    
                    feed_x_idx <= 0; 
                    x_cache_idx <= 0;
                    
                    // Tinh Dia Chi Input X
                    // Input X luu token-first, moi token 64 phan tu (4 dong RAM)
                    // Addr = Base + (Token * 4)
                    core_read_addr <= ADDR_X_INPUT + (token_cnt * 4);
                    
                    // Tinh Dia Chi Weight
                    // Job 0: W_BASE_INPROJ1 + (Chunk * 64 lines)
                    // Job 1: W_BASE_INPROJ2 + (Chunk * 64 lines)
                    if (lin_job_sel == 0)
                        base_weight_addr <= W_BASE_INPROJ1 + (chunk_cnt * 64);
                    else
                        base_weight_addr <= W_BASE_INPROJ2 + (chunk_cnt * 64);
                        
                    weight_read_addr <= (lin_job_sel == 0) ? (W_BASE_INPROJ1 + (chunk_cnt * 64)) 
                                                           : (W_BASE_INPROJ2 + (chunk_cnt * 64));

                    lin_en <= 0; 
                    state <= S_LIN_READ_XW;
                end
                
                S_LIN_READ_XW: begin
                    lin_start <= 0; 
                    state <= S_LIN_RAM_WAIT;
                end
                
                S_LIN_RAM_WAIT: state <= S_LIN_PRE_FEED;
                
                S_LIN_PRE_FEED: begin
                    x_cache <= core_read_data; 
                    lin_en <= 1; 
                    // Chuan bi weight cho nhip tiep theo (nhip dau tien la weight_addr + 1)
                    weight_read_addr <= base_weight_addr + feed_x_idx + 1;
                    state <= S_LIN_FEED_X;
                end

                S_LIN_FEED_X: begin                  
                    x_cache_idx <= x_cache_idx + 1; 
                    
                    if (x_cache_idx == 15) begin 
                        lin_en <= 0; // Pause de load cache moi
                        core_read_addr <= core_read_addr + 1; 
                        x_cache_idx <= 0; 
                        state <= S_LIN_READ_XW; 
                        
                    end else if (feed_x_idx == 64) begin
                        lin_en <= 0; 
                        state <= S_LIN_WAIT;
                        feed_x_idx <= 0;
                        x_cache_idx <= 0;
                        
                    end else begin
                        weight_read_addr <= weight_read_addr + 1;
                    end
                    
                    if (state == S_LIN_FEED_X) feed_x_idx <= feed_x_idx + 1; 
                end

                S_LIN_WAIT: begin
                    if (lin_done || lin_done_flag) begin
                        // GHI KET QUA
                        core_write_en <= 1;
                        
                        // Transpose Logic: Ghi theo Channel-First
                        // Addr = Base + (Chunk * 1000) + Token
                        if (lin_job_sel == 0) begin
                            // Job 0: Ghi vao vung X_PRIM (RAM A)
                            // Memory System bank_sel 1 ghi A
                            bank_sel <= 1;
                            core_write_addr <= ADDR_X_PRIM + (chunk_cnt * 1000) + token_cnt;
                        end else begin
                            // Job 1 (Gate): Ghi vao RAM B
                            bank_sel <= 0; 
                            core_write_addr <= ADDR_GATE + (chunk_cnt * 1000) + token_cnt;
                        end
                        
                        core_write_data <= lin_y_out_in;
                        state <= S_LIN_WRITE;
                    end
                end

                S_LIN_WRITE: begin
                    core_write_en <= 0;
                    
                    bank_sel <= 0; 
                    
                    if (chunk_cnt == 7) begin
                        chunk_cnt <= 0;
                        
                        if (token_cnt == 999) begin
                            // Xong het 1000 token cho Job hien tai
                            token_cnt <= 0;
                            
                            if (lin_job_sel == 0) begin
                                lin_job_sel <= 1; // Job 1 (Gate)
                                state <= S_LIN_SETUP;
                            end else begin
                                // Job 1 -> Sang Conv
                                state <= S_CONV_SETUP;
                            end
                        end else begin
                            token_cnt <= token_cnt + 1;
                            state <= S_LIN_SETUP;
                        end
                        
                    end else begin
                        chunk_cnt <= chunk_cnt + 1;
                        state <= S_LIN_SETUP;
                    end
                end

                // ============================================================
                // PHASE 2: CONV1D
                // Input: ADDR_X_PRIM 
                // Output: ADDR_CONV_OUT
                // ============================================================
                S_CONV_SETUP: begin
                    mode_select <= 3'd2; // Mode Conv
                    chunk_cnt <= 0;
                    token_cnt <= 0;
                    conv_start <= 1;
                    conv_en    <= 0;
                    
                    // Base Weight Conv
                    base_weight_addr <= W_BASE_CONV;
                    
                    // Bias (Const)
                    const_read_addr <= 0; 
                    
                    // Set Bank Sel = 0 (Doc RAM A - X_PRIM)
                    // Luu y: Memory System phai hieu la doc ADDR_X_PRIM
                    bank_sel <= 0; 
                    
                    state <= S_CONV_LOAD_WEIGHT;
                    w_load_cnt <= 0;
                end
    
                S_CONV_LOAD_WEIGHT: begin
                    conv_start <= 0;
                    weight_read_addr <= base_weight_addr + w_load_cnt; 
                    state <= S_CONV_WAIT_WEIGHT;
                end
                
                S_CONV_WAIT_WEIGHT: state <= S_CONV_LATCH_WEIGHT;
                
                S_CONV_LATCH_WEIGHT: begin
                    w_conv_cache[w_load_cnt*256 +: 256] <= weight_read_data;
                    if (w_load_cnt == 3) begin
                        w_load_cnt <= 0;
                        conv_b_vec <= const_read_data; 
                        state <= S_CONV_READ_X;
                    end else begin
                        w_load_cnt <= w_load_cnt + 1;
                        state <= S_CONV_LOAD_WEIGHT; 
                    end
                end
                
                S_CONV_READ_X: begin
                    // Doc tu ADDR_X_PRIM (Channel-First)
                    // Addr = ADDR_X_PRIM + (Chunk * 1000) + Token
                    core_read_addr <= ADDR_X_PRIM + (chunk_cnt * 1000) + token_cnt;
                    
                    conv_en <= 0; 
                    state <= S_CONV_WAIT_X; 
                end
                
                S_CONV_WAIT_X: state <= S_CONV_PRE_FEED;
                
                S_CONV_PRE_FEED: begin
                    x_cache <= core_read_data; 
                    conv_en <= 1; 
                    state <= S_CONV_FEED;
                end
                
                S_CONV_FEED: begin
                    conv_x_vec <= core_read_data;
                    conv_valid_in <= 1;
                    if (conv_ready_in) state <= S_CONV_EXEC_WAIT;
                end
                
                S_CONV_EXEC_WAIT: begin
                    conv_valid_in <= 0;
                    if (conv_valid_out) begin
                        core_write_en <= 1;
                        // Ghi ket qua Conv ra vung nho khac (ADDR_CONV_OUT)
                        // Van giu cau truc Channel-First
                        bank_sel <= 1; 
                        core_write_addr <= ADDR_CONV_OUT + (chunk_cnt * 1000) + token_cnt;
                        core_write_data <= conv_y_vec;
                        state <= S_CONV_WRITE_MEM;
                    end
                end
                
                S_CONV_WRITE_MEM: begin
                    core_write_en <= 0;
                    token_cnt <= token_cnt + 1;
                    bank_sel <= 0;
                    
                    if (token_cnt == 999) begin
                        token_cnt <= 0;
                        chunk_cnt <= chunk_cnt + 1;
                        
                        if (chunk_cnt == 7) begin 
                            state <= S_PHASE3_SETUP; 
                            //state <= S_DONE; // DEBUG CHECK
                        end else begin
                             // Sang Chunk moi -> Load Weight moi
                             base_weight_addr <= base_weight_addr + 4; 
                             const_read_addr <= chunk_cnt + 1;
                             state <= S_CONV_LOAD_WEIGHT;
                             conv_start <= 1; 
                        end
                    end else begin
                        state <= S_CONV_READ_X;
                    end
                end
                
                // ============================================================
                // PHASE 3.1: X_PROJECTION (Calculate B, C, dt_raw)
                // ============================================================
                S_PHASE3_SETUP: begin
                    token_cnt <= 0;
                    chunk_cnt <= 0; 
                    state <= S_XPROJ_SETUP;
                end

                S_XPROJ_SETUP: begin
                    mode_select <= 3'd1; // Linear
                    lin_len <= 128;      // Input length = 128
                    lin_start <= 1;
                    
                    feed_x_idx <= 0;
                    x_cache_idx <= 0;
                    
                    // Input X_Proj �?c t? Output Conv (ADDR_CONV_OUT)
                    // D? li?u Conv l�u Channel-First. Linear c?n Token-First.
                    // Token i n?m ?: Base+i, Base+1000+i, Base+2000+i...
                    stride_addr <= ADDR_CONV_OUT + token_cnt; 
                    core_read_addr <= ADDR_CONV_OUT + token_cnt;
                    
                    // Weight X_Proj
                    base_weight_addr <= W_BASE_XPROJ + (chunk_cnt * 128);
                    weight_read_addr <= W_BASE_XPROJ + (chunk_cnt * 128);
                    
                    bank_sel <= 0; // �?c RAM A
                    lin_en <= 0;
                    
                    state <= S_XPROJ_READ;
                end
                
                S_XPROJ_READ: begin
                    lin_start <= 0;
                    state <= S_XPROJ_WAIT_RAM;
                end
                
                S_XPROJ_WAIT_RAM: state <= S_XPROJ_FEED;
                
                S_XPROJ_FEED: begin
                    // 1. N?P CACHE (Khi EN = 0)
                    if (lin_en == 0) begin 
                        x_cache <= core_read_data; 
                        lin_en <= 1;
                        
                        // Nh?y c�c 1000 �?a ch? �? l?y 16 k�nh ti?p theo c?a c�ng Token
                        stride_addr <= stride_addr + 1000;
                        core_read_addr <= stride_addr + 1000; 
                        
                        weight_read_addr <= weight_read_addr + 1;
                    end
                    
                    // 2. FEED LINEAR (Khi EN = 1)
                    else begin 
                        x_cache_idx <= x_cache_idx + 1;
                        feed_x_idx <= feed_x_idx + 1;
                        
                        if (feed_x_idx == 127) begin 
                            lin_en <= 0; 
                            state <= S_XPROJ_WAIT_LIN;
                            feed_x_idx <= 0; 
                            x_cache_idx <= 0;
                        end 
                        else if (x_cache_idx == 15) begin
                            lin_en <= 0; // Pause �?c RAM
                            x_cache_idx <= 0;
                            state <= S_XPROJ_READ; 
                        end 
                        else begin
                            weight_read_addr <= base_weight_addr + feed_x_idx + 2;
                        end
                    end
                    
                end
                
                S_XPROJ_WAIT_LIN: begin
                    if (lin_done || lin_done_flag) begin
                        state <= S_XPROJ_WRITE;
                    end
                end
                
                S_XPROJ_WRITE: begin
                    core_write_en <= 1;
                    bank_sel <= 0; 
                    
                    // --- GHI K?T QU? PHASE 3.1 ---
                    if (chunk_cnt == 0) begin
                        core_write_addr <= ADDR_B_BASE + token_cnt; // B
                        core_write_data <= lin_y_out_in;
                    end 
                    else if (chunk_cnt == 1) begin
                        core_write_addr <= ADDR_C_BASE + token_cnt; // C
                        core_write_data <= lin_y_out_in;
                    end 
                    else begin 
                        core_write_addr <= ADDR_DT_RAW_BASE + token_cnt; // dt_raw
                        // Zero padding logic (Gi? 4 s? �?u, �p 0 ph?n �u�i)
                        core_write_data <= {192'd0, lin_y_out_in[63:0]};
                    end
                    
                    // --- �I?U KHI?N LU?NG (QUAN TR?NG) ---
                    if (chunk_cnt == 2) begin 
                        chunk_cnt <= 0;
                        if (token_cnt == 999) begin
                            // Xong 1000 Token c?a Phase 3.1
                            // Chuy?n sang Phase 3.2 (Reset token_cnt �? ch?y l?i t? �?u)
                            token_cnt <= 0;
                            state <= S_DT_READ_INPUT; // <--- CHUY?N QUA PHASE 3.2 T?I ��Y
                        end else begin
                            token_cnt <= token_cnt + 1;
                            state <= S_XPROJ_SETUP; // Token ti?p theo c?a 3.1
                        end
                    end else begin
                        chunk_cnt <= chunk_cnt + 1;
                        state <= S_XPROJ_SETUP; 
                    end
                end
                
                // ============================================================
                // PHASE 3.2: DT_PROJECTION (Token-First Processing)
                // �?c dt_raw t? RAM -> Linear -> Softplus -> Delta -> RAM
                // ============================================================
                
                // --- B�?C 1: �?C DT_RAW T? RAM (Ch? l�m 1 l?n m?i Token) ---
                S_DT_READ_INPUT: begin
                    mode_select <= 3'd1; // Linear Mode
                    // �?c t? �?a ch? �? l�u ? Phase 3.1
                    core_read_addr <= ADDR_DT_RAW_BASE + token_cnt;
                    bank_sel <= 1;  //nay luu B, doc thi doi lai A
                    state <= S_DT_WAIT_INPUT_RAM;
                end
                
                S_DT_WAIT_INPUT_RAM: begin
                    // Ch? RAM tr? d? li?u 
                    // (Cycle sau data s? c� s?n ? port core_read_data)
                    state <= S_DT_SETUP;
                    chunk_cnt <= 0; // Reset chunk cho 8 chunks delta
                end

                // --- B�?C 2: SETUP & FEED LINEAR ---
                S_DT_SETUP: begin
                    // Latch data t? RAM v�o Cache ? chunk �?u ti�n
                    if (chunk_cnt == 0) begin
                        x_cache <= core_read_data; 
                    end
                    // V?i c�c chunk sau, x_cache v?n gi? gi� tr? c? (dt_raw) -> ��ng logic!

                    lin_len <= 4; // Input length = 4 (dt_rank)
                    
                    base_weight_addr <= W_BASE_DTPROJ + (chunk_cnt * 4);
                    weight_read_addr <= W_BASE_DTPROJ + (chunk_cnt * 4);
                    
                    const_read_addr  <= CONST_DT_BIAS_BASE + chunk_cnt; 
                    
                    dt_idx <= 0;
                    lin_en <= 0;
                    lin_start <= 0;
                    
                    state <= S_DT_WAIT_RAM;
                end
                
                S_DT_WAIT_RAM: begin
//                    lin_bias_vals <= const_read_data; // Latch Bias
                    lin_start <= 1; 
                    state <= S_DT_FEED;
                end
                
                S_DT_FEED: begin
                    lin_start <= 0; 
                    
                    if (lin_en == 0) begin
                        lin_en <= 1;
                        weight_read_addr <= base_weight_addr + 1; 
                    end 
                    else begin
                        // Feed 4 gi� tr? t? x_cache (dt_raw)
                        if (dt_idx == 3) begin 
                            lin_en <= 0;
                            state <= S_DT_WAIT_LIN;
                            dt_idx <= 0;
                        end else begin
                            dt_idx <= dt_idx + 1;
                            weight_read_addr <= base_weight_addr + dt_idx + 2;
                        end
                    end
                end
                
                S_DT_WAIT_LIN: begin
                    if (lin_done || lin_done_flag) begin
                        // Xong Linear -> C� k?t qu? th� -> �?y qua Softplus
                        delta_buf_idx <= 0;
                        state <= S_DT_SOFTPLUS;
                    end
                end
                
                S_DT_SOFTPLUS: begin
                    // �?y t?ng s? (trong 16 s?) v�o Softplus
                    //softplus_in <= lin_y_out_in[delta_buf_idx*16 +: 16];
                    state <= S_DT_LATCH_SOFTPLUS;
                end
                
                S_DT_LATCH_SOFTPLUS: begin
                    // Nh?n k?t qu?
                    delta_buffer[delta_buf_idx*16 +: 16] <= softplus_out;
                    
                    if (delta_buf_idx == 15) begin
                        state <= S_DT_WRITE; // �? xong 16 s?
                    end else begin
                        delta_buf_idx <= delta_buf_idx + 1;
                        state <= S_DT_SOFTPLUS;
                    end
                end
                
                S_DT_WRITE: begin
                    core_write_en <= 1;
                    bank_sel <= 0; 
                    
                    // Ghi Delta xu?ng RAM
                    core_write_addr <= ADDR_DELTA_BASE + (token_cnt * 8) + chunk_cnt;
                    core_write_data <= delta_buffer;
                    
                    state <= S_DT_CHECK_LOOP;
                end
                
                S_DT_CHECK_LOOP: begin
                    core_write_en <= 0;
                    
                    if (chunk_cnt == 7) begin
                        // Xong 8 chunk delta (128 k�nh) cho token n�y
                        if (token_cnt == 999) begin
                            state <= S_SCAN_SETUP; // Xong to�n b? Phase 3.2
                            //state <= S_DONE;         // Test phase 1 2 3
                        end else begin
                            token_cnt <= token_cnt + 1;
                            state <= S_DT_READ_INPUT; // L�m ti?p token sau
                        end
                    end else begin
                        chunk_cnt <= chunk_cnt + 1;
                        state <= S_DT_SETUP; // Chunk delta ti?p theo
                    end
                end
                
            // ============================================================
            // PHASE 4: SCAN CORE EXECUTION
            // ============================================================
            
            S_SCAN_SETUP: begin
                mode_select <= 3'd3; // Scan Mode
                scan_ch_cnt <= 0;    // Channel 0 (0..127)
                token_cnt   <= 0;    // Token 0 (0..999)
                scan_en     <= 0;
                scan_out_idx <= 0;   // Index buffer output
                
                scan_clear_h <= 1; // Reset ngay t? �?u
                
                // Chu?n b? v�o loop
                state <= S_SCAN_LOAD_STATIC;
            end
            
            // --- B�?C 1: LOAD THAM S? T?NH C?A K�NH (A, D) ---
            // Ch? l�m 1 l?n m?i K�nh (Channel)
            S_SCAN_LOAD_STATIC: begin
                // 1. Load A (Vector 16) - T? Const RAM
                // A c� shape (128, 16) -> L�u ? 128 d?ng �?u Const RAM
                const_read_addr <= ADDR_A_BASE + scan_ch_cnt;
                
                // --- K�CH HO?T X�A H ---
                scan_clear_h <= 1; 
                
                // Ch? RAM tr? d? li?u
                state <= S_SCAN_RAM_WAIT;
                next_state_after_wait <= 70; // State ?o (xem case b�n d�?i)
            end
            
            // Logic nh?n d? li?u Static (Sau RAM WAIT)
            70: begin
                scan_A_vec <= const_read_data; // L?y A
                
                // 2. Load D (Scalar) - T? Const RAM
                // D l�u t? d?ng 128 tr? �i. Gi? s? D l�u packed 16 s?/d?ng.
                // Addr = Base + (Ch / 16)
                const_read_addr <= ADDR_D_BASE + (scan_ch_cnt / 16);
                
                state <= S_SCAN_RAM_WAIT;
                next_state_after_wait <= 71;
            end
            
            71: begin
                // B�c t�ch D (Scalar)
                sub_idx = scan_ch_cnt % 16;
                scan_D_val <= const_read_data[sub_idx*16 +: 16];
                
                // Xong Static -> V�o v?ng l?p Token
                token_cnt <= 0;
                state <= S_SCAN_LOAD_DYN_1;
            end

            // --- B�?C 2: LOAD D? LI?U �?NG (THEO TOKEN) ---
            
            // 2.1 Load Delta (RAM B - Token First Strided)
            S_SCAN_LOAD_DYN_1: begin
                bank_sel <= 1; // �?c RAM B
                // Addr = Base + (Token * 8) + Chunk
                core_read_addr <= ADDR_DELTA_BASE + (token_cnt << 3) + (scan_ch_cnt[7:4]);
                scan_clear_h <= 0; 
                state <= S_SCAN_RAM_WAIT;
                next_state_after_wait <= 72;
            end
            
            72: begin
                // B�c t�ch Delta (Scalar)
                // Trong 1 chunk (16 k�nh), k�nh hi?n t?i n?m ? v? tr� (ch % 16)
                sub_idx = scan_ch_cnt % 16;
                scan_delta_val <= core_read_data[sub_idx*16 +: 16];
                
                state <= S_SCAN_LOAD_DYN_2;
            end

            // 2.2 Load X (RAM A - Channel First)
            S_SCAN_LOAD_DYN_2: begin
                bank_sel <= 0; // �?c RAM A
                // X l�u Channel-First (t? Conv output)
                // Addr = Base + (Ch * 63) + (Token / 16)
                
                //core_read_addr <= ADDR_CONV_OUT + (scan_ch_cnt * 63) + (token_cnt[15:4]);
                
                core_read_addr <= ADDR_CONV_OUT + ({8'd0, scan_ch_cnt[7:4]} * 1000) + token_cnt;
                
                state <= S_SCAN_RAM_WAIT;
                next_state_after_wait <= 73;
            end
            
            73: begin
                // B�c t�ch X (Scalar)
                sub_idx = scan_ch_cnt % 16;
                scan_x_val <= core_read_data[sub_idx*16 +: 16];
                
                state <= S_SCAN_LOAD_DYN_3;
            end
            
            // 2.3 Load Gate (RAM B - Channel First)
            S_SCAN_LOAD_DYN_3: begin
                bank_sel <= 1; // �?c RAM B
                //core_read_addr <= ADDR_GATE  + (scan_ch_cnt * 63) + (token_cnt[15:4]);
                
                //core_read_addr <= ADDR_GATE + (token_cnt << 3) + {11'd0, scan_ch_cnt[7:4]};
                core_read_addr <= ADDR_CONV_OUT + ({8'd0, scan_ch_cnt[7:4]} * 1000) + token_cnt;
                
                state <= S_SCAN_RAM_WAIT;
                next_state_after_wait <= 74;
            end
            
            74: begin
                //sub_idx = token_cnt % 16;
                sub_idx = scan_ch_cnt % 16;
                scan_gate_val <= core_read_data[sub_idx*16 +: 16];
                
                // Chuy?n v? RAM A �? �?c B, C (Shared)
                bank_sel <= 0;
                state <= S_SCAN_LOAD_SHARED_1;
            end

            // 2.4 Load B (RAM B - Token First Linear)
            S_SCAN_LOAD_SHARED_1: begin
                bank_sel <= 1; // V?n RAM B
                // Addr = Base + Token
                core_read_addr <= ADDR_B_BASE + token_cnt; 
                state <= S_SCAN_RAM_WAIT;
                next_state_after_wait <= 75;
            end
            
            75: begin
                scan_B_vec <= core_read_data; // B l� vector 256-bit
                state <= S_SCAN_LOAD_SHARED_2;
            end
            
            // 2.5 Load C (RAM B - Token First Linear)
            S_SCAN_LOAD_SHARED_2: begin
                // Addr = Base + Token
                core_read_addr <= ADDR_C_BASE + token_cnt;
                state <= S_SCAN_RAM_WAIT;
                next_state_after_wait <= 76;
            end
            
            76: begin
                scan_C_vec <= core_read_data; // C l� vector 256-bit
                
                // �? nguy�n li?u -> CH?Y!
                state <= S_SCAN_RUN;
            end

            // --- B�?C 3: CH?Y SCAN CORE ---
            S_SCAN_RUN: begin
                scan_en <= 1;    // B?t Enable
                scan_start <= 1; // Pulse Start
                
                state <= S_SCAN_WAIT;
            end
            
            S_SCAN_WAIT: begin
                scan_start <= 0;
                
                if (scan_done) begin
                    scan_en <= 0; // T?t Enable
                    
                    // Gom k?t qu? v�o Buffer Output
                    scan_out_buffer[scan_out_idx*16 +: 16] <= scan_y_out;
                    scan_out_idx <= scan_out_idx + 1;
                    
                    // N?u Buffer �?y (�? 16 token) ho?c l� token cu?i c�ng -> Ghi RAM
                    if (scan_out_idx == 15 || token_cnt == 999) begin
                        state <= S_SCAN_WRITE;
                    end else begin
                        // Ch�a �?y -> L�m token ti?p theo
                        token_cnt <= token_cnt + 1;
                        state <= S_SCAN_LOAD_DYN_1; // Quay l?i load �?ng
                    end
                end
            end
            
            // --- B�?C 4: GHI K?T QU? ---
            S_SCAN_WRITE: begin
                // Ghi v�o RAM B (Target Output Scan)
                // Addr = Base + (Ch * 63) + (Token / 16)
                // L�u ?: token_cnt hi?n t?i l� token cu?i c?a batch 16
                core_write_en <= 1;
                bank_sel <= 1; // Ghi RAM A
                
                core_write_addr <= ADDR_SCAN_Y_BASE + (scan_ch_cnt * 63) + (token_cnt[15:4]);
                core_write_data <= scan_out_buffer;
                
                // Reset buffer index
                scan_out_idx <= 0;
                
                // Check xem h?t Token ch�a
                if (token_cnt == 999) begin
                    core_write_en <= 0; // T?t write ? cycle sau
                    
                    // H?t token cho k�nh n�y -> Sang k�nh kh�c
                    if (scan_ch_cnt == 127) begin
                        state <= S_DONE; // XONG TO�N B? (Phase 4 Done)
                    end else begin
                        scan_ch_cnt <= scan_ch_cnt + 1;
                        state <= S_SCAN_LOAD_STATIC; // Load A, D cho k�nh m?i
                    end
                end else begin
                    // Ch�a h?t token -> Ti?p t?c token sau
                    token_cnt <= token_cnt + 1;
                    state <= S_SCAN_LOAD_DYN_1;
                end
            end
            
            // State trung gian �? ch? RAM (d�ng chung)
            S_SCAN_RAM_WAIT: begin
                 if (core_write_en) core_write_en <= 0; // �?m b?o t?t write
                 state <= next_state_after_wait; // Nh?y v? ��ch �? �?nh
            end
            
            // ============================================================
            // PHASE 5: OUTPUT PROJECTION (Linear: 128 -> 64)
            // ============================================================
            
            S_PHASE5_SETUP: begin
                token_cnt <= 0;
                chunk_cnt <= 0; // Chunk ? ��y l� Output Group (64 output -> 4 chunks 16)
                state <= S_OUTPROJ_SETUP;
            end

            S_OUTPROJ_SETUP: begin
                mode_select <= 3'd1; // Linear Mode
                lin_len <= 128;      // Input length = 128 (D_INNER)
                lin_start <= 1;
                
                feed_x_idx <= 0;
                x_cache_idx <= 0;
                
                // 1. INPUT X (SCAN OUT):
                // Scan Out l�u Channel-First (Group-First).
                // Token hi?n t?i n?m r?i r�c ?: Base + (Group*1000) + Token
                // B?t �?u t? Group 0
                stride_addr <= ADDR_SCAN_Y_BASE + token_cnt; 
                core_read_addr <= ADDR_SCAN_Y_BASE + token_cnt;
                
                // 2. WEIGHT:
                // W_out shape (64, 128). Chia l�m 4 chunks output.
                // M?i chunk x? l? 16 h�ng.
                // Addr = Base + (Chunk * 128 d?ng input)
                base_weight_addr <= W_BASE_OUTPROJ + (chunk_cnt * 128);
                weight_read_addr <= W_BASE_OUTPROJ + (chunk_cnt * 128);
                
                bank_sel <= 0; // �?c RAM A (Scan Out)
                lin_en <= 0;
                
                state <= S_OUTPROJ_READ;
            end
            
            S_OUTPROJ_READ: begin
                lin_start <= 0;
                state <= S_OUTPROJ_WAIT;
            end
            
            S_OUTPROJ_WAIT: state <= S_OUTPROJ_FEED;
            
            S_OUTPROJ_FEED: begin
                // 1. N?P CACHE (Khi EN = 0)
                if (lin_en == 0) begin 
                    x_cache <= core_read_data; 
                    lin_en <= 1;
                    
                    // Logic Stride Read (Gi?ng Phase 3.1)
                    // Nh?y �?n Group ti?p theo: +1000
                    stride_addr <= stride_addr + 1000;
                    core_read_addr <= stride_addr + 1000; 
                    
                    // FIX: KH�NG t�ng �?a ch? weight ? ��y.
                    // W[0] �? ��?c load t? SETUP/WAIT, c?n gi? nguy�n cho nh?p �?u ti�n (x[0]).
                    weight_read_addr <= weight_read_addr + 1; // <-- X�A D?NG N�Y
                end
                
                // 2. FEED LINEAR (Khi EN = 1)
                else begin 
                    x_cache_idx <= x_cache_idx + 1;
                    feed_x_idx <= feed_x_idx + 1;
                    
                    if (feed_x_idx == 127) begin // �? 128 ph?n t?
                        lin_en <= 0; 
                        state <= S_OUTPROJ_WAIT_L;
                        feed_x_idx <= 0; 
                        x_cache_idx <= 0;
                    end 
                    else if (x_cache_idx == 15) begin // H?t 16 s? trong Cache -> Reload
                        lin_en <= 0; 
                        x_cache_idx <= 0;
                        state <= S_OUTPROJ_READ; // Quay l?i �?c Group ti?p theo
                        
                        // FIX: Khi quay l?i READ, �?a ch? Weight c?n t�ng l�n 1 (cho x ti?p theo)
                        // L�c n�y feed_x_idx �ang l� 15, 31...
                        // Next weight addr = Base + feed_x_idx + 1
                    end 
                    else begin
                        // FIX: Logic Pipeline
                        // T?i nh?p i (�ang t�nh x[i]*W[i]), ta c?n request W[i+1] cho nh?p sau.
                        // feed_x_idx hi?n t?i l� i.
                        // Addr = Base + i + 1.
                        weight_read_addr <= base_weight_addr + feed_x_idx + 2;
                    end
                end
            end
            
            S_OUTPROJ_WAIT_L: begin
                if (lin_done || lin_done_flag) begin
                    state <= S_OUTPROJ_WRITE;
                end
            end
            
            S_OUTPROJ_WRITE: begin
                // Ghi k?t qu? Final Output (64 channels)
                core_write_en <= 1;
                bank_sel <= 0; // Ghi RAM B (ADDR_FINAL_OUT)
                
                // Addr: channel f
                core_write_addr <= ADDR_FINAL_OUT + (chunk_cnt * 1000) + token_cnt;
                core_write_data <= lin_y_out_in;
                
                // Loop Logic
                if (chunk_cnt == 3) begin // Xong 4 chunks (64 channels)
                    chunk_cnt <= 0;
                    if (token_cnt == 999) begin
                        state <= S_DONE; // MISSION COMPLETE!
                    end else begin
                        token_cnt <= token_cnt + 1;
                        state <= S_OUTPROJ_SETUP; 
                    end
                end else begin
                    chunk_cnt <= chunk_cnt + 1;
                    state <= S_OUTPROJ_SETUP; // Chunk ti?p theo (16 output channels ti?p theo)
                end
            end
            
            // ============================================================
            // DEBUG PHASE: LINEAR 128->64 (TOKEN-FIRST READ MODE)
            // ============================================================
            
            S_DEBUG_SETUP: begin
                mode_select <= 3'd1; // Linear
                lin_len <= 128;      // Input 128
                lin_start <= 1;
                lin_bias_vals <= 0;  // X�a Bias c? (QUAN TR?NG)
                
                feed_x_idx <= 0;
                x_cache_idx <= 0;
                
                // --- �?C KI?U PHASE 1 (TU?N T?) ---
                // Input 128 ph?n t? = 8 d?ng RAM (16 s?/d?ng)
                // Addr = Base + (Token * 8)
                core_read_addr <= ADDR_DEBUG_IN + (token_cnt << 3);
                
                // Weight (V?n d�ng Chunk)
                base_weight_addr <= W_BASE_OUTPROJ + (chunk_cnt * 128);
                weight_read_addr <= W_BASE_OUTPROJ + (chunk_cnt * 128);
                
                bank_sel <= 0; // �?c RAM A
                lin_en <= 0;
                state <= S_DEBUG_READ;
            end
            
            S_DEBUG_READ: begin
                lin_start <= 0;
                state <= S_DEBUG_FEED; // B? qua b�?c WAIT RAM cho nhanh (ho?c th�m n?u RAM tr?)
            end
            
            S_DEBUG_FEED: begin
                // 1. N?P CACHE (Khi EN = 0)
                if (lin_en == 0) begin 
                    x_cache <= core_read_data; 
                    lin_en <= 1;
                    
                    // --- KH�C PHASE 5: �?C TU?N T? (+1) ---
                    core_read_addr <= core_read_addr + 1; 
                    weight_read_addr <= base_weight_addr + feed_x_idx + 1;
                end
                
                // 2. FEED (Khi EN = 1)
                else begin 
                    x_cache_idx <= x_cache_idx + 1;
                    feed_x_idx <= feed_x_idx + 1;
                    
                    if (feed_x_idx == 127) begin 
                        lin_en <= 0; 
                        state <= S_DEBUG_WAIT;
                        feed_x_idx <= 0; x_cache_idx <= 0;
                    end 
                    else if (x_cache_idx == 15) begin 
                        lin_en <= 0; 
                        x_cache_idx <= 0;
                        state <= S_DEBUG_READ; // Quay l?i �?c d?ng ti?p theo
                        
                        //weight_read_addr <= base_weight_addr + feed_x_idx + 1;
                    end 
                    else begin
                        weight_read_addr <= base_weight_addr + feed_x_idx + 2;
                    end
                end
            end
            
            S_DEBUG_WAIT: begin
                if (lin_done || lin_done_flag) begin
                    state <= S_DEBUG_WRITE;
                end
            end
            
            S_DEBUG_WRITE: begin
                core_write_en <= 1;
                bank_sel <= 0; // Ghi RAM B
                
                // Ghi Token-First ��n gi?n �? d? check
                // Addr = Base + (Token * 4) + Chunk
                core_write_addr <= ADDR_DEBUG_OUT + (token_cnt << 2) + chunk_cnt;
                core_write_data <= lin_y_out_in;
                
                if (chunk_cnt == 3) begin 
                    chunk_cnt <= 0;
                    if (token_cnt == 999) state <= S_DONE;
                    else begin
                        token_cnt <= token_cnt + 1;
                        state <= S_DEBUG_SETUP;
                    end
                end else begin
                    chunk_cnt <= chunk_cnt + 1;
                    state <= S_DEBUG_SETUP;
                end
            end

                S_DONE: begin
                    done_system <= 1;
                    if (!start_system) state <= S_IDLE;
                end
            endcase
        end
    end

    // --- Output Logic ---
    always @(*) begin
        lin_x_val  = 0;
        lin_W_vals = 0;
        conv_w_vec = w_conv_cache;
        softplus_in = lin_y_out_in[delta_buf_idx * 16 +: 16];
        
        // 1. PHASE 1, 3.1 & 5: D�ng x_cache index
        if (state == S_LIN_FEED_X || state == S_XPROJ_FEED || state == S_OUTPROJ_FEED || state == S_DEBUG_FEED) begin
            lin_x_val = x_cache[x_cache_idx * 16 +: 16];
            lin_W_vals = weight_read_data;
        end
        
        // 2. PHASE 3.2: D�ng dt_idx (l?y t? x_cache 4 ph?n t? �?u)
        else if (state == S_DT_FEED) begin
            lin_x_val  = x_cache[dt_idx * 16 +: 16]; 
            lin_W_vals = weight_read_data;
            lin_bias_vals = const_read_data;
        end
        
//        // Debug: trace MAC for token0, out_ch0 (chunk0 lane0)
//        if (state == S_DEBUG_FEED && lin_en &&
//            token_cnt == 0 && chunk_cnt == 0) begin
        
//            // feed_x_idx ch�nh l� input index i (0..127) c?a linear
//            if (feed_x_idx < 8) begin
//                $display("[MAC] t=%0d out_ch=%0d i=%0d | x_addr=%0d w_addr=%0d | x=%h w=%h | prod=%0d",
//                    token_cnt, 0, feed_x_idx,
//                    core_read_addr, weight_read_addr,
//                    lin_x_val,
//                    lin_W_vals[0*16 +: 16], // lane_out=0
//                    $signed(lin_x_val) * $signed(lin_W_vals[0*16 +: 16])
//                );
//            end
//        end


    end

endmodule