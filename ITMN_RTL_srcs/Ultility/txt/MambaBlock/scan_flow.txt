#include "Main.h"
#include <iostream>
#include "Layers.h"
#include <cmath>

void transpose_2d_D_INNER_to_SEQ_LEN(const model_dtype in[D_INNER][SEQ_LEN], model_dtype out[SEQ_LEN][D_INNER]) {
    for (int i = 0; i < D_INNER; ++i) {
        for (int j = 0; j < SEQ_LEN; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

void transpose_2d_SEQ_LEN_to_D_INNER(const model_dtype in[SEQ_LEN][D_INNER], model_dtype out[D_INNER][SEQ_LEN]) {
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int j = 0; j < D_INNER; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

template<size_t L, size_t D>
void transpose_ld_to_dl(const model_dtype in[L][D], model_dtype out[D][L]) {
    for (size_t i = 0; i < L; ++i) {
        for (size_t j = 0; j < D; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

template<size_t L, size_t N>
void transpose_2d_SEQ_LEN_to_D_STATE(const model_dtype in[L][N], model_dtype out[N][L]) {
    for (size_t i = 0; i < L; ++i) {
        for (size_t j = 0; j < N; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

void ITMN_mamba_block(
    const model_dtype hidden_states[SEQ_LEN][D_MODEL],
    model_dtype output[SEQ_LEN][D_MODEL],
    const ITMNMambaMixerWeights* weights
) {

    static model_dtype x_norm[SEQ_LEN][D_MODEL];
    static model_dtype x_primary_expanded[SEQ_LEN][D_INNER];
    static model_dtype x_primary_transposed[D_INNER][SEQ_LEN];
    static model_dtype x_conv[D_INNER][SEQ_LEN];
    static model_dtype x_prime[D_INNER][SEQ_LEN]; 
    static model_dtype x_prime_rearranged[SEQ_LEN][D_INNER];
    static model_dtype ssm_parameters[SEQ_LEN][DT_RANK + D_STATE * 2];
    static model_dtype dt_raw[SEQ_LEN][DT_RANK];
    static model_dtype B_raw[SEQ_LEN][D_STATE];
    static model_dtype C_raw[SEQ_LEN][D_STATE];
    static model_dtype dt_proj_out[SEQ_LEN][D_INNER];
    static model_dtype dt_softplus[SEQ_LEN][D_INNER];
    static model_dtype delta[D_INNER][SEQ_LEN];
    static model_dtype A_cpp[D_INNER][D_STATE];
    static model_dtype discrete_A[D_INNER][SEQ_LEN][D_STATE];
    static model_dtype deltaB_u[D_INNER][SEQ_LEN][D_STATE];
    static model_dtype scan_output_raw[D_INNER][SEQ_LEN]; 
    
    static model_dtype scan_output_with_D[D_INNER][SEQ_LEN]; 

    static model_dtype z_expanded[SEQ_LEN][D_INNER];
    static model_dtype z_activated[SEQ_LEN][D_INNER];
    static model_dtype y_rearranged[SEQ_LEN][D_INNER];
    static model_dtype y_gated[SEQ_LEN][D_INNER];

    // BỘ NHỚ TRẠNG THÁI CHÍNH
    static model_dtype h_state[D_INNER][D_STATE] = {0.0f};

    // BỘ NHỚ GATE CHO TÍNH TOÁN GATING
    static model_dtype gate[D_INNER][SEQ_LEN];

    for (int i = 0; i < SEQ_LEN; ++i) {
        RMSNorm(hidden_states[i], x_norm[i], weights->rms_norm_weight, D_MODEL);
    }

    for (int i = 0; i < SEQ_LEN; ++i) {
        linear(x_norm[i], x_primary_expanded[i], weights->in_proj1_weight, nullptr, D_MODEL, D_INNER);
    }
    transpose_2d_SEQ_LEN_to_D_INNER(x_primary_expanded, x_primary_transposed);

    causal_conv1d(x_primary_transposed, x_conv, (const model_dtype(*)[D_CONV])weights->conv1d_weight, weights->conv1d_bias);
    for (int i = 0; i < D_INNER; ++i) for (int j = 0; j < SEQ_LEN; ++j) {
        x_prime[i][j] = silu(x_conv[i][j]);
    }

    transpose_2d_D_INNER_to_SEQ_LEN(x_prime, x_prime_rearranged);
    for (int i = 0; i < SEQ_LEN; ++i) {
        linear(x_prime_rearranged[i], ssm_parameters[i], weights->x_proj_weight, nullptr, D_INNER, DT_RANK + D_STATE * 2);
    }
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int j = 0; j < DT_RANK; ++j) dt_raw[i][j] = ssm_parameters[i][j];
        for (int j = 0; j < D_STATE; ++j) B_raw[i][j] = ssm_parameters[i][j + DT_RANK];
        for (int j = 0; j < D_STATE; ++j) C_raw[i][j] = ssm_parameters[i][j + DT_RANK + D_STATE];
    }
    for (int i = 0; i < SEQ_LEN; ++i) {
        linear(dt_raw[i], dt_proj_out[i], weights->dt_proj_weight, weights->dt_proj_bias, DT_RANK, D_INNER);
    }
    for (int i = 0; i < SEQ_LEN; ++i) for (int j = 0; j < D_INNER; ++j) {
        dt_softplus[i][j] = softplus(dt_proj_out[i][j]);
    }
    transpose_2d_SEQ_LEN_to_D_INNER(dt_softplus, delta);

    // --- BƯỚC 6-9: VÒNG LẶP THỜI GIAN HỢP NHẤT (TIME-STATIONARY FUSED LOOP) ---
    for (int l = 0; l < SEQ_LEN; ++l) {
        
        for (int d = 0; d < D_INNER; ++d) {
            
            model_dtype h_new[D_STATE];
            
            // 6. Rời rạc hóa và Cập nhật h
            for (int n = 0; n < D_STATE; ++n) {
                model_dtype discrete_A_val = std::exp(A_cpp[d][n] * delta[d][l]);
                model_dtype deltaB_u_val = (delta[d][l] * B_raw[l][n]) * x_prime[d][l];
                
                h_new[n] = discrete_A_val * h_state[d][n] + deltaB_u_val;
            }

            // 7. Tính y_scan
            model_dtype y_scan = 0.0f;
            for (int n = 0; n < D_STATE; ++n) {
                y_scan += C_raw[l][n] * h_new[n];
            }

            // 8. Thêm D và Gating
            model_dtype y_with_D = y_scan + (x_prime[d][l] * weights->D[d]);
            model_dtype y_gated = y_with_D * silu(gate[d][l]);

            // Cập nhật trạng thái cho kênh d
            for (int n = 0; n < D_STATE; ++n) {
                h_state[d][n] = h_new[n];
            }
        }
    }

    for (int i = 0; i < SEQ_LEN; ++i) {
        linear(x_norm[i], z_expanded[i], weights->in_proj2_weight, nullptr, D_MODEL, D_INNER);
    }
    for (int i = 0; i < SEQ_LEN; ++i) for (int j = 0; j < D_INNER; ++j) {
        z_activated[i][j] = silu(z_expanded[i][j]);
    }

    transpose_2d_D_INNER_to_SEQ_LEN(scan_output_with_D, y_rearranged);

    for (int i = 0; i < SEQ_LEN; ++i) for (int j = 0; j < D_INNER; ++j) {
        y_gated[i][j] = y_rearranged[i][j] * z_activated[i][j];
    }

    for (int i = 0; i < SEQ_LEN; ++i) {
        linear(y_gated[i], output[i], weights->out_proj_weight, nullptr, D_INNER, D_MODEL);
    }
}