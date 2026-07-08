#pragma once
#include "MambaConfig.h"

void linear(const model_dtype x[], model_dtype y[], 
            const model_dtype* W, const model_dtype b[],
            int in_dim, int out_dim);

void causal_conv1d(const model_dtype x[][SEQ_LEN], model_dtype out[][SEQ_LEN],
                   const model_dtype weight[][D_CONV], const model_dtype bias[]);

model_dtype silu(model_dtype x);

void scan_core(
    const model_dtype discrete_A[][SEQ_LEN][D_STATE],
    const model_dtype deltaB_u[][SEQ_LEN][D_STATE],
    const model_dtype C_raw[][D_STATE],
    model_dtype scan_output_raw[][SEQ_LEN]
);

model_dtype softplus(model_dtype x);
void RMSNorm(const model_dtype in[], model_dtype out[], const model_dtype weight[], int size);