#pragma once
#define USE_D_MODEL_64 1


#if USE_D_MODEL_64
    const int D_MODEL = 64;

#else
    const int D_MODEL = 128;

#endif
const int SEQ_LEN = 1000; 
const int D_STATE = 16;
const int D_CONV = 4;
const int EXPAND = 2;
const int D_INNER = EXPAND * D_MODEL;
const int DT_RANK = (D_MODEL + 15) / 16; 

typedef float model_dtype;