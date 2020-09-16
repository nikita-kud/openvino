// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <moviVectorUtils.h>
#include <moviVectorFunctions.h>
#include <math.h>
#include <stdio.h>

const int NUM_SHAVES = 16;

extern "C" void ShuffleChannel(
    const half* __restrict__ src_data, half* __restrict__ dst_data,
    int C, int H, int W, int G, int shave_id);

// default entry 0x1f
extern "C" void custom_cpp(uint32_t* params, int shave_id) {    
    ShuffleChannel((const half*)params[0], (half*)params[1], 
                    (int)params[2], (int)params[3], (int)params[4], (int)params[5], shave_id);
}

extern "C" void ShuffleChannel(const half* __restrict__ src_data,
                               half* __restrict__ dst_data,
                               int C,
                               int H,
                               int W,
                               int G,
                               int shave_id) 
{
    int chunk = C / NUM_SHAVES;
    if(shave_id == NUM_SHAVES - 1) {
        chunk += C % NUM_SHAVES;
    }

    int start = shave_id * chunk;
    int end = start + chunk;
    for (int c = start; c < end; c++)
    {    
        int CX = C / G;
        int CY = G;
        int cy = c % G;
        int cx = c / G;

        const half8* src_line = ((const half8*)(src_data + cy*CX*H*W + cx*H*W));
              half8* dst_line = ((      half8*)(dst_data + cx*CY*H*W + cy*H*W));

        for (int i = 0; i < W*H/8; i++)
        {
            dst_line[i] = src_line[i];
        }

        for (int i = W*H/8*8; i < W*H; i++)
        {
            dst_data[cx*CY*H*W + cy*H*W + i] = src_data[cy*CX*H*W + cx*H*W + i];
        }
    }
}
