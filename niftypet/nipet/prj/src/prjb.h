#include "def.h"
#include "scanner_0.h"
#include "tprj.h"
#include <stdio.h>

#ifndef PRJB_H
#define PRJB_H

// used from Python
void gpu_bprj(float *d_im, float *d_sino, float *li2rng, short *li2sn, char *li2nos, short *s2c,
              int *aw2ali, float *crs, int *subs, int Nprj, int Naw, int N0crs, Cnst Cnt);

// to be used within CUDA C reconstruction
void rec_bprj(float *d_bimg, float *d_sino, int *sub, int Nprj,

              float *d_tt, unsigned char *d_tv,

              float *li2rng, short *li2sn, char *li2nos,

              Cnst Cnt);

#endif
