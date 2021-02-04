/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for back projection in PET image
reconstruction.

author: Pawel Markiewicz
Copyrights: 2018
------------------------------------------------------------------------*/
#include "auxmath.h"
#include "prjb.h"
#include "tprj.h"

__constant__ float2 c_li2rng[NLI2R];
__constant__ short2 c_li2sn[NLI2R];
__constant__ char c_li2nos[NLI2R];

//===============================================================
// copy to the smaller axially image
__global__ void imReduce(float *imr, float *im, int vz0, int nvz) {
  int iz = vz0 + threadIdx.x;
  int iy = SZ_IMZ * threadIdx.y + SZ_IMZ * blockDim.y * blockIdx.x;
  if (iy < SZ_IMY * SZ_IMZ) {
    int idx = SZ_IMZ * SZ_IMY * blockIdx.y + iy + iz;
    int idxr = threadIdx.x + (nvz * threadIdx.y + nvz * blockDim.y * blockIdx.x) +
               nvz * SZ_IMY * blockIdx.y;
    // copy to the axially smaller image
    imr[idxr] = im[idx];
  }
}
//===============================================================

//**************** DIRECT ***********************************
__global__ void bprj_drct(const float *sino, float *im, const float *tt, const unsigned char *tv,
                          const int *subs, const short snno) {
  int ixt = subs[blockIdx.x]; // transaxial indx
  int ixz = threadIdx.x;      // axial (z)

  float bin = sino[c_li2sn[ixz].x + blockIdx.x * snno];

  float z = c_li2rng[ixz].x + .5 * SZ_RING;
  int w = (floorf(.5 * SZ_IMZ + SZ_VOXZi * z));

  //-------------------------------------------------
  /*** accumulation ***/
  // vector a (at) component signs
  int sgna0 = tv[N_TV * ixt] - 1;
  int sgna1 = tv[N_TV * ixt + 1] - 1;
  bool rbit = tv[N_TV * ixt + 2] & 0x01; // row bit

  int u = (int)tt[N_TT * ixt + 8];
  int v = (u >> UV_SHFT);
  int uv = SZ_IMZ * ((u & 0x000001ff) + SZ_IMX * v);
  // next voxel (skipping the first fractional one)
  uv += !rbit * sgna0 * SZ_IMZ;
  uv -= rbit * sgna1 * SZ_IMZ * SZ_IMX;

  float dtr = tt[N_TT * ixt + 2];
  float dtc = tt[N_TT * ixt + 3];

  float trc = tt[N_TT * ixt] + rbit * dtr;
  float tcc = tt[N_TT * ixt + 1] + dtc * !rbit;
  rbit = tv[N_TV * ixt + 3] & 0x01;

  float tn = trc * rbit + tcc * !rbit; // next t
  float tp = tt[N_TT * ixt + 5];       // previous t

  float lt;
  //-------------------------------------------------

  for (int k = 3; k < (int)tt[N_TT * ixt + 9]; k++) {
    lt = tn - tp;

    atomicAdd(&im[uv + w], lt * bin);

    trc += dtr * rbit;
    tcc += dtc * !rbit;
    uv += !rbit * sgna0 * SZ_IMZ;
    uv -= rbit * sgna1 * SZ_IMZ * SZ_IMX;
    tp = tn;
    rbit = tv[N_TV * ixt + k + 1] & 0x01;
    tn = trc * rbit + tcc * !rbit;
  }
}

//************** OBLIQUE **************************************************
__global__ void bprj_oblq(const float *sino, float *im, const float *tt, const unsigned char *tv,
                          const int *subs, const short snno, const int zoff) {
  int ixz = threadIdx.x + zoff; // axial (z)
  if (ixz < NLI2R) {
    int ixt = subs[blockIdx.x]; // blockIdx.x is the transaxial bin index
                                // bin values to be back projected
    float bin = sino[c_li2sn[ixz].x + snno * blockIdx.x];
    float bin_ = sino[c_li2sn[ixz].y + snno * blockIdx.x];

    //-------------------------------------------------
    /*** accumulation ***/
    // vector a (at) component signs
    int sgna0 = tv[N_TV * ixt] - 1;
    int sgna1 = tv[N_TV * ixt + 1] - 1;
    bool rbit = tv[N_TV * ixt + 2] & 0x01; // row bit

    int u = (int)tt[N_TT * ixt + 8];
    int v = (u >> UV_SHFT);
    int uv = SZ_IMZ * ((u & 0x000001ff) + SZ_IMX * v);
    // next voxel (skipping the first fractional one)
    uv += !rbit * sgna0 * SZ_IMZ;
    uv -= rbit * sgna1 * SZ_IMZ * SZ_IMX;

    float dtr = tt[N_TT * ixt + 2];
    float dtc = tt[N_TT * ixt + 3];

    float trc = tt[N_TT * ixt] + rbit * dtr;
    float tcc = tt[N_TT * ixt + 1] + dtc * !rbit;
    rbit = tv[N_TV * ixt + 3] & 0x01;

    float tn = trc * rbit + tcc * !rbit; // next t
    float tp = tt[N_TT * ixt + 5];       // previous t
                                         //--------------------------------------------------

    //**** AXIAL *****
    float atn = tt[N_TT * ixt + 7];
    float az = c_li2rng[ixz].y - c_li2rng[ixz].x;
    float az_atn = az / atn;
    float s_az_atn = sqrtf(az_atn * az_atn + 1);
    int sgnaz;
    if (az >= 0)
      sgnaz = 1;
    else
      sgnaz = -1;

    float pz = c_li2rng[ixz].x + .5 * SZ_RING;
    float z = pz + az_atn * tp; // here was t1 = tt[N_TT*ixt+4]<<<<<<<<
    int w = (floorf(.5 * SZ_IMZ + SZ_VOXZi * z));
    float lz1 = (ceilf(.5 * SZ_IMZ + SZ_VOXZi * z)) * SZ_VOXZ -
                .5 * SZ_IMZ * SZ_VOXZ; // w is like in matlab by one greater

    z = c_li2rng[ixz].y + .5 * SZ_RING - az_atn * tp; // here was t1 = tt[N_TT*ixt+4]<<<<<<<<<
    int w_ = (floorf(.5 * SZ_IMZ + SZ_VOXZi * z));
    z = pz + az_atn * tt[N_TT * ixt + 6]; // t2
    float lz2 = (floorf(.5 * SZ_IMZ + SZ_VOXZi * z)) * SZ_VOXZ - .5 * SZ_IMZ * SZ_VOXZ;
    int nz = fabsf(lz2 - lz1) / SZ_VOXZ; // rintf
    float tz1 = (lz1 - pz) / az_atn;     // first ray interaction with a row
    float tz2 = (lz2 - pz) / az_atn;     // last ray interaction with a row
    float dtz = (tz2 - tz1) / nz;
    float tzc = tz1;
    //****************

    float fr, lt;

    for (int k = 3; k < tt[N_TT * ixt + 9];
         k++) { //<<< k=3 as 0 and 1 are for sign and 2 is skipped
      lt = tn - tp;
      if ((tn - tzc) > 0) {
        fr = (tzc - tp) / lt;
        atomicAdd(im + uv + w, fr * lt * s_az_atn * bin);
        atomicAdd(im + uv + w_, fr * lt * s_az_atn * bin_);
        // acc += fr*lt*s_az_atn * im[ w + uv ];
        // acc_+= fr*lt*s_az_atn * im[ w_+ uv ];
        w += sgnaz;
        w_ -= sgnaz;
        atomicAdd(im + uv + w, (1 - fr) * lt * s_az_atn * bin);
        atomicAdd(im + uv + w_, (1 - fr) * lt * s_az_atn * bin_);
        // acc += (1-fr)*lt*s_az_atn * im[ w + uv];
        // acc_+= (1-fr)*lt*s_az_atn * im[ w_+ uv];
        tzc += dtz;
      } else {
        atomicAdd(im + uv + w, lt * s_az_atn * bin);
        atomicAdd(im + uv + w_, lt * s_az_atn * bin_);
        // acc += lt*s_az_atn * im[ w + uv ];
        // acc_+= lt*s_az_atn * im[ w_+ uv ];
      }

      trc += dtr * rbit;
      tcc += dtc * !rbit;

      uv += !rbit * sgna0 * SZ_IMZ;
      uv -= rbit * sgna1 * SZ_IMZ * SZ_IMY;

      tp = tn;
      rbit = tv[N_TV * ixt + k + 1] & 0x01;
      tn = trc * rbit + tcc * !rbit;
    }
  }
}

//--------------------------------------------------------------------------------------------------
void gpu_bprj(float *d_im, float *d_sino, float *li2rng, short *li2sn, char *li2nos, short *s2c,
              int *aw2ali, float *crs, int *subs, int Nprj, int Naw, int N0crs, Cnst Cnt) {

  int dev_id;
  cudaGetDevice(&dev_id);
  if (Cnt.LOG <= LOGDEBUG) printf("i> using CUDA device #%d\n", dev_id);

  //--- TRANSAXIAL COMPONENT
  float4 *d_crs;
  HANDLE_ERROR(cudaMalloc(&d_crs, N0crs * sizeof(float4)));
  HANDLE_ERROR(cudaMemcpy(d_crs, crs, N0crs * sizeof(float4), cudaMemcpyHostToDevice));

  short2 *d_s2c;
  HANDLE_ERROR(cudaMalloc(&d_s2c, AW * sizeof(short2)));
  HANDLE_ERROR(cudaMemcpy(d_s2c, s2c, AW * sizeof(short2), cudaMemcpyHostToDevice));

  float *d_tt;
  HANDLE_ERROR(cudaMalloc(&d_tt, N_TT * AW * sizeof(float)));

  unsigned char *d_tv;
  HANDLE_ERROR(cudaMalloc(&d_tv, N_TV * AW * sizeof(unsigned char)));
  HANDLE_ERROR(cudaMemset(d_tv, 0, N_TV * AW * sizeof(unsigned char)));

  // array of subset projection bins
  int *d_subs;
  HANDLE_ERROR(cudaMalloc(&d_subs, Nprj * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(d_subs, subs, Nprj * sizeof(int), cudaMemcpyHostToDevice));
  //---

  //-----------------------------------------------------------------
  // RINGS: either all or a subset of rings can be used for fast calc.
  //-----------------------------------------------------------------
  // number of rings customised
  int nrng_c, nil2r_c, vz0, vz1, nvz;
  // number of sinos
  short snno = -1;
  if (Cnt.SPN == 1) {
    // number of direct rings considered
    nrng_c = Cnt.RNG_END - Cnt.RNG_STRT;
    // number of "positive" michelogram elements used for projection (can be smaller than the
    // maximum)
    nil2r_c = (nrng_c + 1) * nrng_c / 2;
    snno = nrng_c * nrng_c;
    // correct for the max. ring difference in the full axial extent (don't use ring range (1,63)
    // as for this case no correction)
    if (nrng_c == NRINGS) {
      snno -= 12;
      nil2r_c -= 6;
    }
  } else if (Cnt.SPN == 11) {
    snno = NSINOS11;
    nrng_c = NRINGS;
    nil2r_c = NLI2R;
  }
  // voxels in axial direction
  vz0 = 2 * Cnt.RNG_STRT;
  vz1 = 2 * (Cnt.RNG_END - 1);
  nvz = 2 * nrng_c - 1;
  if (Cnt.LOG <= LOGDEBUG) {
    printf("i> detector rings range: [%d, %d) => number of  sinos: %d\n", Cnt.RNG_STRT,
           Cnt.RNG_END, snno);
    printf("   corresponding voxels: [%d, %d] => number of voxels: %d\n", vz0, vz1, nvz);
  }
  //-----------------------------------------------------------------

  float *d_imf;
  // when rings are reduced
  if (nvz < SZ_IMZ)
    HANDLE_ERROR(cudaMalloc(&d_imf, SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(float)));
  else
    d_imf = d_im;
  HANDLE_ERROR(cudaMemset(d_imf, 0, SZ_IMX * SZ_IMY * SZ_IMZ * sizeof(float)));
  //---

  cudaMemcpyToSymbol(c_li2rng, li2rng, nil2r_c * sizeof(float2));
  cudaMemcpyToSymbol(c_li2sn, li2sn, nil2r_c * sizeof(short2));
  cudaMemcpyToSymbol(c_li2nos, li2nos, nil2r_c * sizeof(char));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  if (Cnt.LOG <= LOGDEBUG) printf("i> calculating image through back projection... ");

  //------------DO TRANSAXIAL CALCULATIONS---------------------------------
  gpu_siddon_tx(d_crs, d_s2c, d_tt, d_tv);
  //-----------------------------------------------------------------------

  //============================================================================
  bprj_drct<<<Nprj, nrng_c>>>(d_sino, d_imf, d_tt, d_tv, d_subs, snno);
  HANDLE_ERROR(cudaGetLastError());
  //============================================================================

  int zoff = nrng_c;
  // number of oblique sinograms
  int Noblq = (nrng_c - 1) * nrng_c / 2;

  // cudaGetDeviceCount(&nDevices);
  // for (int i = 0; i < nDevices; i++) {
  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, i);
  // printf("Device Number: %d\n", i);
  // printf("  Device name: %s\n", prop.name);
  // printf("  Device supports concurrentManagedAccess?: %s\n", prop.concurrentManagedAccess);
  //}

  // cudaMemPrefetchAsync(d_sino, Nprj*snno * sizeof(float), nDevices, NULL);

  if (Cnt.SPN == 1 && Noblq <= 1024) {
    bprj_oblq<<<Nprj, Noblq>>>(d_sino, d_imf, d_tt, d_tv, d_subs, snno, zoff);
    HANDLE_ERROR(cudaGetLastError());
  } else {
    bprj_oblq<<<Nprj, NSINOS / 4>>>(d_sino, d_imf, d_tt, d_tv, d_subs, snno, zoff);
    HANDLE_ERROR(cudaGetLastError());
    zoff += NSINOS / 4;
    bprj_oblq<<<Nprj, NSINOS / 4>>>(d_sino, d_imf, d_tt, d_tv, d_subs, snno, zoff);
    HANDLE_ERROR(cudaGetLastError());
  }
  //============================================================================

  // // the actual axial size used (due to the customised ring subset used)
  // int vz0 = 2*Cnt.RNG_STRT;
  // int vz1 = 2*(Cnt.RNG_END-1);
  // // number of voxel for reduced number of rings (customised)
  // int nvz = vz1-vz0+1;

  // when rings are reduced
  if (nvz < SZ_IMZ) {
    // number of axial row for max threads
    int nar = NIPET_CU_THREADS / nvz;
    dim3 THRD(nvz, nar, 1);
    dim3 BLCK((SZ_IMY + nar - 1) / nar, SZ_IMX, 1);
    imReduce<<<BLCK, THRD>>>(d_im, d_imf, vz0, nvz);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaFree(d_imf));
    if (Cnt.LOG <= LOGDEBUG) printf("i> reduced the axial (z) image size to %d\n", nvz);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  // cudaDeviceSynchronize();
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (Cnt.LOG <= LOGDEBUG) printf("DONE in %fs.\n", 0.001 * elapsedTime);

  HANDLE_ERROR(cudaFree(d_tt));
  HANDLE_ERROR(cudaFree(d_tv));
  HANDLE_ERROR(cudaFree(d_subs));
  HANDLE_ERROR(cudaFree(d_crs));
  HANDLE_ERROR(cudaFree(d_s2c));
}

//=======================================================================
void rec_bprj(float *d_bimg, float *d_sino, int *d_sub, int Nprj, float *d_tt, unsigned char *d_tv,
              float *li2rng, short *li2sn, char *li2nos, Cnst Cnt)

{

  int dev_id;
  cudaGetDevice(&dev_id);
  if (Cnt.LOG <= LOGDEBUG) printf("i> using CUDA device #%d\n", dev_id);

  // get the axial LUTs in constant memory
  cudaMemcpyToSymbol(c_li2rng, li2rng, NLI2R * sizeof(float2));
  cudaMemcpyToSymbol(c_li2sn, li2sn, NLI2R * sizeof(short2));
  cudaMemcpyToSymbol(c_li2nos, li2nos, NLI2R * sizeof(char));

  // number of sinos
  short snno = -1;
  if (Cnt.SPN == 1)
    snno = NSINOS;
  else if (Cnt.SPN == 11)
    snno = NSINOS11;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  if (Cnt.LOG <= LOGDEBUG) printf("i> subset back projection (Nprj=%d)... ", Nprj);

  //============================================================================
  bprj_drct<<<Nprj, NRINGS>>>(d_sino, d_bimg, d_tt, d_tv, d_sub, snno);
  // HANDLE_ERROR(cudaGetLastError());
  //============================================================================

  int zoff = NRINGS;
  //============================================================================
  bprj_oblq<<<Nprj, NSINOS / 4>>>(d_sino, d_bimg, d_tt, d_tv, d_sub, snno, zoff);
  // HANDLE_ERROR(cudaGetLastError());
  //============================================================================

  zoff += NSINOS / 4;
  //============================================================================
  bprj_oblq<<<Nprj, NSINOS / 4>>>(d_sino, d_bimg, d_tt, d_tv, d_sub, snno, zoff);
  // HANDLE_ERROR(cudaGetLastError());
  //============================================================================

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (Cnt.LOG <= LOGDEBUG) printf("DONE in %fs.\n", 0.001 * elapsedTime);

  cudaDeviceSynchronize();

  return;
}
