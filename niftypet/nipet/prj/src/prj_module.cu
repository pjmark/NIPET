/*------------------------------------------------------------------------
CUDA C extension for Python
Provides functionality for forward and back projection in PET image
reconstruction.

author: Pawel Markiewicz
Copyrights: 2019
------------------------------------------------------------------------*/

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // NPY_API_VERSION

#include "def.h"
#include "pycuvec.cuh"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>

#include "prjb.h"
#include "prjf.h"

#include "tprj.h"

#include "recon.h"
#include "scanner_0.h"

//===================== START PYTHON INIT ==============================

//--- Available functions
static PyObject *trnx_prj(PyObject *self, PyObject *args);
static PyObject *frwd_prj(PyObject *self, PyObject *args);
static PyObject *back_prj(PyObject *self, PyObject *args);
static PyObject *osem_rec(PyObject *self, PyObject *args);
//---

//> Module Method Table
static PyMethodDef petprj_methods[] = {
    {"tprj", trnx_prj, METH_VARARGS, "Transaxial projector."},
    {"fprj", frwd_prj, METH_VARARGS, "PET forward projector."},
    {"bprj", back_prj, METH_VARARGS, "PET back projector."},
    {"osem", osem_rec, METH_VARARGS, "OSEM reconstruction of PET data."},
    {NULL, NULL, 0, NULL} // Sentinel
};

//> Module Definition Structure
static struct PyModuleDef petprj_module = {
    PyModuleDef_HEAD_INIT,
    "petprj", //> name of module
    //> module documentation, may be NULL
    "This module provides an interface for GPU routines of PET forward and back projection.",
    -1, //> the module keeps state in global variables.
    petprj_methods};

//> Initialization function
PyMODINIT_FUNC PyInit_petprj(void) {

  Py_Initialize();

  //> load NumPy functionality
  import_array();

  return PyModule_Create(&petprj_module);
}

//====================== END PYTHON INIT ===============================

//==============================================================================
// T R A N S A X I A L   P R O J E C T O R
//------------------------------------------------------------------------------
static PyObject *trnx_prj(PyObject *self, PyObject *args) {
  // Structure of constants
  Cnst Cnt;

  // Dictionary of scanner constants
  PyObject *o_mmrcnst;

  // transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
  PyObject *o_txLUT;

  // input/output image
  PyObject *o_im;

  // input/output projection sinogram
  PyObject *o_prjout;

  // output transaxial sampling parameters
  PyObject *o_tv;
  PyObject *o_tt;

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOOOOO", &o_prjout, &o_im, &o_tv, &o_tt, &o_txLUT, &o_mmrcnst))
    return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  PyObject *pd_log = PyDict_GetItemString(o_mmrcnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);
  PyObject *pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);

  // transaxial sino LUTs:
  PyObject *pd_crs = PyDict_GetItemString(o_txLUT, "crs");
  PyObject *pd_s2c = PyDict_GetItemString(o_txLUT, "s2c");

  // sino to crystal, crystals
  PyArrayObject *p_s2c = NULL, *p_crs = NULL;
  p_s2c = (PyArrayObject *)PyArray_FROM_OTF(pd_s2c, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_crs = (PyArrayObject *)PyArray_FROM_OTF(pd_crs, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  // image object
  PyArrayObject *p_im = NULL;
  p_im = (PyArrayObject *)PyArray_FROM_OTF(o_im, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);

  // output sino object
  PyArrayObject *p_prjout = NULL;
  p_prjout = (PyArrayObject *)PyArray_FROM_OTF(o_prjout, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);

  // transaxial voxel sampling (ray-driven)
  PyArrayObject *p_tv = NULL;
  p_tv = (PyArrayObject *)PyArray_FROM_OTF(o_tv, NPY_UINT8, NPY_ARRAY_INOUT_ARRAY2);

  // transaxial parameters for voxel sampling (ray-driven)
  PyArrayObject *p_tt = NULL;
  p_tt = (PyArrayObject *)PyArray_FROM_OTF(o_tt, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);

  //--

  /* If that didn't work, throw an exception. */
  if (p_s2c == NULL || p_im == NULL || p_crs == NULL || p_prjout == NULL || p_tv == NULL ||
      p_tt == NULL) {
    // sino 2 crystals
    Py_XDECREF(p_s2c);
    Py_XDECREF(p_crs);

    // image object
    PyArray_DiscardWritebackIfCopy(p_im);
    Py_XDECREF(p_im);

    // output sino object
    PyArray_DiscardWritebackIfCopy(p_prjout);
    Py_XDECREF(p_prjout);

    // transaxial outputs
    PyArray_DiscardWritebackIfCopy(p_tv);
    Py_XDECREF(p_tv);

    PyArray_DiscardWritebackIfCopy(p_tt);
    Py_XDECREF(p_tt);

    return NULL;
  }

  short *s2c = (short *)PyArray_DATA(p_s2c);
  float *crs = (float *)PyArray_DATA(p_crs);

  int N0crs = PyArray_DIM(p_crs, 0);
  int N1crs = PyArray_DIM(p_crs, 1);
  if (Cnt.LOG <= LOGDEBUG) printf("\ni> N0crs=%d, N1crs=%d\n", N0crs, N1crs);

  float *im = (float *)PyArray_DATA(p_im);
  if (Cnt.LOG <= LOGDEBUG)
    printf("i> forward-projection image dimensions: %ld, %ld\n", PyArray_DIM(p_im, 0),
           PyArray_DIM(p_im, 1));

  // input/output projection sinogram
  float *prjout = (float *)PyArray_DATA(p_prjout);

  // output sampling
  unsigned char *tv = (unsigned char *)PyArray_DATA(p_tv);
  float *tt = (float *)PyArray_DATA(p_tt);

  // CUDA --------------------------------------------------------------------

  // sets the device on which to calculate
  HANDLE_ERROR(cudaSetDevice(Cnt.DEVID));

  int dev_id;
  cudaGetDevice(&dev_id);
  if (Cnt.LOG <= LOGDEBUG) printf("i> using CUDA device #%d\n", dev_id);

  //--- TRANSAXIAL COMPONENTS
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

  //------------DO TRANSAXIAL CALCULATIONS------------------------------------
  gpu_siddon_tx(d_crs, d_s2c, d_tt, d_tv);
  //--------------------------------------------------------------------------

  HANDLE_ERROR(cudaMemcpy(tt, d_tt, N_TT * AW * sizeof(float), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(tv, d_tv, N_TV * AW * sizeof(unsigned char), cudaMemcpyDeviceToHost));

  // CUDA END-----------------------------------------------------------------

  // Clean up
  Py_DECREF(p_s2c);
  Py_DECREF(p_crs);

  PyArray_ResolveWritebackIfCopy(p_im);
  Py_DECREF(p_im);

  PyArray_ResolveWritebackIfCopy(p_tv);
  Py_DECREF(p_tv);

  PyArray_ResolveWritebackIfCopy(p_tt);
  Py_DECREF(p_tt);

  PyArray_ResolveWritebackIfCopy(p_prjout);
  Py_DECREF(p_prjout);

  Py_INCREF(Py_None);
  return Py_None;
}

//------------------------------------------------------------------------------

//==============================================================================
// F O R W A R D   P R O J E C T O R
//------------------------------------------------------------------------------

static PyObject *frwd_prj(PyObject *self, PyObject *args) {
  // Structure of constants
  Cnst Cnt;

  // Dictionary of scanner constants
  PyObject *o_mmrcnst;

  // axial LUT dictionary. contains such LUTs: li2rno, li2sn, li2nos.
  PyObject *o_axLUT;

  // transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
  PyObject *o_txLUT;

  // input image to be forward projected  (reshaped for GPU execution)
  PyCuVec<float> *o_im;

  // subsets for OSEM, first the default
  PyObject *o_subs;

  // output projection sino
  PyCuVec<float> *o_prjout;

  // flag for attenuation factors to be found based on mu-map; if 0 normal emission projection is
  // used
  int att;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOOOOOi", (PyObject **)&o_prjout, (PyObject **)&o_im, &o_txLUT,
                        &o_axLUT, &o_subs, &o_mmrcnst, &att))
    return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  PyObject *pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
  Cnt.SPN = (char)PyLong_AsLong(pd_span);
  PyObject *pd_rngstrt = PyDict_GetItemString(o_mmrcnst, "RNG_STRT");
  Cnt.RNG_STRT = (char)PyLong_AsLong(pd_rngstrt);
  PyObject *pd_rngend = PyDict_GetItemString(o_mmrcnst, "RNG_END");
  Cnt.RNG_END = (char)PyLong_AsLong(pd_rngend);
  PyObject *pd_log = PyDict_GetItemString(o_mmrcnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);
  PyObject *pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);

  /* Interpret the input objects as numpy arrays. */
  // axial LUTs:
  PyObject *pd_li2rno = PyDict_GetItemString(o_axLUT, "li2rno");
  PyObject *pd_li2sn = PyDict_GetItemString(o_axLUT, "li2sn");
  PyObject *pd_li2sn1 = PyDict_GetItemString(o_axLUT, "li2sn1");
  PyObject *pd_li2nos = PyDict_GetItemString(o_axLUT, "li2nos");
  PyObject *pd_li2rng = PyDict_GetItemString(o_axLUT, "li2rng");

  //-- get the arrays from the dictionaries
  // axLUTs
  PyArrayObject *p_li2rno = NULL, *p_li2sn1 = NULL, *p_li2sn = NULL;
  PyArrayObject *p_li2nos = NULL, *p_li2rng = NULL;
  p_li2rno = (PyArrayObject *)PyArray_FROM_OTF(pd_li2rno, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_li2sn1 = (PyArrayObject *)PyArray_FROM_OTF(pd_li2sn1, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_li2sn = (PyArrayObject *)PyArray_FROM_OTF(pd_li2sn, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_li2nos = (PyArrayObject *)PyArray_FROM_OTF(pd_li2nos, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_li2rng = (PyArrayObject *)PyArray_FROM_OTF(pd_li2rng, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  // transaxial sino LUTs:
  PyObject *pd_crs = PyDict_GetItemString(o_txLUT, "crs");
  PyObject *pd_s2c = PyDict_GetItemString(o_txLUT, "s2c");
  PyObject *pd_aw2ali = PyDict_GetItemString(o_txLUT, "aw2ali");

  // sino to crystal, crystals
  PyArrayObject *p_s2c = NULL, *p_crs = NULL, *p_aw2ali = NULL;
  p_s2c = (PyArrayObject *)PyArray_FROM_OTF(pd_s2c, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_crs = (PyArrayObject *)PyArray_FROM_OTF(pd_crs, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  p_aw2ali = (PyArrayObject *)PyArray_FROM_OTF(pd_aw2ali, NPY_INT32, NPY_ARRAY_IN_ARRAY);

  // subsets if using e.g., OSEM
  PyArrayObject *p_subs = NULL;
  p_subs = (PyArrayObject *)PyArray_FROM_OTF(o_subs, NPY_INT32, NPY_ARRAY_IN_ARRAY);

  //--

  /* If that didn't work, throw an exception. */
  if (p_li2rno == NULL || p_li2sn == NULL || p_li2sn1 == NULL || p_li2nos == NULL ||
      p_aw2ali == NULL || p_s2c == NULL || !o_im || p_crs == NULL || p_subs == NULL || !o_prjout ||
      p_li2rng == NULL) {
    // axLUTs
    Py_XDECREF(p_li2rno);
    Py_XDECREF(p_li2sn);
    Py_XDECREF(p_li2sn1);
    Py_XDECREF(p_li2nos);
    Py_XDECREF(p_li2rng);

    // 2D sino LUT
    Py_XDECREF(p_aw2ali);
    // sino 2 crystals
    Py_XDECREF(p_s2c);
    Py_XDECREF(p_crs);
    // subset definition object
    Py_XDECREF(p_subs);
    return NULL;
  }

  int *subs_ = (int *)PyArray_DATA(p_subs);
  short *s2c = (short *)PyArray_DATA(p_s2c);
  int *aw2ali = (int *)PyArray_DATA(p_aw2ali);
  short *li2sn;
  if (Cnt.SPN == 11) {
    li2sn = (short *)PyArray_DATA(p_li2sn);
  } else if (Cnt.SPN == 1) {
    li2sn = (short *)PyArray_DATA(p_li2sn1);
  }
  char *li2nos = (char *)PyArray_DATA(p_li2nos);
  float *li2rng = (float *)PyArray_DATA(p_li2rng);
  float *crs = (float *)PyArray_DATA(p_crs);

  if (Cnt.LOG <= LOGDEBUG)
    printf("i> forward-projection image dimensions: %ld, %ld, %ld\n", o_im->shape[0],
           o_im->shape[1], o_im->shape[2]);

  int Nprj = PyArray_DIM(p_subs, 0);
  int N0crs = PyArray_DIM(p_crs, 0);
  int N1crs = PyArray_DIM(p_crs, 1);
  int Naw = PyArray_DIM(p_aw2ali, 0);

  if (Cnt.LOG <= LOGDEBUG)
    printf("\ni> N0crs=%d, N1crs=%d, Naw=%d, Nprj=%d\n", N0crs, N1crs, Naw, Nprj);

  int *subs;
  if (subs_[0] == -1) {
    Nprj = AW;
    if (Cnt.LOG <= LOGDEBUG)
      printf("i> no subsets defined.  number of projection bins in 2D: %d\n", Nprj);
    // all projections in
    subs = (int *)malloc(Nprj * sizeof(int));
    for (int i = 0; i < Nprj; i++) { subs[i] = i; }
  } else {
    if (Cnt.LOG <= LOGDEBUG)
      printf("i> subsets defined.  number of subset projection bins in 2D: %d\n", Nprj);
    subs = subs_;
  }

  // sets the device on which to calculate
  HANDLE_ERROR(cudaSetDevice(Cnt.DEVID));

  //<><><><><><><<><><><><><><><><><><><><><><><><><<><><><><><><><><><><><><><><><><><><<><><><><><><><><><><>
  gpu_fprj(o_prjout->vec.data(), o_im->vec.data(), li2rng, li2sn, li2nos, s2c, aw2ali, crs, subs,
           Nprj, Naw, N0crs, Cnt, att);
  //<><><><><><><><<><><><><><><><><><><><><><><><><<><><><><><><><><><><><><><><><><><><<><><><><><><><><><><>

  // Clean up
  Py_DECREF(p_li2rno);
  Py_DECREF(p_li2rng);
  Py_DECREF(p_li2sn);
  Py_DECREF(p_li2sn1);
  Py_DECREF(p_li2nos);
  Py_DECREF(p_aw2ali);
  Py_DECREF(p_s2c);
  Py_DECREF(p_crs);
  Py_DECREF(p_subs);

  if (subs_[0] == -1) free(subs);

  Py_INCREF(Py_None);
  return Py_None;
}

//==============================================================================
// B A C K   P R O J E C T O R
//------------------------------------------------------------------------------
static PyObject *back_prj(PyObject *self, PyObject *args) {

  // Structure of constants
  Cnst Cnt;

  // Dictionary of scanner constants
  PyObject *o_mmrcnst;

  // axial LUT dicionary. contains such LUTs: li2rno, li2sn, li2nos.
  PyObject *o_axLUT;

  // transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
  PyObject *o_txLUT;

  // sino to be back projected to image (both reshaped for GPU execution)
  PyCuVec<float> *o_sino;

  // subsets for OSEM, first the default
  PyObject *o_subs;

  // output backprojected image
  PyCuVec<float> *o_bimg;

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOOOOO", (PyObject **)&o_bimg, (PyObject **)&o_sino, &o_txLUT,
                        &o_axLUT, &o_subs, &o_mmrcnst))
    return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  PyObject *pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
  Cnt.SPN = (char)PyLong_AsLong(pd_span);
  PyObject *pd_rngstrt = PyDict_GetItemString(o_mmrcnst, "RNG_STRT");
  Cnt.RNG_STRT = (char)PyLong_AsLong(pd_rngstrt);
  PyObject *pd_rngend = PyDict_GetItemString(o_mmrcnst, "RNG_END");
  Cnt.RNG_END = (char)PyLong_AsLong(pd_rngend);
  PyObject *pd_log = PyDict_GetItemString(o_mmrcnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);
  PyObject *pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);

  /* Interpret the input objects as numpy arrays. */
  // axial LUTs:
  PyObject *pd_li2rno = PyDict_GetItemString(o_axLUT, "li2rno");
  PyObject *pd_li2sn = PyDict_GetItemString(o_axLUT, "li2sn");
  PyObject *pd_li2sn1 = PyDict_GetItemString(o_axLUT, "li2sn1");
  PyObject *pd_li2nos = PyDict_GetItemString(o_axLUT, "li2nos");
  PyObject *pd_li2rng = PyDict_GetItemString(o_axLUT, "li2rng");

  // transaxial sino LUTs:
  PyObject *pd_crs = PyDict_GetItemString(o_txLUT, "crs");
  PyObject *pd_s2c = PyDict_GetItemString(o_txLUT, "s2c");
  PyObject *pd_aw2ali = PyDict_GetItemString(o_txLUT, "aw2ali");

  //-- get the arrays from the dictionaries
  // axLUTs
  PyArrayObject *p_li2rno = NULL, *p_li2sn1 = NULL, *p_li2sn = NULL;
  PyArrayObject *p_li2nos = NULL, *p_li2rng = NULL;
  p_li2rno = (PyArrayObject *)PyArray_FROM_OTF(pd_li2rno, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_li2sn1 = (PyArrayObject *)PyArray_FROM_OTF(pd_li2sn1, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_li2sn = (PyArrayObject *)PyArray_FROM_OTF(pd_li2sn, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_li2nos = (PyArrayObject *)PyArray_FROM_OTF(pd_li2nos, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_li2rng = (PyArrayObject *)PyArray_FROM_OTF(pd_li2rng, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  // sino to crystal, crystals
  PyArrayObject *p_s2c = NULL, *p_crs = NULL, *p_aw2ali = NULL;
  p_s2c = (PyArrayObject *)PyArray_FROM_OTF(pd_s2c, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_crs = (PyArrayObject *)PyArray_FROM_OTF(pd_crs, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  p_aw2ali = (PyArrayObject *)PyArray_FROM_OTF(pd_aw2ali, NPY_INT32, NPY_ARRAY_IN_ARRAY);

  // subsets if using e.g., OSEM
  PyArrayObject *p_subs = NULL;
  p_subs = (PyArrayObject *)PyArray_FROM_OTF(o_subs, NPY_INT32, NPY_ARRAY_IN_ARRAY);
  //--

  /* If that didn't work, throw an exception. */
  if (p_li2rno == NULL || p_li2sn == NULL || p_li2sn1 == NULL || p_li2nos == NULL ||
      p_aw2ali == NULL || p_s2c == NULL || !o_sino || p_crs == NULL || p_subs == NULL ||
      p_li2rng == NULL || !o_bimg) {
    // axLUTs
    Py_XDECREF(p_li2rno);
    Py_XDECREF(p_li2sn);
    Py_XDECREF(p_li2sn1);
    Py_XDECREF(p_li2nos);
    Py_XDECREF(p_li2rng);

    // 2D sino LUT
    Py_XDECREF(p_aw2ali);
    // sino 2 crystals
    Py_XDECREF(p_s2c);
    Py_XDECREF(p_crs);
    // subset definition object
    Py_XDECREF(p_subs);

    return NULL;
  }

  int *subs_ = (int *)PyArray_DATA(p_subs);
  short *s2c = (short *)PyArray_DATA(p_s2c);
  int *aw2ali = (int *)PyArray_DATA(p_aw2ali);
  short *li2sn;
  if (Cnt.SPN == 11) {
    li2sn = (short *)PyArray_DATA(p_li2sn);
  } else if (Cnt.SPN == 1) {
    li2sn = (short *)PyArray_DATA(p_li2sn1);
  }
  char *li2nos = (char *)PyArray_DATA(p_li2nos);
  float *li2rng = (float *)PyArray_DATA(p_li2rng);
  float *crs = (float *)PyArray_DATA(p_crs);

  int Nprj = PyArray_DIM(p_subs, 0);
  int N0crs = PyArray_DIM(p_crs, 0);
  int N1crs = PyArray_DIM(p_crs, 1);
  int Naw = PyArray_DIM(p_aw2ali, 0);

  int *subs;
  if (subs_[0] == -1) {
    Nprj = AW;
    if (Cnt.LOG <= LOGDEBUG)
      printf("\ni> no subsets defined.  number of projection bins in 2D: %d\n", Nprj);
    // all projections in
    subs = (int *)malloc(Nprj * sizeof(int));
    for (int i = 0; i < Nprj; i++) { subs[i] = i; }
  } else {
    if (Cnt.LOG <= LOGDEBUG)
      printf("\ni> subsets defined.  number of subset projection bins in 2D: %d\n", Nprj);
    subs = subs_;
  }

  if (Cnt.LOG <= LOGDEBUG)
    printf("i> back-projection image dimensions: %ld, %ld, %ld\n", o_bimg->shape[0],
           o_bimg->shape[1], o_bimg->shape[2]);

  // sets the device on which to calculate
  HANDLE_ERROR(cudaSetDevice(Cnt.DEVID));

  //<><><<><><><><><><><><><><><><><><><><><<><><><><<><><><><><><><><><><><><><><><><><<><><><><><><>
  gpu_bprj(o_bimg->vec.data(), o_sino->vec.data(), li2rng, li2sn, li2nos, s2c, aw2ali, crs, subs,
           Nprj, Naw, N0crs, Cnt);
  //<><><><><><><><><><><>><><><><><><><><><<><><><><<><><><><><><><><><><><><><><><><><<><><><><><><>

  // Clean up
  Py_DECREF(p_li2rno);
  Py_DECREF(p_li2rng);
  Py_DECREF(p_li2sn);
  Py_DECREF(p_li2sn1);
  Py_DECREF(p_li2nos);
  Py_DECREF(p_aw2ali);
  Py_DECREF(p_s2c);
  Py_DECREF(p_crs);
  Py_DECREF(p_subs);

  if (subs_[0] == -1) free(subs);

  Py_INCREF(Py_None);
  return Py_None;
}

//==============================================================================
// O S E M   R E C O N S T R U C T I O N
//------------------------------------------------------------------------------
static PyObject *osem_rec(PyObject *self, PyObject *args) {
  // Structure of constants
  Cnst Cnt;

  // output image
  PyObject *o_imgout;

  // output image mask
  PyObject *o_rcnmsk;

  // Dictionary of scanner constants
  PyObject *o_mmrcnst;

  // axial LUT dicionary. contains such LUTs: li2rno, li2sn, li2nos.
  PyObject *o_axLUT;

  // transaxial LUT dictionary (e.g., 2D sino where dead bins are out).
  PyObject *o_txLUT;

  // subsets for OSEM, first the default
  PyObject *o_subs;

  // separable kernel matrix, for x, y, and z dimensions
  PyObject *o_krnl;

  // sinos using in reconstruction (reshaped for GPU execution)
  PyObject *o_psng; // prompts (measured)
  PyObject *o_rsng; // randoms
  PyObject *o_ssng; // scatter
  PyObject *o_nsng; // norm
  PyObject *o_asng; // attenuation

  // sensitivity image
  PyObject *o_imgsens;

  /* ^^^^^^^^^^^^^^^^^^^^^^^ Parse the input tuple ^^^^^^^^^^^^^^^^^^^^^^^^^^^ */
  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOO", &o_imgout, &o_psng, &o_rsng, &o_ssng, &o_nsng,
                        &o_asng, &o_subs, &o_imgsens, &o_rcnmsk, &o_krnl, &o_txLUT, &o_axLUT,
                        &o_mmrcnst))
    return NULL;
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  PyObject *pd_log = PyDict_GetItemString(o_mmrcnst, "LOG");
  Cnt.LOG = (char)PyLong_AsLong(pd_log);
  PyObject *pd_span = PyDict_GetItemString(o_mmrcnst, "SPN");
  Cnt.SPN = (char)PyLong_AsLong(pd_span);
  PyObject *pd_sigma_rm = PyDict_GetItemString(o_mmrcnst, "SIGMA_RM");
  Cnt.SIGMA_RM = (float)PyFloat_AsDouble(pd_sigma_rm);
  PyObject *pd_devid = PyDict_GetItemString(o_mmrcnst, "DEVID");
  Cnt.DEVID = (char)PyLong_AsLong(pd_devid);

  /* Interpret the input objects as numpy arrays. */
  // axial LUTs:
  PyObject *pd_li2rno = PyDict_GetItemString(o_axLUT, "li2rno");
  PyObject *pd_li2sn = PyDict_GetItemString(o_axLUT, "li2sn");
  PyObject *pd_li2sn1 = PyDict_GetItemString(o_axLUT, "li2sn1");
  PyObject *pd_li2nos = PyDict_GetItemString(o_axLUT, "li2nos");
  PyObject *pd_li2rng = PyDict_GetItemString(o_axLUT, "li2rng");
  // transaxial sino LUTs:
  PyObject *pd_crs = PyDict_GetItemString(o_txLUT, "crs");
  PyObject *pd_s2c = PyDict_GetItemString(o_txLUT, "s2c");
  PyObject *pd_aw2ali = PyDict_GetItemString(o_txLUT, "aw2ali");

  //-- get the arrays from the dictionaries
  // output back-projection image
  PyArrayObject *p_imgout = NULL;
  p_imgout = (PyArrayObject *)PyArray_FROM_OTF(o_imgout, NPY_FLOAT32, NPY_ARRAY_INOUT_ARRAY2);

  // image mask
  PyArrayObject *p_rcnmsk = NULL;
  p_rcnmsk = (PyArrayObject *)PyArray_FROM_OTF(o_rcnmsk, NPY_BOOL, NPY_ARRAY_IN_ARRAY);

  // sensitivity image
  PyArrayObject *p_imgsens = NULL;
  p_imgsens = (PyArrayObject *)PyArray_FROM_OTF(o_imgsens, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  //> PSF kernel
  PyArrayObject *p_krnl = NULL;
  p_krnl = (PyArrayObject *)PyArray_FROM_OTF(o_krnl, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  //> sinogram objects
  PyArrayObject *p_psng = NULL, *p_rsng = NULL, *p_ssng = NULL, *p_nsng = NULL, *p_asng = NULL;
  p_psng = (PyArrayObject *)PyArray_FROM_OTF(o_psng, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
  p_rsng = (PyArrayObject *)PyArray_FROM_OTF(o_rsng, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  p_ssng = (PyArrayObject *)PyArray_FROM_OTF(o_ssng, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  p_nsng = (PyArrayObject *)PyArray_FROM_OTF(o_nsng, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  p_asng = (PyArrayObject *)PyArray_FROM_OTF(o_asng, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  // subset definition
  PyArrayObject *p_subs = NULL;
  p_subs = (PyArrayObject *)PyArray_FROM_OTF(o_subs, NPY_INT32, NPY_ARRAY_IN_ARRAY);

  // axLUTs
  PyArrayObject *p_li2rno = NULL, *p_li2sn1 = NULL, *p_li2sn = NULL;
  PyArrayObject *p_li2nos = NULL, *p_li2rng = NULL;
  p_li2rno = (PyArrayObject *)PyArray_FROM_OTF(pd_li2rno, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_li2sn = (PyArrayObject *)PyArray_FROM_OTF(pd_li2sn, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_li2sn1 = (PyArrayObject *)PyArray_FROM_OTF(pd_li2sn1, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_li2nos = (PyArrayObject *)PyArray_FROM_OTF(pd_li2nos, NPY_INT8, NPY_ARRAY_IN_ARRAY);
  p_li2rng = (PyArrayObject *)PyArray_FROM_OTF(pd_li2rng, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

  // 2D sino index LUT:
  PyArrayObject *p_aw2ali = NULL;
  p_aw2ali = (PyArrayObject *)PyArray_FROM_OTF(pd_aw2ali, NPY_INT32, NPY_ARRAY_IN_ARRAY);

  // sino to crystal, crystals
  PyArrayObject *p_s2c = NULL, *p_crs = NULL;
  p_s2c = (PyArrayObject *)PyArray_FROM_OTF(pd_s2c, NPY_INT16, NPY_ARRAY_IN_ARRAY);
  p_crs = (PyArrayObject *)PyArray_FROM_OTF(pd_crs, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  //--

  /* If that didn't work, throw an exception. */
  if (p_imgout == NULL || p_rcnmsk == NULL || p_subs == NULL || p_psng == NULL || p_rsng == NULL ||
      p_ssng == NULL || p_nsng == NULL || p_asng == NULL || p_imgsens == NULL ||
      p_li2rno == NULL || p_li2sn == NULL || p_li2sn1 == NULL || p_li2nos == NULL ||
      p_aw2ali == NULL || p_s2c == NULL || p_crs == NULL || p_krnl == NULL) {
    //> output image
    PyArray_DiscardWritebackIfCopy(p_imgout);
    Py_XDECREF(p_imgout);

    Py_XDECREF(p_rcnmsk);

    //>  objects in the sinogram space
    Py_XDECREF(p_psng);
    Py_XDECREF(p_rsng);
    Py_XDECREF(p_ssng);
    Py_XDECREF(p_nsng);
    Py_XDECREF(p_asng);

    //> subsets
    Py_XDECREF(p_subs);

    //> objects in the image space
    Py_XDECREF(p_imgsens);
    Py_XDECREF(p_krnl);

    //> axLUTs
    Py_XDECREF(p_li2rno);
    Py_XDECREF(p_li2sn);
    Py_XDECREF(p_li2sn1);
    Py_XDECREF(p_li2nos);
    //> 2D sinogram LUT
    Py_XDECREF(p_aw2ali);
    //> sinogram to crystal LUTs
    Py_XDECREF(p_s2c);
    Py_XDECREF(p_crs);

    return NULL;
  }

  float *imgout = (float *)PyArray_DATA(p_imgout);
  bool *rcnmsk = (bool *)PyArray_DATA(p_rcnmsk);
  unsigned short *psng = (unsigned short *)PyArray_DATA(p_psng);
  float *rsng = (float *)PyArray_DATA(p_rsng);
  float *ssng = (float *)PyArray_DATA(p_ssng);
  float *nsng = (float *)PyArray_DATA(p_nsng);
  float *asng = (float *)PyArray_DATA(p_asng);

  //> sensitivity image
  float *imgsens = (float *)PyArray_DATA(p_imgsens);

  //>--- PSF KERNEL ---
  float *krnl;
  int SZ_KRNL = (int)PyArray_DIM(p_krnl, 1);
  if (Cnt.LOG <= LOGDEBUG) printf("d> kernel size [voxels]: %d\n", SZ_KRNL);

  if (SZ_KRNL != KERNEL_LENGTH) {
    if (Cnt.LOG <= LOGWARNING) printf("w> wrong kernel size.\n");
    krnl = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    krnl[0] = -1;
  } else {
    krnl = (float *)PyArray_DATA(p_krnl);
  }
  //>-------------------

  short *li2sn;
  if (Cnt.SPN == 11) {
    li2sn = (short *)PyArray_DATA(p_li2sn);
  } else if (Cnt.SPN == 1) {
    li2sn = (short *)PyArray_DATA(p_li2sn1);
  }
  char *li2nos = (char *)PyArray_DATA(p_li2nos);
  float *li2rng = (float *)PyArray_DATA(p_li2rng);
  float *crs = (float *)PyArray_DATA(p_crs);
  short *s2c = (short *)PyArray_DATA(p_s2c);
  int *aw2ali = (int *)PyArray_DATA(p_aw2ali);

  int N0crs = PyArray_DIM(p_crs, 0);
  int N1crs = PyArray_DIM(p_crs, 1);

  // number of subsets
  int Nsub = PyArray_DIM(p_subs, 0);
  // number of elements used to store max. number of subsets projection - 1
  int Nprj = PyArray_DIM(p_subs, 1);
  if (Cnt.LOG <= LOGDEBUG)
    printf("i> number of subsets = %d, and max. number of projections/subset = %d\n", Nsub,
           Nprj - 1);

  int *subs = (int *)PyArray_DATA(p_subs);

  // sets the device on which to calculate
  HANDLE_ERROR(cudaSetDevice(Cnt.DEVID));

  //<><><<><><><><<><><><><><><><><><><>
  osem(imgout, rcnmsk, psng, rsng, ssng, nsng, asng, subs, imgsens, krnl, li2rng, li2sn, li2nos,
       s2c, crs, Nsub, Nprj, N0crs, Cnt);
  //<><><><><><><><<><><><>><><><><><><>

  // Clean up
  PyArray_ResolveWritebackIfCopy(p_imgout);
  Py_DECREF(p_imgout);

  Py_DECREF(p_rcnmsk);
  Py_DECREF(p_psng);
  Py_DECREF(p_rsng);
  Py_DECREF(p_ssng);
  Py_DECREF(p_nsng);
  Py_DECREF(p_asng);

  Py_DECREF(p_subs);

  Py_DECREF(p_imgsens);
  Py_DECREF(p_krnl);

  Py_DECREF(p_li2rno);
  Py_DECREF(p_li2rng);
  Py_DECREF(p_li2sn);
  Py_DECREF(p_li2sn1);
  Py_DECREF(p_li2nos);
  Py_DECREF(p_aw2ali);
  Py_DECREF(p_s2c);
  Py_DECREF(p_crs);

  Py_INCREF(Py_None);
  return Py_None;
}
