"""Image reconstruction from raw PET data"""
import logging
import os
import time
from collections import namedtuple
from collections.abc import Iterable
from numbers import Real

import cuvec as cu
import numpy as np
import scipy.ndimage as ndi
from tqdm.auto import trange

from niftypet import nimpa

# resources contain isotope info
from .. import mmraux, mmrnorm, resources
from ..img import mmrimg
from ..lm.mmrhist import randoms
from ..sct import vsm
from . import petprj

log = logging.getLogger(__name__)

# reconstruction mode:
# 0 - no attenuation  and  no scatter
# 1 - attenuation  and   no scatter
# 2 - attenuation and scatter given as input parameter
# 3 - attenuation  and  scatter
recModeStr = ['_noatt_nosct_', '_nosct_', '_noatt_', '_', '_ute_']


# fwhm in [mm]
def fwhm2sig(fwhm, voxsize=1.):
    return (fwhm/voxsize) / (2 * (2 * np.log(2))**.5)


# ========================================================================
# OSEM RECON
# ------------------------------------------------------------------------


def get_subsets14(n, params):
    '''Define the n-th subset out of 14 in the transaxial projection space'''
    Cnt = params['Cnt']
    txLUT = params['txLUT']

    # just for check of sums (have to be equal for all subsets to make them balanced)
    aisum = np.sum(txLUT['msino'], axis=0)
    # number of subsets
    N = 14
    # projections per subset
    P = Cnt['NSANGLES'] // N
    # the remaining projections which have to be spread over the N subsets with a given frequency
    fs = N / float(P - N)
    # generate sampling pattern for subsets up to N out of P
    sp = np.array([np.arange(i, Cnt['NSANGLES'], P) for i in range(N)])
    # ======================================
    S = np.zeros((N, P), dtype=np.int16)
    # ======================================
    # sum of sino angle projections
    totsum = np.zeros(N, dtype=np.int32)
    # iterate subset (which is also the angle iterator within block b)
    for s in range(N):
        # list of sino angular indexes for a given subset
        si = []
        # ::::: iterate sino blocks.
        # This bit may be unnecessary, it can be taken directly from sp array
        for b in range(N):
            # --angle index within a sino block depending on subset s
            ai = (s+b) % N
            # --angle index for whole sino
            sai = sp[ai, b]
            si.append(sai)
            totsum[s] += aisum[sai]
        # :::::
        # deal with the remaining part, ie, P-N per block
        rai = np.int16(np.floor(np.arange(s, 2 * N, fs)[:4] % N))
        for i in range(P - N):
            sai = sp[-1, rai[i]] + i + 1
            totsum[s] += aisum[sai]
            si.append(sai)
        S[s] = np.array((si))

    # get the projection bin index for transaxial gpu sinos
    tmsk = txLUT['msino'] > 0
    Smsk = -1 * np.ones(tmsk.shape, dtype=np.int32)
    Smsk[tmsk] = list(range(Cnt['Naw']))

    iprj = Smsk[:, S[n]]
    iprj = iprj[iprj >= 0]

    return iprj, S


def psf_config(psf, Cnt):
    '''
    Generate separable PSF kernel (x, y, z) based on FWHM for x, y, z

    Args:
      psf:
        None: PSF reconstruction is switched off
        'measured': PSF based on measurement (line source in air)
        float: an isotropic PSF with the FWHM defined by the float or int scalar
        [x, y, z]: list or Numpy array of separate FWHM of the PSF for each direction
        ndarray: 3 x 2*RSZ_PSF_KRNL+1 Numpy array directly defining the kernel in each direction
    '''
    def _config(fwhm3, check_len=True):
        # resolution modelling by custom kernels
        if check_len:
            if len(fwhm3) != 3 or any(f < 0 for f in fwhm3):
                raise ValueError('Incorrect separable kernel FWHM definition')

        kernel = np.empty((3, 2 * Cnt['RSZ_PSF_KRNL'] + 1), dtype=np.float32)
        for i, psf in enumerate(fwhm3):
            # > FWHM -> sigma conversion for all dimensions separately
            if i == 2:
                sig = fwhm2sig(psf, voxsize=Cnt['SZ_VOXZ'] * 10)
            else:
                sig = fwhm2sig(psf, voxsize=Cnt['SZ_VOXY'] * 10)

            x = np.arange(-Cnt['RSZ_PSF_KRNL'], Cnt['RSZ_PSF_KRNL'] + 1)
            kernel[i, :] = np.exp(-0.5 * (x**2 / sig**2))
            kernel[i, :] /= np.sum(kernel[i, :])

        psfkernel = np.empty((3, 2 * Cnt['RSZ_PSF_KRNL'] + 1), dtype=np.float32)
        psfkernel[0, :] = kernel[2, :]
        psfkernel[1, :] = kernel[0, :]
        psfkernel[2, :] = kernel[1, :]

        return psfkernel

    if psf is None:
        psfkernel = _config([], False)
        # switch off PSF reconstruction by setting negative first element
        psfkernel[0, 0] = -1
    elif psf == 'measured':
        psfkernel = nimpa.psf_measured(scanner='mmr', scale=1)
    elif isinstance(psf, Real):
        psfkernel = _config([psf] * 3)
    elif isinstance(psf, Iterable):
        psf = np.asanyarray(psf)
        if psf.shape == (3, 2 * Cnt['RSZ_PSF_KRNL'] + 1):
            psfkernel = _config([], False)
            psfkernel[0, :] = psf[2, :]
            psfkernel[1, :] = psf[0, :]
            psfkernel[2, :] = psf[1, :]
        elif len(psf) == 3:
            psfkernel = _config(psf)
        else:
            raise ValueError(f"invalid PSF dimensions ({psf.shape})")
    else:
        raise ValueError(f"unrecognised PSF definition ({psf})")
    return psfkernel


def osemone(datain, mumaps, hst, scanner_params, recmod=3, itr=4, fwhm=0., psf=None,
            mask_radius=29., decay_ref_time=None, attnsino=None, sctsino=None, randsino=None,
            normcomp=None, emmskS=False, frmno='', fcomment='', outpath=None, store_img=False,
            store_itr=None, ret_sinos=False):
    '''
    OSEM image reconstruction with several modes
    (with/without scatter and/or attenuation correction)

    Args:
      psf: Reconstruction with PSF, passed to `psf_config`
    '''

    # > Get particular scanner parameters: Constants, transaxial and axial LUTs
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    # ---------- sort out OUTPUT ------------
    # -output file name for the reconstructed image
    if outpath is None:
        opth = os.path.join(datain['corepath'], 'reconstructed')
    else:
        opth = outpath

    if store_img is True or store_itr is not None:
        mmraux.create_dir(opth)

    return_ssrb, return_mask = ret_sinos, ret_sinos

    # ----------

    log.info('reconstruction in mode: %d', recmod)

    # get object and hardware mu-maps
    muh, muo = mumaps

    # get the GPU version of the image dims
    mus = mmrimg.convert2dev(muo + muh, Cnt)

    # remove gaps from the prompt sino
    psng = mmraux.remgaps(hst['psino'], txLUT, Cnt)

    # ========================================================================
    # GET NORM
    # -------------------------------------------------------------------------
    if normcomp is None:
        ncmp, _ = mmrnorm.get_components(datain, Cnt)
    else:
        ncmp = normcomp
        log.warning('using user-defined normalisation components')
    nsng = mmrnorm.get_sinog(datain, hst, axLUT, txLUT, Cnt, normcomp=ncmp)
    # ========================================================================

    # ========================================================================
    # ATTENUATION FACTORS FOR COMBINED OBJECT AND BED MU-MAP
    # -------------------------------------------------------------------------
    # > combine attenuation and norm together depending on reconstruction mode
    if recmod == 0:
        asng = np.ones(psng.shape, dtype=np.float32)
    else:
        # > check if the attenuation sino is given as an array
        if isinstance(attnsino, np.ndarray) \
                and attnsino.shape==(Cnt['NSN11'], Cnt['NSANGLES'], Cnt['NSBINS']):
            asng = mmraux.remgaps(attnsino, txLUT, Cnt)
            log.info('using provided attenuation factor sinogram')
        elif isinstance(attnsino, np.ndarray) \
                and attnsino.shape==(Cnt['Naw'], Cnt['NSN11']):
            asng = attnsino
            log.info('using provided attenuation factor sinogram')
        else:
            asng = cu.zeros(psng.shape, dtype=np.float32)
            petprj.fprj(asng.cuvec,
                        cu.asarray(mus).cuvec, txLUT, axLUT, np.array([-1], dtype=np.int32), Cnt,
                        1)
    # > combine attenuation and normalisation
    ansng = asng * nsng
    # ========================================================================

    # ========================================================================
    # Randoms
    # -------------------------------------------------------------------------
    if isinstance(randsino, np.ndarray):
        rsino = randsino
        rsng = mmraux.remgaps(randsino, txLUT, Cnt)
    else:
        rsino, snglmap = randoms(hst, scanner_params)
        rsng = mmraux.remgaps(rsino, txLUT, Cnt)
    # ========================================================================

    # ========================================================================
    # SCAT
    # -------------------------------------------------------------------------
    if recmod == 2:
        if sctsino is not None:
            ssng = mmraux.remgaps(sctsino, txLUT, Cnt)
        elif sctsino is None and os.path.isfile(datain['em_crr']):
            emd = nimpa.getnii(datain['em_crr'])
            ssn = vsm(
                datain,
                mumaps,
                emd['im'],
                scanner_params,
                histo=hst,
                rsino=rsino,
                prcnt_scl=0.1,
                emmsk=False,
            )
            ssng = mmraux.remgaps(ssn, txLUT, Cnt)
        else:
            raise ValueError("No emission image available for scatter estimation! " +
                             " Check if it's present or the path is correct.")
    else:
        ssng = np.zeros(rsng.shape, dtype=rsng.dtype)
    # ========================================================================

    log.info('------ OSEM (%d) -------', itr)
    # ------------------------------------
    Sn = 14   # number of subsets

    # -get one subset to get number of projection bins in a subset
    Sprj, s = get_subsets14(0, scanner_params)
    Nprj = len(Sprj)
    # -init subset array and sensitivity image for a given subset
    sinoTIdx = np.zeros((Sn, Nprj + 1), dtype=np.int32)
    # -init sensitivity images for each subset
    imgsens = np.zeros((Sn, Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)
    for n in range(Sn):
        # first number of projection for the given subset
        sinoTIdx[n, 0] = Nprj
        sinoTIdx[n, 1:], s = get_subsets14(n, scanner_params)
        # sensitivity image
        petprj.bprj(imgsens[n, :, :, :], ansng[sinoTIdx[n, 1:], :], txLUT, axLUT, sinoTIdx[n, 1:],
                    Cnt)
    # -------------------------------------

    # -mask for reconstructed image.  anything outside it is set to zero
    msk = mmrimg.get_cylinder(Cnt, rad=mask_radius, xo=0, yo=0, unival=1, gpu_dim=True) > 0.9

    # -init image
    img = np.ones((Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)

    # -decay correction
    lmbd = np.log(2) / resources.riLUT[Cnt['ISOTOPE']]['thalf']
    if Cnt['DCYCRR'] and 't0' in hst and 'dur' in hst:
        # > decay correct to the reference time (e.g., injection time) if provided
        # > otherwise correct in reference to the scan start time (using the time
        # > past from the start to the start time frame)
        if decay_ref_time is not None:
            tref = decay_ref_time
        else:
            tref = hst['t0']

        dcycrr = np.exp(lmbd * tref) * lmbd * hst['dur'] / (1 - np.exp(-lmbd * hst['dur']))
        # apply quantitative correction to the image
        qf = ncmp['qf'] / resources.riLUT[Cnt['ISOTOPE']]['BF'] / float(hst['dur'])
        qf_loc = ncmp['qf_loc']

    elif not Cnt['DCYCRR'] and 't0' in hst and 'dur' in hst:
        dcycrr = 1.
        # apply quantitative correction to the image
        qf = ncmp['qf'] / resources.riLUT[Cnt['ISOTOPE']]['BF'] / float(hst['dur'])
        qf_loc = ncmp['qf_loc']

    else:
        dcycrr = 1.
        qf = 1.
        qf_loc = 1.

    # -affine matrix for the reconstructed images
    B = mmrimg.image_affine(datain, Cnt)

    # resolution modelling
    psfkernel = psf_config(psf, Cnt)

    # -time it
    stime = time.time()

    # import pdb; pdb.set_trace()

    # ========================================================================
    # OSEM RECONSTRUCTION
    # -------------------------------------------------------------------------
    with trange(itr, desc="OSEM", disable=log.getEffectiveLevel() > logging.INFO,
                leave=log.getEffectiveLevel() <= logging.INFO) as pbar:

        for k in pbar:

            petprj.osem(img, psng, rsng, ssng, nsng, asng, sinoTIdx, imgsens, msk, psfkernel,
                        txLUT, axLUT, Cnt)

            if np.nansum(img) < 0.1:
                log.warning('it seems there is not enough true data to render reasonable image')
                # img[:]=0
                itr = k
                break
            if recmod >= 3 and k < itr - 1 and itr > 1:
                sct_time = time.time()
                sct = vsm(datain, mumaps, mmrimg.convert2e7(img, Cnt), scanner_params, histo=hst,
                          rsino=rsino, emmsk=emmskS, return_ssrb=return_ssrb,
                          return_mask=return_mask)

                if isinstance(sct, dict):
                    ssn = sct['sino']
                else:
                    ssn = sct

                ssng = mmraux.remgaps(ssn, txLUT, Cnt)
                pbar.set_postfix(scatter="%.3gs" % (time.time() - sct_time))
            # save images during reconstruction if requested
            if store_itr and k in store_itr:
                im = mmrimg.convert2e7(img * (dcycrr*qf*qf_loc), Cnt)

                fout = os.path.join(
                    opth, (os.path.basename(datain['lm_bf'])[:16].replace('.','-') +
                           f"{frmno}_t{hst['t0']}-{hst['t1']}sec_itr{k}{fcomment}_inrecon.nii.gz"))
                nimpa.array2nii(im[::-1, ::-1, :], B, fout)

    log.info('recon time: %.3g', time.time() - stime)
    # ========================================================================

    log.info('applying decay correction of: %r', dcycrr)
    log.info('applying quantification factor: %r to the whole image', qf)
    log.info('for the frame duration of: %r', hst['dur'])

    # additional factor for making it quantitative in absolute terms (derived from measurements)
    img *= dcycrr * qf * qf_loc

    # ---- save images -----
    # -first convert to standard mMR image size
    im = mmrimg.convert2e7(img, Cnt)

    # -description text to NIfTI
    # -attenuation number: if only bed present then it is 0.5
    attnum = (1 * (np.sum(muh) > 0.5) + 1 * (np.sum(muo) > 0.5)) / 2.
    descrip = (f"alg=osem"
               f";sub=14"
               f";att={attnum*(recmod>0)}"
               f";sct={1*(recmod>1)}"
               f";spn={Cnt['SPN']}"
               f";itr={itr}"
               f";fwhm=0"
               f";t0={hst['t0']}"
               f";t1={hst['t1']}"
               f";dur={hst['dur']}"
               f";qf={qf}")

    # > file name of the output reconstructed image
    # > (maybe used later even if not stored now)
    fpet = os.path.join(opth, (os.path.basename(datain['lm_bf']).split('.')[0] +
                               f"{frmno}_t{hst['t0']}-{hst['t1']}sec_itr{itr}{fcomment}.nii.gz"))

    if store_img:
        log.info('saving image to: %s', fpet)
        nimpa.array2nii(im[::-1, ::-1, :], B, fpet, descrip=descrip)

    im_smo = None
    fsmo = None
    if fwhm > 0:
        im_smo = ndi.filters.gaussian_filter(im, fwhm2sig(fwhm, voxsize=Cnt['SZ_VOXY'] * 10),
                                             mode='mirror')

        if store_img:
            fsmo = fpet.split('.nii.gz')[0] + '_smo-' + str(fwhm).replace('.', '-') + 'mm.nii.gz'
            log.info('saving smoothed image to: ' + fsmo)
            descrip.replace(';fwhm=0', ';fwhm=str(fwhm)')
            nimpa.array2nii(im_smo[::-1, ::-1, :], B, fsmo, descrip=descrip)

    # returning:
    # (0) E7 image [can be smoothed];
    # (1) file name of saved E7 image
    # (2) [optional] scatter sino
    # (3) [optional] single slice rebinned scatter
    # (4) [optional] mask for scatter scaling based on attenuation data
    # (5) [optional] random sino
    # if ret_sinos and recmod>=3:
    #     recout = namedtuple('recout', 'im, fpet, ssn, sssr, amsk, rsn')
    #     recout.im   = im
    #     recout.fpet = fout
    #     recout.ssn  = ssn
    #     recout.sssr = sssr
    #     recout.amsk = amsk
    #     recout.rsn  = rsino
    # else:
    #     recout = namedtuple('recout', 'im, fpet')
    #     recout.im   = im
    #     recout.fpet = fout

    if ret_sinos and recmod >= 3 and itr > 1:
        RecOut = namedtuple('RecOut', 'im, fpet, imsmo, fsmo, affine, ssn, sssr, amsk, rsn')
        recout = RecOut(im, fpet, im_smo, fsmo, B, ssn, sct['ssrb'], sct['mask'], rsino)
    else:
        RecOut = namedtuple('RecOut', 'im, fpet, imsmo, fsmo, affine')
        recout = RecOut(im, fpet, im_smo, fsmo, B)

    return recout
