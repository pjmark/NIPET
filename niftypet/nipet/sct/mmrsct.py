'''
Voxel-driven scatter modelling for PET data
'''
import logging
import os
import time
from math import pi

import cuvec as cu
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import interp2d
from scipy.special import erfc

from .. import mmr_auxe, mmraux, mmrnorm
from ..img import mmrimg
from ..prj import mmrprj, mmrrec, petprj
from . import nifty_scatter

log = logging.getLogger(__name__)


def fwhm2sig(fwhm, Cnt):
    '''
    Convert FWHM to sigma (standard deviation)
    '''
    return (fwhm / Cnt['SO_VXY']) / (2 * (2 * np.log(2))**.5)


# ======================================================================
# S C A T T E R
# ----------------------------------------------------------------------


def get_scrystals(scanner_params):
    '''
    Get table of selected transaxial and axial (ring) crystals
    used for scatter modelling
    '''
    # > decompose constants, transaxial and axial LUTs are extracted
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    # ------------------------------------------------------
    # > transaxial crystals definitions
    crs = txLUT['crs']

    # > period of scatter crystals (needed for definition)
    SCRS_T = 7

    # > counter for crystal period, SCRS_T
    cntr = 0

    # > scatter crystal index
    iscrs = 0

    # > initialise list of transaxial scatter crystal table
    scrs = []

    # > transaxial scatter crystal selection for modelling
    for c in range(Cnt['NCRS']):
        if (((c+1) % 9) == 0):
            continue
        cntr += 1
        if (cntr == SCRS_T):
            cntr = 0
            scrs.append([c, 0.5 * (crs[c, 0] + crs[c, 2]), 0.5 * (crs[c, 1] + crs[c, 3])])
            iscrs += 1

    # > convert the scatter crystal table to Numpy array
    scrs = np.array(scrs, dtype=np.float32)
    # ------------------------------------------------------

    # ------------------------------------------------------
    # > scatter ring definition (axially)
    sct_irng = np.int16([0, 10, 19, 28, 35, 44, 53, 63])
    # number of scatter rings (used for scatter estimation)
    NSRNG = len(sct_irng)
    # ------------------------------------------------------

    logtxt = ''

    srng = np.zeros((NSRNG, 2), dtype=np.float32)
    for ir in range(NSRNG):
        srng[ir, 0] = float(sct_irng[ir])
        srng[ir, 1] = axLUT['rng'][sct_irng[ir], :].mean()
        logtxt += '> [{}]: ring_i={}, ring_z={}\n'.format(ir, int(srng[ir, 0]), srng[ir, 1])

    log.debug(logtxt)
    return {'scrs': scrs, 'srng': srng, 'sirng': sct_irng, 'NSCRS': scrs.shape[0], 'NSRNG': NSRNG}


# ======================================================================
def get_sctlut2d(txLUT, scrs_def):

    # > scatter to sinogram bin index LUT
    sct2aw = np.zeros(scrs_def['NSCRS'] * scrs_def['NSCRS'], dtype=np.int32)

    # scatter/unscattered crystal x-coordinate (used for determining +/- sino segments)
    xsxu = np.zeros((scrs_def['NSCRS'], scrs_def['NSCRS']), dtype=np.int8)

    scrs = scrs_def['scrs']
    # > loop over unscattered crystals
    for uc in range(scrs_def['NSCRS']):
        # > loop over scatter crystals
        for sc in range(scrs_def['NSCRS']):
            # > sino linear index (full including any gaps)
            # > scrs is a 2D array of rows [sct_crs_idx, mid_x, mid_y]
            sct2aw[scrs_def['NSCRS'] * uc + sc] = txLUT['c2sFw'][int(scrs[uc, 0]),
                                                                 int(scrs[sc, 0])]
            # > scattered and unscattered crystal positions
            # (used for determining +/- sino segments)
            if scrs[sc, 1] > scrs[uc, 1]:
                xsxu[uc, sc] = 1

    sct2aw.shape = scrs_def['NSCRS'], scrs_def['NSCRS']
    return {'sct2aw': sct2aw, 'xsxu': xsxu, 'c2sFw': txLUT['c2sFw']}


# ======================================================================


def get_knlut(Cnt):
    '''
    get Klein-Nishina LUTs
    '''

    SIG511 = Cnt['ER'] * Cnt['E511'] / 2.35482

    CRSSavg = (2 * (4/3.0 - np.log(3)) + .5 * np.log(3) - 4/9.0)

    COSSTP = (1 - Cnt['COSUPSMX']) / (Cnt['NCOS'] - 1)

    log.debug('using these scatter constants:\nCOS(UPSMAX) = {},\nCOSSTP = {}'.format(
        Cnt['COSUPSMX'], COSSTP))

    knlut = np.zeros((Cnt['NCOS'], 2), dtype=np.float32)

    for i in range(Cnt['NCOS']):
        cosups = Cnt['COSUPSMX'] + i*COSSTP
        alpha = 1 / (2-cosups)
        KNtmp = ((0.5 * Cnt['R02']) * alpha * alpha * (alpha + 1/alpha - (1 - cosups*cosups)))
        knlut[i, 0] = KNtmp / (2 * pi * Cnt['R02'] * CRSSavg)
        knlut[i, 1] = ((1+alpha) / (alpha*alpha) *
                       (2 * (1+alpha) /
                        (1 + 2*alpha) - 1 / alpha * np.log(1 + 2*alpha)) + np.log(1 + 2*alpha) /
                       (2*alpha) - (1 + 3*alpha) / ((1 + 2*alpha) * (1 + 2*alpha))) / CRSSavg

        # Add energy resolution:
        if Cnt['ER'] > 0:
            log.info('using energy resolution for scatter simulation, ER = {}'.format(Cnt['ER']))
            knlut[i, 0] *= .5 * erfc(
                (Cnt['LLD'] - alpha * Cnt['E511']) / (SIG511 * np.sqrt(2 * alpha)))
            # knlut[i,0] *= .5*erfc( (Cnt['LLD']-alpha*Cnt['E511'])/(SIG511) );

        # for large angles (small cosups)
        # when the angle in GPU calculations is greater than COSUPSMX
        if i == 0:
            knlut[0, 0] = 0

    return knlut


# ======================================================================


# =================================================================================================
# GET SCATTER LUTs
# -------------------------------------------------------------------------------------------------
def rd2sni(offseg, r1, r0):
    rd = np.abs(r1 - r0)
    rdi = (2*rd - 1 * (r1 > r0))
    sni = offseg[rdi] + np.minimum(r0, r1)
    return sni


# -------------------------------------------------------------------------------------------------


def get_sctLUT(scanner_params):

    # > decompose constants, transaxial and axial LUTs are extracted
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']

    # > get the Klein-Nishina LUT:
    KN = get_knlut(Cnt)

    # > get scatter crystal tables:
    scrs_def = get_scrystals(scanner_params)

    # > get 2D scatter LUT (for transaxial sinograms)
    sctlut2d = get_sctlut2d(txLUT, scrs_def)

    # get the indexes of rings used for scatter estimation
    irng = scrs_def['sirng']

    # get number of ring accounting for the possible ring reduction (to save computation time)
    # NRNG = Cnt['RNG_END']-Cnt['RNG_STRT']

    # -span-1 LUT (rings to sino index)
    seg = np.append([Cnt['NRNG']], np.ceil(np.arange(Cnt['NRNG'] - 1, 0, -.5)).astype(np.int16))
    offseg = np.int16(np.append([0], np.cumsum(seg)))

    # -3D scatter sino LUT. axial component based on michelogram.
    sctaxR = np.zeros((Cnt['NRNG']**2, 4), dtype=np.int32)
    sctaxW = np.zeros((Cnt['NRNG']**2, 4), dtype=np.float32)

    # -just for local check and display of the interpolation at work
    mich = np.zeros((Cnt['NRNG'], Cnt['NRNG']), dtype=np.float32)
    mich2 = np.zeros((Cnt['NRNG'], Cnt['NRNG']), dtype=np.float32)

    J, I = np.meshgrid(irng, irng)                           # NOQA: E741
    mich[J, I] = np.reshape(np.arange(scrs_def['NSRNG']**2),
                            (scrs_def['NSRNG'], scrs_def['NSRNG']))

    # plt.figure(64)
    # plt.imshow(mich, interpolation='none')

    for r1 in range(Cnt['RNG_STRT'], Cnt['RNG_END']):
        # border up and down
        bd = next(idx for idx in irng if idx >= r1)
        bu = next(idx for idx in irng[::-1] if idx <= r1)
        for r0 in range(Cnt['RNG_STRT'], Cnt['RNG_END']):

            # if (np.abs(r1-r0)>MRD):
            #     continue
            # border left and right
            br = next(idx for idx in irng if idx >= r0)
            bl = next(idx for idx in irng[::-1] if idx <= r0)
            # print '(r0,r1)=', r0,r1, '(bl,br,bu,bd)', bl,br,bu,bd

            # span-1 sino index (sni) creation:
            sni = rd2sni(offseg, r1, r0)

            # see: https://en.wikipedia.org/wiki/Bilinear_interpolation
            if (br == bl) and (bu != bd):

                sctaxR[sni, 0] = rd2sni(offseg, bd, r0)
                sctaxW[sni, 0] = (r1-bu) / float(bd - bu)
                sctaxR[sni, 1] = rd2sni(offseg, bu, r0)
                sctaxW[sni, 1] = (bd-r1) / float(bd - bu)

                mich2[r1, r0] = mich[bd, r0] * sctaxW[sni, 0] + mich[bu, r0] * sctaxW[sni, 1]

            elif (bu == bd) and (br != bl):

                sctaxR[sni, 0] = rd2sni(offseg, r1, bl)
                sctaxW[sni, 0] = (br-r0) / float(br - bl)
                sctaxR[sni, 1] = rd2sni(offseg, r1, br)
                sctaxW[sni, 1] = (r0-bl) / float(br - bl)

                mich2[r1, r0] = mich[r1, bl] * sctaxW[sni, 0] + mich[r1, br] * sctaxW[sni, 1]

            elif (bu == bd) and (br == bl):

                mich2[r1, r0] = mich[r1, r0]
                sctaxR[sni, 0] = rd2sni(offseg, r1, r0)
                sctaxW[sni, 0] = 1
                continue

            else:

                cf = float(((br-bl) * (bd-bu)))

                sctaxR[sni, 0] = rd2sni(offseg, bd, bl)
                sctaxW[sni, 0] = (br-r0) * (r1-bu) / cf
                sctaxR[sni, 1] = rd2sni(offseg, bd, br)
                sctaxW[sni, 1] = (r0-bl) * (r1-bu) / cf

                sctaxR[sni, 2] = rd2sni(offseg, bu, bl)
                sctaxW[sni, 2] = (br-r0) * (bd-r1) / cf
                sctaxR[sni, 3] = rd2sni(offseg, bu, br)
                sctaxW[sni, 3] = (r0-bl) * (bd-r1) / cf

                mich2[r1, r0] = mich[bd, bl] * sctaxW[sni, 0] + mich[bd, br] * sctaxW[
                    sni, 1] + mich[bu, bl] * sctaxW[sni, 2] + mich[bu, br] * sctaxW[sni, 3]

    # plt.figure(65), plt.imshow(mich2, interpolation='none')

    sctLUT = {
        'sctaxR': sctaxR, 'sctaxW': sctaxW, 'offseg': offseg, 'KN': KN, 'mich_chck': [mich, mich2],
        **scrs_def, **sctlut2d}

    return sctLUT


# ------------------------------------------------------------------------------------------------
# S C A T T E R    I N T E R P O L A T I O N
# ------------------------------------------------------------------------------------------------


# =============================================================================
def intrp_bsct(sct3d, Cnt, sctLUT, ssrlut, dtype=np.float32):
    '''
    interpolate the basic scatter distributions which are then
    transferred into the scatter sinograms.
    '''

    # > number of sinograms
    snno = sct3d.shape[1]

    i_scrs = sctLUT['scrs'][:, 0].astype(int)

    x = i_scrs
    y = np.append([-1], i_scrs)
    xnew = np.arange(Cnt['NCRS'])
    ynew = np.arange(Cnt['NCRS'])

    # > advanced indexing matrix for rolling the non-interpolated results
    jj, ii = np.mgrid[0:sctLUT['NSCRS'], 0:sctLUT['NSCRS']]

    # > roll each row according to the position
    for i in range(sctLUT['NSCRS']):
        ii[i, :] = np.roll(ii[i, :], -1 * i)

    jjnew, iinew = np.mgrid[0:Cnt['NCRS'], 0:Cnt['NCRS']]
    for i in range(Cnt['NCRS']):
        iinew[i, :] = np.roll(iinew[i, :], i)

    ssn = np.zeros((Cnt['TOFBINN'], snno, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=dtype)
    sssr = np.zeros((Cnt['TOFBINN'], Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=dtype)

    for ti in range(Cnt['TOFBINN']):
        sn2d = np.zeros(Cnt['NSANGLES'] * Cnt['NSBINS'], dtype=dtype)

        for si in range(snno):

            sn2d[:] = 0

            sct2d = sct3d[0, si, jj, ii]

            z = np.vstack([sct2d[-1, :], sct2d])
            f = interp2d(x, y, z, kind='cubic')
            znew = f(xnew, ynew)

            # unroll
            znew = znew[jjnew, iinew]

            # > upper triangle
            # > add '1' to include index zero (distinguished from after triangulation)
            qi = np.triu(sctLUT['c2sFw'] + 1) > 0
            sidx = sctLUT['c2sFw'][qi]
            s = znew[qi]
            sn2d[sidx] = s

            # > lower triangle
            qi = np.tril(sctLUT['c2sFw'] + 1) > 0
            sidx = sctLUT['c2sFw'][qi]
            s = znew[qi]
            sn2d[sidx] += s

            ssn[ti, si, ...] = np.reshape(sn2d, (Cnt['NSANGLES'], Cnt['NSBINS']))
            sssr[ti, ssrlut[si], ...] += ssn[ti, si, :, :]

    return np.squeeze(ssn), np.squeeze(sssr)
    # -------------------------------------------------


# ===================================================================================================


def vsm(
    datain,
    mumaps,
    em,
    scanner_params,
    histo=None,
    rsino=None,
    prcnt_scl=0.1,
    fwhm_input=0.42,
    mask_threshlod=0.999,
    snmsk=None,
    emmsk=False,
    interpolate=True,
    return_uninterp=False,
    return_ssrb=False,
    return_mask=False,
    return_scaling=False,
    scaling=True,
    self_scaling=False,
    save_sax=False,
):
    '''
    Voxel-driven scatter modelling (VSM).
    Obtain a scatter sinogram using the mu-maps (hardware and object mu-maps)
    an estimate of emission image, the prompt measured sinogram, an
    estimate of the randoms sinogram and a normalisation sinogram.
    Input:
        - datain:       Contains the data used for scatter-specific detector
                        normalisation.  May also include the non-corrected
                        emission image used for masking, when requested.
        - mumaps:       A tuple of hardware and object mu-maps (in this order).
        - em:           An estimate of the emission image.
        - histo:          Dictionary containing the histogrammed measured data into
                        sinograms.
        - rsino:       Randoms sinogram (3D).  Needed for proper scaling of
                        scatter to the prompt data.
        - scanner_params: Scanner specific parameters.
        - prcnt_scl:    Ratio of the maximum scatter intensities below which the
                        scatter is not used for fitting it to the tails of prompt
                        data.  Default is 10%.
        - emmsk:        When 'True' it will use uncorrected emission image for
                        masking the sources (voxels) of photons to be used in the
                        scatter modelling.
        - scaling:      performs scaling to the data (sinogram)
        - self_scaling: Scaling is performed on span-1 without the help of SSR
                        scaling and using the sax factors (scatter axial factors).
                        If False (default), the sax factors have to be provided.
        - sax:          Scatter axial factors used for scaling with SSR sinograms.

    '''

    # > decompose constants, transaxial and axial LUTs are extracted
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    if self_scaling:
        scaling = True

    # > decompose mu-maps
    muh, muo = mumaps

    if emmsk and not os.path.isfile(datain['em_nocrr']):
        log.info('reconstructing emission data without scatter and attenuation corrections'
                 ' for mask generation...')
        recnac = mmrrec.osemone(datain, mumaps, histo, scanner_params, recmod=0, itr=3, fwhm=2.0,
                                store_img=True)
        datain['em_nocrr'] = recnac.fpet

    # if rsino is None and not histo is None and 'rsino' in histo:
    #     rsino = histo['rsino']

    # > if histogram data or randoms sinogram not given, then no scaling or normalisation
    if (histo is None) or (rsino is None):
        scaling = False

    # -get the normalisation components
    nrmcmp, nhdr = mmrnorm.get_components(datain, Cnt)

    # -smooth for defining the sino scatter only regions
    if fwhm_input > 0.:
        mu_sctonly = ndi.filters.gaussian_filter(mmrimg.convert2dev(muo, Cnt),
                                                 fwhm2sig(fwhm_input, Cnt), mode='mirror')
    else:
        mu_sctonly = muo

    if Cnt['SPN'] == 1:
        snno = Cnt['NSN1']
        snno_ = Cnt['NSN64']
        ssrlut = axLUT['sn1_ssrb']
        saxnrm = nrmcmp['sax_f1']
    elif Cnt['SPN'] == 11:
        snno = Cnt['NSN11']
        snno_ = snno
        ssrlut = axLUT['sn11_ssrb']
        saxnrm = nrmcmp['sax_f11']

    # LUTs for scatter
    sctLUT = get_sctLUT(scanner_params)

    # > smooth before scaling/down-sampling the mu-map and emission images
    if fwhm_input > 0.:
        muim = ndi.filters.gaussian_filter(muo + muh, fwhm2sig(fwhm_input, Cnt), mode='mirror')
        emim = ndi.filters.gaussian_filter(em, fwhm2sig(fwhm_input, Cnt), mode='mirror')
    else:
        muim = muo + muh
        emim = em

    muim = ndi.interpolation.zoom(muim, Cnt['SCTSCLMU'], order=3) # (0.499, 0.5, 0.5)
    emim = ndi.interpolation.zoom(emim, Cnt['SCTSCLEM'], order=3) # (0.34, 0.33, 0.33)

    # -smooth the mu-map for mask creation.
    # the mask contains voxels for which attenuation ray LUT is found.
    if fwhm_input > 0.:
        smomu = ndi.filters.gaussian_filter(muim, fwhm2sig(fwhm_input, Cnt), mode='mirror')
        mumsk = np.int8(smomu > 0.003)
    else:
        mumsk = np.int8(muim > 0.001)

    # CORE SCATTER ESTIMATION
    NSCRS, NSRNG = sctLUT['NSCRS'], sctLUT['NSRNG']
    sctout = {
        'sct_3d': np.zeros((Cnt['TOFBINN'], snno_, NSCRS, NSCRS), dtype=np.float32),
        'sct_val': np.zeros((Cnt['TOFBINN'], NSRNG, NSCRS, NSRNG, NSCRS), dtype=np.float32)}

    # <<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>>
    nifty_scatter.vsm(sctout, muim, mumsk, emim, sctLUT, axLUT, Cnt)
    # <<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>>

    sct3d = sctout['sct_3d']
    sctind = sctLUT['sct2aw']

    log.debug('total scatter sum: {}'.format(np.sum(sct3d)))

    # -------------------------------------------------------------------
    # > initialise output dictionary
    out = {}

    if return_uninterp:
        out['uninterp'] = sct3d
        out['indexes'] = sctind
    # -------------------------------------------------------------------

    if np.sum(sct3d) < 1e-04:
        log.warning('total scatter below threshold: {}'.format(np.sum(sct3d)))
        sss = np.zeros((snno, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
        asnmsk = np.zeros((snno, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
        sssr = np.zeros((Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
        return sss, sssr, asnmsk

    # import pdb; pdb.set_trace()

    # -------------------------------------------------------------------
    if interpolate:
        # > interpolate basic scatter distributions into full size and
        # > transfer them to sinograms

        log.debug('transaxial scatter interpolation...')
        start = time.time()
        ssn, sssr = intrp_bsct(sct3d, Cnt, sctLUT, ssrlut)
        stop = time.time()
        log.debug('scatter interpolation done in {} sec.'.format(stop - start))

        if not scaling:
            out['ssrb'] = sssr
            out['sino'] = ssn
            return out
    else:
        return out
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # import pdb; pdb.set_trace()
    '''
    debugging scatter:
    import matplotlib.pyplot as plt
    ss = np.squeeze(sct3d)
    ss = np.sum(ss, axis=0)
    plt.matshow(ss)
    plt.matshow(sct3d[0,41,...])
    plt.matshow(np.sum(sct3d[0,0:72,...],axis=0))

    plt.plot(np.sum(sct3d, axis=(0,2,3)))

    rslt = sctout['sct_val']
    rslt.shape
    plt.matshow(rslt[0,4,:,4,:])

    debugging scatter:
    plt.matshow(np.sum(sssr, axis=(0,1)))
    plt.matshow(np.sum(ssn, axis=(0,1)))
    plt.matshow(sssr[0,70,...])
    plt.matshow(sssr[0,50,...])
    '''
    # -------------------------------------------------------------------

    # > get SSR for randoms from span-1 or span-11
    rssr = np.zeros((Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
    if scaling:
        for i in range(snno):
            rssr[ssrlut[i], :, :] += rsino[i, :, :]

    # ATTENUATION FRACTIONS for scatter only regions, and NORMALISATION for all SCATTER
    # <<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>>
    currentspan = Cnt['SPN']
    Cnt['SPN'] = 1
    atto = cu.zeros((txLUT['Naw'], Cnt['NSN1']), dtype=np.float32)
    petprj.fprj(atto.cuvec,
                cu.asarray(mu_sctonly).cuvec, txLUT, axLUT, np.array([-1], dtype=np.int32), Cnt, 1)
    atto = mmraux.putgaps(atto, txLUT, Cnt)
    # --------------------------------------------------------------
    # > get norm components setting the geometry and axial to ones
    # as they are accounted for differently
    nrmcmp['geo'][:] = 1
    nrmcmp['axe1'][:] = 1
    # get sino with no gaps
    nrmg = np.zeros((txLUT['Naw'], Cnt['NSN1']), dtype=np.float32)
    mmr_auxe.norm(nrmg, nrmcmp, histo['buckets'], axLUT, txLUT['aw2ali'], Cnt)
    nrm = mmraux.putgaps(nrmg, txLUT, Cnt)
    # --------------------------------------------------------------

    # > get attenuation + norm in (span-11) and SSR
    attossr = np.zeros((Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
    nrmsssr = np.zeros((Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)

    for i in range(Cnt['NSN1']):
        si = axLUT['sn1_ssrb'][i]
        attossr[si, :, :] += atto[i, :, :] / float(axLUT['sn1_ssrno'][si])
        nrmsssr[si, :, :] += nrm[i, :, :] / float(axLUT['sn1_ssrno'][si])
    if currentspan == 11:
        Cnt['SPN'] = 11
        nrmg = np.zeros((txLUT['Naw'], snno), dtype=np.float32)
        mmr_auxe.norm(nrmg, nrmcmp, histo['buckets'], axLUT, txLUT['aw2ali'], Cnt)
        nrm = mmraux.putgaps(nrmg, txLUT, Cnt)
    # --------------------------------------------------------------

    # <<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>>

    # get the mask for the object from uncorrected emission image
    if emmsk and os.path.isfile(datain['em_nocrr']):
        nim = nib.load(datain['em_nocrr'])
        eim = nim.get_fdata(dtype=np.float32)
        eim = eim[:, ::-1, ::-1]
        eim = np.transpose(eim, (2, 1, 0))

        em_sctonly = ndi.filters.gaussian_filter(eim, fwhm2sig(.6, Cnt), mode='mirror')
        msk = np.float32(em_sctonly > 0.07 * np.max(em_sctonly))
        msk = ndi.filters.gaussian_filter(msk, fwhm2sig(.6, Cnt), mode='mirror')
        msk = np.float32(msk > 0.01)
        msksn = mmrprj.frwd_prj(msk, txLUT, axLUT, Cnt)

        mssr = mmraux.sino2ssr(msksn, axLUT, Cnt)
        mssr = mssr > 0
    else:
        mssr = np.zeros((Cnt['NSEG0'], Cnt['NSANGLES'], Cnt['NSBINS']), dtype=bool)

    # <<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>><<+>>

    # ======= SCALING ========
    # > scale scatter using non-TOF SSRB sinograms

    # > gap mask
    rmsk = (txLUT['msino'] > 0).T
    rmsk.shape = (1, Cnt['NSANGLES'], Cnt['NSBINS'])
    rmsk = np.repeat(rmsk, Cnt['NSEG0'], axis=0)

    # > include attenuating object into the mask (and the emission if selected)
    amsksn = np.logical_and(attossr >= mask_threshlod, rmsk) * ~mssr

    # > scaling factors for SSRB scatter
    scl_ssr = np.zeros((Cnt['NSEG0']), dtype=np.float32)

    for sni in range(Cnt['NSEG0']):
        # > region for scaling defined by the percentage of lowest
        # > but usable/significant scatter
        thrshld = prcnt_scl * np.max(sssr[sni, :, :])
        amsksn[sni, :, :] *= (sssr[sni, :, :] > thrshld)
        amsk = amsksn[sni, :, :]

        # > normalised estimated scatter
        mssn = sssr[sni, :, :] * nrmsssr[sni, :, :]
        vpsn = histo['pssr'][sni, amsk] - rssr[sni, amsk]
        scl_ssr[sni] = np.sum(vpsn) / np.sum(mssn[amsk])

        # > scatter SSRB sinogram output
        sssr[sni, :, :] *= nrmsssr[sni, :, :] * scl_ssr[sni]

    # === scale scatter for the full-size sinogram ===
    sss = np.zeros((snno, Cnt['NSANGLES'], Cnt['NSBINS']), dtype=np.float32)
    for i in range(snno):
        sss[i, :, :] = ssn[i, :, :] * scl_ssr[ssrlut[i]] * saxnrm[i] * nrm[i, :, :]
    '''
    # > debug
    si = 60
    ai = 60
    matshow(sssr[si,...])

    figure()
    plot(histo['pssr'][si,ai,:])
    plot(rssr[si,ai,:]+sssr[si,ai,:])

    plot(np.sum(histo['pssr'],axis=(0,1)))
    plot(np.sum(rssr+sssr,axis=(0,1)))
    '''

    # === OUTPUT ===
    if return_uninterp:
        out['uninterp'] = sct3d
        out['indexes'] = sctind

    if return_ssrb:
        out['ssrb'] = sssr
        out['rssr'] = rssr

    if return_mask:
        out['mask'] = amsksn

    if return_scaling:
        out['scaling'] = scl_ssr

    # if self_scaling:
    #     out['scl_sn1'] = scl_ssn

    if not out:
        return sss
    else:
        out['sino'] = sss
        return out
