"""Simulations for image reconstruction with recommended reduced axial field of view"""
import logging

import cuvec as cu
import numpy as np
from scipy import ndimage as ndi
from tqdm.auto import trange

from niftypet import nimpa

from .. import mmraux
from ..img import mmrimg
from . import mmrprj, mmrrec, petprj

log = logging.getLogger(__name__)


def simulate_sino(
    petim,
    ctim,
    scanner_params,
    simulate_3d=False,
    slice_idx=-1,
    mu_input=False,
):
    '''
    Simulate the measured sinogram with photon attenuation.

    petim  : the input PET image based on which the emission sinogram is found
    ctim  : CT image, in register with PET and the same dimensions, is used
        for estimating the attenuation factors, which are then applied to
        simulate emission sinogram with realistic photon attenuation.
    slice_idx  : chosen 2D slice out of the 3D image for the fast simulation.
    scanner_params  : scanner parameters containing scanner constants and
        axial and transaxial look up tables (LUTs)
    mu_input  : if True, the values are representative of a mu-map in [1/cm],
        otherwise it represents the CT in [HU].
    '''
    # > decompose the scanner constants and LUTs for easier access
    Cnt = scanner_params['Cnt']

    if petim.shape != ctim.shape:
        raise ValueError('The shapes of the PET and CT images are inconsistent.')

    if simulate_3d:
        if petim.ndim != 3 \
                or petim.shape != (Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']):
            raise ValueError('The input image shape does not match the scanner image size.')
        if petim.max() > 200:
            log.warning('the PET image may have too large intensities for robust simulation.')
    else:
        # > 2D case with reduced rings
        if len(petim.shape) == 3:
            # make sure that the shape of the input image matches the image size of the scanner
            if petim.shape[1:] != (Cnt['SO_IMY'], Cnt['SO_IMX']):
                raise ValueError(
                    'The input image shape for x and y does not match the scanner image size.')
            # pick the right slice index (slice_idx) if not given or mistaken
            if slice_idx < 0:
                log.warning(
                    'the axial index <slice_idx> is chosen to be in the middle of axial FOV.')
                slice_idx = petim.shape[0] / 2
            if slice_idx >= petim.shape[0]:
                raise ValueError('The axial index for 2D slice selection is outside the image.')
        elif len(petim.shape) == 2:
            # make sure that the shape of the input image matches the image size of the scanner
            if petim.shape != (Cnt['SO_IMY'], Cnt['SO_IMX']):
                raise ValueError(
                    'The input image shape for x and y does not match the scanner image size.')
            petim.shape = (1,) + petim.shape
            ctim.shape = (1,) + ctim.shape
            slice_idx = 0

        if 'rSZ_IMZ' not in Cnt:
            raise ValueError('Missing reduced axial FOV parameters.')

    # import pdb; pdb.set_trace()

    # --------------------
    if mu_input:
        mui = ctim
    else:
        # > get the mu-map [1/cm] from CT [HU]
        mui = nimpa.ct2mu(ctim)

    # > get rid of negative values
    mui[mui < 0] = 0
    # --------------------

    if simulate_3d:
        rmu = mui
        rpet = petim
    else:
        # > 2D case with reduced rings
        # --------------------
        # > create a number of slices of the same chosen image slice
        # for reduced (fast) 3D simulation
        rmu = mui[slice_idx, :, :]
        rmu.shape = (1,) + rmu.shape
        rmu = np.repeat(rmu, Cnt['rSZ_IMZ'], axis=0)
        # --------------------

        # --------------------
        # > form a short 3D image of the same emission image slice
        rpet = petim[slice_idx, :, :].copy()
        rpet.shape = (1,) + rpet.shape
        rpet = np.repeat(rpet, Cnt['rSZ_IMZ'], axis=0)
        # --------------------

    # > forward project the mu-map to obtain attenuation factors
    attsino = mmrprj.frwd_prj(rmu, scanner_params, attenuation=True)

    # > forward project the PET image to obtain non-attenuated emission sino
    emisino = mmrprj.frwd_prj(rpet, scanner_params, attenuation=False)

    # > return the simulated emission sino with photon attenuation
    return attsino * emisino


def simulate_recon(
    measured_sino,
    ctim,
    scanner_params,
    simulate_3d=False,
    nitr=60,
    fwhm_rm=0.,
    slice_idx=-1,
    randoms=None,
    scatter=None,
    mu_input=False,
    msk_radius=29.,
    psf=None,
):
    '''
    Reconstruct PET image from simulated input data
    using the EM-ML (2D) or OSEM (3D) algorithm.

    measured_sino  : simulated emission data with photon attenuation
    ctim  : either a 2D CT image or a 3D CT image from which a 2D slice
        is chosen (slice_idx) for estimation of the attenuation factors
    slice_idx  : index to extract one 2D slice for this simulation
        if input image is 3D
    nitr  : number of iterations used for the EM-ML reconstruction algorithm
    scanner_params  : scanner parameters containing scanner constants and
        axial and transaxial look up tables (LUTs)
    randoms  : randoms and scatter events (optional)
    '''
    # > decompose the scanner constants and LUTs for easier access
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']
    psfkernel = mmrrec.psf_config(psf, Cnt)

    if simulate_3d:
        if ctim.ndim!=3 \
                or ctim.shape!=(Cnt['SO_IMZ'], Cnt['SO_IMY'], Cnt['SO_IMX']):
            raise ValueError('The CT/mu-map image does not match the scanner image shape.')
    else:
        # > 2D case with reduced rings
        if len(ctim.shape) == 3:
            # make sure that the shape of the input image matches the image size of the scanner
            if ctim.shape[1:] != (Cnt['SO_IMY'], Cnt['SO_IMX']):
                raise ValueError(
                    'The input image shape for x and y does not match the scanner image size.')
            # pick the right slice index (slice_idx) if not given or mistaken
            if slice_idx < 0:
                log.warning(
                    'the axial index <slice_idx> is chosen to be in the middle of axial FOV.')
                slice_idx = ctim.shape[0] / 2
            if slice_idx >= ctim.shape[0]:
                raise ValueError('The axial index for 2D slice selection is outside the image.')
        elif len(ctim.shape) == 2:
            # make sure that the shape of the input image matches the image size of the scanner
            if ctim.shape != (Cnt['SO_IMY'], Cnt['SO_IMX']):
                raise ValueError(
                    'The input image shape for x and y does not match the scanner image size.')
            ctim.shape = (1,) + ctim.shape
            slice_idx = 0

        if 'rSZ_IMZ' not in Cnt:
            raise ValueError('Missing reduced axial FOV parameters.')

    # --------------------
    if mu_input:
        mui = ctim
    else:
        # > get the mu-map [1/cm] from CT [HU]
        mui = nimpa.ct2mu(ctim)

    # > get rid of negative values
    mui[mui < 0] = 0
    # --------------------

    if simulate_3d:
        rmu = mui
        # > number of axial sinograms
        nsinos = Cnt['NSN11']
    else:
        # --------------------
        # > create a number of slides of the same chosen image slice
        # for reduced (fast) 3D simulation
        rmu = mui[slice_idx, :, :]
        rmu.shape = (1,) + rmu.shape
        rmu = np.repeat(rmu, Cnt['rSZ_IMZ'], axis=0)
        # --------------------
        # > number of axial sinograms
        nsinos = Cnt['rNSN1']

    # import pdb; pdb.set_trace()

    # > attenuation factor sinogram
    attsino = mmrprj.frwd_prj(rmu, scanner_params, attenuation=True, dev_out=True)

    nrmsino = np.ones(attsino.shape, dtype=np.float32)

    # > randoms and scatter put together
    if isinstance(randoms, np.ndarray) and measured_sino.shape == randoms.shape:
        rsng = mmraux.remgaps(randoms, txLUT, Cnt)
    else:
        rsng = 1e-5 * np.ones((Cnt['Naw'], nsinos), dtype=np.float32)

    if isinstance(scatter, np.ndarray) and measured_sino.shape == scatter.shape:
        ssng = mmraux.remgaps(scatter, txLUT, Cnt)
    else:
        ssng = 1e-5 * np.ones((Cnt['Naw'], nsinos), dtype=np.float32)

    # resolution modelling
    Cnt['SIGMA_RM'] = mmrrec.fwhm2sig(fwhm_rm, voxsize=Cnt['SZ_VOXZ'] * 10) if fwhm_rm else 0

    if simulate_3d:
        log.debug('------ OSEM (%d) -------', nitr)

        # measured sinogram in GPU-enabled shape
        psng = mmraux.remgaps(measured_sino.astype(np.uint16), txLUT, Cnt)

        # > mask for reconstructed image.  anything outside it is set to zero
        msk = mmrimg.get_cylinder(Cnt, rad=msk_radius, xo=0, yo=0, unival=1, gpu_dim=True) > 0.9

        # > init image
        eimg = np.ones((Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)

        # ------------------------------------
        Sn = 14    # number of subsets

        # -get one subset to get number of projection bins in a subset
        Sprj, s = mmrrec.get_subsets14(0, scanner_params)
        Nprj = len(Sprj)

        # > init subset array and sensitivity image for a given subset
        sinoTIdx = np.zeros((Sn, Nprj + 1), dtype=np.int32)

        # > init sensitivity images for each subset
        sim = np.zeros((Sn, Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)
        tmpsim = cu.zeros((Cnt['SZ_IMY'], Cnt['SZ_IMX'], Cnt['SZ_IMZ']), dtype=np.float32)

        for n in trange(Sn, desc="sensitivity", leave=log.getEffectiveLevel() < logging.INFO):
            # first number of projection for the given subset
            sinoTIdx[n, 0] = Nprj
            sinoTIdx[n, 1:], s = mmrrec.get_subsets14(n, scanner_params)

            # > sensitivity image
            petprj.bprj(tmpsim.cuvec,
                        cu.asarray(attsino[sinoTIdx[n, 1:], :]).cuvec, txLUT, axLUT,
                        sinoTIdx[n, 1:], Cnt)
            sim[n] = tmpsim
        del tmpsim
        # -------------------------------------

        for _ in trange(nitr, desc="OSEM", disable=log.getEffectiveLevel() > logging.INFO,
                        leave=log.getEffectiveLevel() < logging.INFO):
            petprj.osem(eimg, psng, rsng, ssng, nrmsino, attsino, sinoTIdx, sim, msk, psfkernel,
                        txLUT, axLUT, Cnt)
        eim = mmrimg.convert2e7(eimg, Cnt)

    else:

        def psf(x, output=None):
            if Cnt['SIGMA_RM']:
                x = ndi.gaussian_filter(x, sigma=Cnt['SIGMA_RM'], mode='constant', output=None)
            return x

        # > estimated image, initialised to ones
        eim = np.ones(rmu.shape, dtype=np.float32)

        msk = mmrimg.get_cylinder(Cnt, rad=msk_radius, xo=0, yo=0, unival=1, gpu_dim=False) > 0.9

        # > sensitivity image for the EM-ML reconstruction
        sim = mmrprj.back_prj(attsino, scanner_params)
        sim_inv = 1 / psf(sim)
        sim_inv[~msk] = 0

        rndsct = rsng + ssng
        for _ in trange(nitr, desc="MLEM", disable=log.getEffectiveLevel() > logging.INFO,
                        leave=log.getEffectiveLevel() < logging.INFO):
            # > remove gaps from the measured sinogram
            # > then forward project the estimated image
            # > after which divide the measured sinogram
            # by the estimated sinogram (forward projected)
            crrsino = (mmraux.remgaps(measured_sino, txLUT, Cnt) /
                       (mmrprj.frwd_prj(psf(eim), scanner_params, dev_out=True) + rndsct))

            # > back project the correction factors sinogram
            bim = mmrprj.back_prj(crrsino, scanner_params)
            bim = psf(bim, output=bim)

            # > divide the back-projected image by the sensitivity image
            # > update the estimated image and remove NaNs
            eim *= bim * sim_inv
            eim[np.isnan(eim)] = 0

    return eim
