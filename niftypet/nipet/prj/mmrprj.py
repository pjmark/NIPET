"""Forward and back projector for PET data reconstruction"""
import logging

import cuvec as cu
import numpy as np

from .. import mmraux
from ..img import mmrimg
from . import petprj

log = logging.getLogger(__name__)
ISUB_DEFAULT = np.array([-1], dtype=np.int32)

# ========================================================================
# transaxial (one-slice) projector
# ------------------------------------------------------------------------


def trnx_prj(scanner_params, sino=None, im=None):
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']

    # if sino==None and im==None:
    #     raise ValueError('Input sinogram or image has to be given.')
    if sino is not None and im is not None:
        raise ValueError('Only one input should be given: sinogram or image.')

    if sino is None:
        sino = np.zeros((txLUT['Naw'],), dtype=np.float32)
    if im is None:
        im = np.zeros((Cnt['SO_IMY'], Cnt['SO_IMX']), dtype=np.float32)

    tv = np.zeros(Cnt['NTV'] * Cnt['Naw'], dtype=np.uint8)
    tt = np.zeros(Cnt['NTT'] * Cnt['Naw'], dtype=np.float32)

    petprj.tprj(sino, im, tv, tt, txLUT, Cnt)

    return {'tv': tv, 'tt': tt}


# ========================================================================
# forward projector
# ------------------------------------------------------------------------


def frwd_prj(im, scanner_params, isub=ISUB_DEFAULT, dev_out=False, attenuation=False,
             fullsino_out=True, output=None):
    """
    Calculate forward projection (a set of sinograms) for the provided input image.
    Arguments:
        im -- input image (can be emission or mu-map image).
        scanner_params -- dictionary of all scanner parameters, containing scanner constants,
            transaxial and axial look up tables (LUT).
        isub -- array of transaxial indices of all sinograms (angles x bins) used for subsets.
            when the first element is negative, all transaxial bins are used (as in pure EM-ML).
        dev_out -- if True, output sinogram is in the device form, i.e., with two dimensions
            (# bins/angles, # sinograms) instead of default three (# sinograms, # bins, # angles).
        attenuation -- controls whether emission or LOR attenuation probability sinogram
            is calculated; the default is False, meaning emission sinogram; for attenuation
            calculations (attenuation=True), the exponential of the negative of the integrated
            mu-values along LOR path is taken at the end.
        output(CuVec, optional) -- output sinogram.
    """
    # Get particular scanner parameters: Constants, transaxial and axial LUTs
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    # >choose between attenuation forward projection (mu-map is the input)
    # >or the default for emission image forward projection
    if attenuation:
        att = 1
    else:
        att = 0

    if Cnt['SPN'] == 1:
        # number of rings calculated for the given ring range
        # (optionally we can use only part of the axial FOV)
        NRNG_c = Cnt['RNG_END'] - Cnt['RNG_STRT']
        # number of sinos in span-1
        nsinos = NRNG_c**2
        # correct for the max. ring difference in the full axial extent
        # (don't use ring range (1,63) as for this case no correction)
        if NRNG_c == 64:
            nsinos -= 12
    elif Cnt['SPN'] == 11:
        nsinos = Cnt['NSN11']
    elif Cnt['SPN'] == 0:
        nsinos = Cnt['NSEG0']

    if im.shape[0] == Cnt['SO_IMZ'] and im.shape[1] == Cnt['SO_IMY'] and im.shape[2] == Cnt[
            'SO_IMX']:
        ims = mmrimg.convert2dev(im, Cnt)
    elif im.shape[0] == Cnt['SZ_IMX'] and im.shape[1] == Cnt['SZ_IMY'] and im.shape[2] == Cnt[
            'SZ_IMZ']:
        ims = im
    elif im.shape[0] == Cnt['rSO_IMZ'] and im.shape[1] == Cnt['SO_IMY'] and im.shape[2] == Cnt[
            'SO_IMX']:
        ims = mmrimg.convert2dev(im, Cnt)
    elif im.shape[0] == Cnt['SZ_IMX'] and im.shape[1] == Cnt['SZ_IMY'] and im.shape[2] == Cnt[
            'rSZ_IMZ']:
        ims = im
    else:
        raise ValueError('wrong image size;'
                         ' it has to be one of these: (z,y,x) = (127,344,344)'
                         ' or (y,x,z) = (320,320,128)')

    log.debug('number of sinos: %d', nsinos)

    # predefine the sinogram.
    # if subsets are used then only preallocate those bins which will be used.
    if isub[0] < 0:
        out_shape = txLUT['Naw'], nsinos
    else:
        out_shape = len(isub), nsinos

    if output is None:
        sinog = cu.zeros(out_shape, dtype=np.float32)
    else:
        sinog = cu.asarray(output)
        assert sinog.shape == out_shape
        assert sinog.dtype == np.dtype('float32')
    # --------------------
    petprj.fprj(sinog.cuvec, cu.asarray(ims).cuvec, txLUT, axLUT, isub, Cnt, att)
    # --------------------

    # get the sinogram bins in a full sinogram if requested
    if fullsino_out and isub[0] >= 0:
        sino = cu.zeros((txLUT['Naw'], nsinos), dtype=np.float32)
        sino[isub, :] = sinog
    else:
        sino = sinog

    # put the gaps back to form displayable sinogram
    if not dev_out and fullsino_out:
        sino = mmraux.putgaps(sino, txLUT, Cnt)

    return sino


# ========================================================================
# back projector
# ------------------------------------------------------------------------


def back_prj(sino, scanner_params, isub=ISUB_DEFAULT, dev_out=False):
    '''
    Calculate forward projection for the provided input image.
    Arguments:
        sino -- input emission sinogram to be back projected to the image space.
        scanner_params -- dictionary of all scanner parameters, containing scanner constants,
            transaxial and axial look up tables (LUT).
        isub -- array of transaxial indices of all sinograms (angles x bins) used for subsets;
            when the first element is negative, all transaxial bins are used (as in pure EM-ML).
    '''
    # Get particular scanner parameters: Constants, transaxial and axial LUTs
    Cnt = scanner_params['Cnt']
    txLUT = scanner_params['txLUT']
    axLUT = scanner_params['axLUT']

    if Cnt['SPN'] == 1:
        # number of rings calculated for the given ring range
        # (optionally we can use only part of the axial FOV)
        NRNG_c = Cnt['RNG_END'] - Cnt['RNG_STRT']
        # number of sinos in span-1
        nsinos = NRNG_c**2
        # correct for the max. ring difference in the full axial extent
        # (don't use ring range (1,63) as for this case no correction)
        if NRNG_c == 64:
            nsinos -= 12
    elif Cnt['SPN'] == 11:
        nsinos = Cnt['NSN11']
    elif Cnt['SPN'] == 0:
        nsinos = Cnt['NSEG0']

    # > check first the Siemens default sinogram;
    # > for this default shape only full sinograms are expected--no subsets.
    if len(sino.shape) == 3:
        if sino.shape[0] != nsinos or sino.shape[1] != Cnt['NSANGLES'] or sino.shape[2] != Cnt[
                'NSBINS']:
            raise ValueError('Unexpected sinogram array dimensions/shape for Siemens defaults.')
        sinog = mmraux.remgaps(sino, txLUT, Cnt)

    elif len(sino.shape) == 2:
        if isub[0] < 0 and sino.shape[0] != txLUT["Naw"]:
            raise ValueError('Unexpected number of transaxial elements in the full sinogram.')
        elif isub[0] >= 0 and sino.shape[0] != len(isub):
            raise ValueError('Unexpected number of transaxial elements in the subset sinogram.')
        # > check if the number of sinograms is correct
        if sino.shape[1] != nsinos:
            raise ValueError('Inconsistent number of sinograms in the array.')
        # > when found the dimensions/shape are fine:
        sinog = sino
    else:
        raise ValueError('Unexpected shape of the input sinogram.')

    # predefine the output image depending on the number of rings used
    if Cnt['SPN'] == 1 and 'rSZ_IMZ' in Cnt:
        nvz = Cnt['rSZ_IMZ']
    else:
        nvz = Cnt['SZ_IMZ']
    bimg = cu.zeros((Cnt['SZ_IMX'], Cnt['SZ_IMY'], nvz), dtype=np.float32)

    # > run back-projection
    petprj.bprj(bimg.cuvec, cu.asarray(sinog).cuvec, txLUT, axLUT, isub, Cnt)

    if not dev_out:
        # > change from GPU optimised image dimensions to the standard Siemens shape
        bimg = mmrimg.convert2e7(bimg, Cnt)

    return bimg
