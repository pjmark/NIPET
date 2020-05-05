import os
import sys

import numpy as np
from math import pi
import h5py



def constants_h5(pthfn):
    # open the HDF5 file
    f = h5py.File(pthfn,'r')
    # coincidence event mode
    cncdmd = f['HeaderData']['AcqParameters']['EDCATParameters']['coinOutputMode'][0]
    if cncdmd==802:
        # bytes per event in this mode:
        bpe = 6
        print('i> the list-mode data is recorded in the NOMINAL mode (6 bytes per event) and will be processed.')
    elif cncdmd==803:
        bpe = 16
        print('--------------------------')
        print('e> the list-mode data is recorded in the CALIBRATION mode (16 bytes per event) which is not currently supported, sorry.')
        print('--------------------------')
    elif cncdmd==805:
        bpe = 8
        print('--------------------------')
        print('e> the list-mode data is recorded in the ENERGY mode (8 bytes per event) which is not currently supported, sorry.')
        print('--------------------------')
    else:
        bpe = 0
        print('--------------------------')
        print('e> the list-mode data is recorded in an UNKNOWN mode.  Sorry.')
        print('--------------------------')

    CntH5 = {
        # scan start time marker (used as offset)
        'toff':f['HeaderData']['AcqStats']['frameStartCoincTStamp'][0],
        'Deff':f['HeaderData']['SystemGeometry']['effectiveRingDiameter'][0],
        'TFOV':f['HeaderData']['AcqParameters']['EDCATParameters']['transAxialFOV'][0],
        'cpitch':f['HeaderData']['SystemGeometry']['interCrystalPitch'][0],
        'bpitch':f['HeaderData']['SystemGeometry']['interBlockPitch'][0],
        'exLOR':f['HeaderData']['AcqParameters']['RxScanParameters']['extraRsForTFOV'][0],

        'axCB':f['HeaderData']['SystemGeometry']['axialCrystalsPerBlock'][0],
        'axBU':f['HeaderData']['SystemGeometry']['axialBlocksPerUnit'][0],
        'axUM':f['HeaderData']['SystemGeometry']['axialUnitsPerModule'][0],
        'axMno':f['HeaderData']['SystemGeometry']['axialModulesPerSystem'][0],
        'txCB':f['HeaderData']['SystemGeometry']['radialCrystalsPerBlock'][0],
        'txBU':f['HeaderData']['SystemGeometry']['radialBlocksPerUnit'][0],
        'txUM':f['HeaderData']['SystemGeometry']['radialUnitsPerModule'][0],
        'txMno':f['HeaderData']['SystemGeometry']['radialModulesPerSystem'][0],
        'MRD':f['HeaderData']['AcqParameters']['BackEndAcqFilters']['maxRingDiff'][0],

        'tau0':f['HeaderData']['AcqParameters']['EDCATParameters']['negCoincidenceWindow'][0],
        'tau1':f['HeaderData']['AcqParameters']['EDCATParameters']['posCoincidenceWindow'][0],
        'tauP':f['HeaderData']['AcqParameters']['EDCATParameters']['coincTimingPrecision'][0],
        'TOFC':f['HeaderData']['AcqParameters']['RxScanParameters']['tofCompressionFactor'][0],

        'LLD':f['HeaderData']['AcqParameters']['EDCATParameters']['lower_energy_limit'][0],
        'ULD':f['HeaderData']['AcqParameters']['EDCATParameters']['upper_energy_limit'][0],
        'BPE':bpe
    }

    f.close()
    return CntH5


def get_nbins(Cnt):
    txUno = Cnt['txUM']*Cnt['txMno']
    txCU  = Cnt['txCB'] * Cnt['txBU']
    cpitch = Cnt['cpitch']
    minValue =  np.ceil(2*(np.arcsin( (Cnt['TFOV']*10.0)/Cnt['Deff'] ) - np.floor(txUno * np.arcsin((Cnt['TFOV']*10.0)/Cnt['Deff'])/pi)*pi/txUno)/cpitch )
    if txCU < minValue:
        minValue = txCU
    halfFanLORs = np.floor(txUno * np.arcsin( (Cnt['TFOV']*10.0)/Cnt['Deff'] )/pi) * txCU + minValue
    W = 2*int(halfFanLORs) + 2*Cnt['exLOR'] + 1
    C = Cnt['txCB'] * Cnt['txBU'] * Cnt['txUM'] * Cnt['txMno'] * Cnt['axCB']
    if W > C:
        W = C
    return W



#===================================================================================
# SCANNER CONSTANTS
def get_sig_constants(pthfn):

    if not os.path.isfile(pthfn):
        print('e> coult not open the file HDF5 to get SIGNA constants')
        return

    Cnt= constants_h5(pthfn)

    #> default logging set to WARNING only (30)
    Cnt['LOG'] = 30

    # number of sinogram angles
    NSBINS = get_nbins(Cnt)
    # number of transxial crystals
    NCRS = Cnt['txCB'] * Cnt['txBU'] * Cnt['txUM'] * Cnt['txMno']
    # number of rings
    NRNG = Cnt['axCB']*Cnt['axBU']*Cnt['axUM']*Cnt['axMno']
    # number of 2D sinograms
    NSN = NRNG**2-(NRNG-1)
    #bootstrapping of the list-mode data, 0: None, 1: Not used (was non-parametric for the mMR), 2: parametric
    Cnt['BTP'] = 0
    Cnt['NCRS'] = NCRS
    Cnt['NRNG'] = NRNG
    # defnie reduced detector ring
    Cnt['RNG_END'] = NRNG
    Cnt['RNG_STRT'] = 0
    Cnt['NSN'] = NSN
    Cnt['NSBINS'] = NSBINS
    Cnt['NSANGLES'] = NCRS//2

    Cnt['NSEG0'] = 2*Cnt['axUM']*Cnt['axCB']-1

    # LM processing
    # integration time of 1 sec
    Cnt['ITIME'] = 1000
    # number of CUDA streams
    Cnt['NSTREAMS'] = 32
    # number of elements per data chunk
    Cnt['ELECHNK'] = (268435456//Cnt['NSTREAMS']) # 2^{28} = 268435456 elements (6Bytes) to make up 1.6GB

    # projection view integration time (length of the short time frames t = 2^VTIME)
    Cnt['VTIME'] = 2
    # the short time frame projection views are limited to 90 mins only
    Cnt['MXNITAG'] = 5400

    # transaxial and axial crystal sizes in cm
    Cnt['TXCRS'] = 0.395
    Cnt['AXCRS'] = 0.530
    # gap between axial detector units
    Cnt['AXGAP'] = 0.280
    # axial FOV
    Cnt['AXFOV'] = (Cnt['axUM']-1)*Cnt['AXGAP'] + Cnt['axUM']*Cnt['axCB']*Cnt['AXCRS']

    return Cnt
#===================================================================================



#===================================================================================
# AXIAL LUTS
def get_axLUT(Cnt):

    # calculated rings
    NRNG_c = Cnt['RNG_END'] - Cnt['RNG_STRT']
    NSN1_c = NRNG_c**2-(NRNG_c-1)

    # get the sino LUTs
    M = np.zeros((NRNG_c,NRNG_c), dtype=np.int16)
    # sino index
    Msn = np.zeros((NRNG_c,NRNG_c), dtype=np.int16)
    # sino index SIGNA native
    Msig = np.zeros((NRNG_c,NRNG_c), dtype=np.int16)

    sn_rno = np.zeros((NRNG_c**2,2), dtype=np.int16)

    # diagonal linear index (positive only)
    dli = 0
    # full diagonal linear index (positive and negative)
    sni = 0

    for ro in range(0, NRNG_c):
        if ro==0:
            oblique = 1
        else:
            oblique = 2
        for m in range(oblique):
            # selects the sub-michelogram of the whole michelogram
            strt = Cnt['NRNG']*(ro+Cnt['RNG_STRT']) + Cnt['RNG_STRT']
            stop = (Cnt['RNG_STRT']+NRNG_c)*Cnt['NRNG']
            step = Cnt['NRNG']+1
            for li in range(strt, stop, step): #goes along a diagonal started in the first row at r2o
                #from the linear indices of michelogram get the subscript indices
                if m==0:
                    r0 = li//Cnt['NRNG']
                    r1 = li - r0*Cnt['NRNG']
                    M[r1,r0] = dli
                    dli += 1
                else:
                    r1 = li//Cnt['NRNG']
                    r0 = li - r1*Cnt['NRNG']

                Msn[r1,r0] = sni

                sn_rno[sni,0] = r0
                sn_rno[sni,1] = r1

                sni += 1

                #--- SIGNA native ---
                rdiff = r0-r1
                rsum  = r0+r1
                if (rdiff>1):
                    angle = rdiff//2
                    if ( angle<=Cnt['MRD']/2 ):
                        snis =  rsum + (4*angle-2)*Cnt['NRNG'] - (4*angle*angle - 1)
                elif (rdiff<-1):
                    angle = -rdiff//2
                    if (angle<=Cnt['MRD']//2):
                        snis = rsum + (4*angle)*Cnt['NRNG'] - ((angle+1)*4*angle)
                else:
                    snis = rsum
                Msig[r1,r0] = snis
                # ------
    axLUT = {'r2s':Msn, 'r2sig':Msig, 's2r':sn_rno}
    return axLUT
            

            
#===================================================================================        


#===================================================================================
# TRANSIAXIAL LUTS
def get_txLUT(Cnt):
    # number of bins per sinogram angle
    NSBINS = Cnt['NSBINS']
    # number of sinogram angles
    NSANGLES = Cnt['NSANGLES']
    # number of rings
    NCRS = Cnt['NCRS']
    # crystal to sinogram index lookup table (LUT)
    c2s = np.zeros((NCRS, NCRS), dtype=np.int32)
    # sinogram to crystal index LUT
    s2c = np.zeros((NSANGLES*NSBINS, 2), dtype=np.int16)
    for c0 in range(NCRS):
        for c1 in range(NCRS):
            if ((NCRS//2)<=(c0+c1)) and ((c0+c1) < (3*NCRS//2)):
                iw = (NSBINS-1)//2 + (c0-c1-NCRS//2)
            else:
                iw = (NSBINS-1)//2 - (c0-c1-NCRS//2)
            if (iw >= 0) and (iw <= (NSBINS-1)):
                ia = ((c0+c1 + NCRS//2)%NCRS)//2
                aw = ia + NSANGLES*iw
                c2s[c1,c0] = aw
                c2s[c0,c1] = aw
                s2c[aw,0] = c0
                s2c[aw,1] = c1
            else:
                c2s[c1,c0] = -1
                c2s[c0,c1] = -1

    txLUT = {'c2s':c2s, 's2c':s2c}
    return txLUT
#===================================================================================


def init_sig(pthfn):

    # get the constants for the mMR
    Cnt = get_sig_constants(pthfn)

    # transaxial look up tables
    txLUT = get_txLUT(Cnt)

    # axial look up tables
    axLUT = get_axLUT(Cnt)

    return Cnt, txLUT, axLUT
