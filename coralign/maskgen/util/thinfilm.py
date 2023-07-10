"""Module for functions to generate complex-valued HLC occulters."""
import os
import numpy as np

from coralign.util.check import real_positive_scalar, oneD_array,\
                            real_nonnegative_scalar, scalar_integer


def calc_complex_occulter(lam, aoi, t_Ti_vec, t_Ni_vec, t_PMGI_vec,
                          d0, pol, flagOPD=False, SUBSTRATE='FS'):
    """
    Calculate the complex transmission and reflectance of an occulter.
    
    Using the thin film equations, calculates the complex transmission and
    reflectance for the provided combinations of metal and dielectric
    thicknesses and set of wavelengths.

    Parameters
    ----------
    lam : float
        Wavelength in meters.
    aoi : float
        Angle of incidence in degrees.
    t_Ti_vec : float
        1-D array of titanium thicknesses in meters. Titanium goes between
        fused silica and nickel layers.
    t_Ni_vec : array_like
        1-D array of nickel thicknesses in meters. Nickel goes between
        titanium and PMGI layers.
    t_PMGI_vec : array_like
        1-D array of PMGI thicknesses in meters.
    d0 : float
        Reference height for all phase offsets. Must be larger than the stack
        of materials, not including the substrate. Units of meters.
    pol : {0, 1, 2}
        Polarization state to compute values for.
        0 for TE(s) polarization,
        1 for TM(p) polarization,
        2 for mean of s and p polarizations
    flagOPD : bool, optional
        Flag to use the OPD convention. The default is False.
    SUBSTRATE : str, optional
        Material to use as the substrate. The default is 'FS'.

    Returns
    -------
    tCoef : numpy ndarray
        2-D array of complex transmission amplitude values.
    rCoef : numpy ndarray
        2-D array of complex reflection amplitude values.
    """
    real_positive_scalar(lam, 'lam', ValueError)
    real_nonnegative_scalar(aoi, 'aoi', ValueError)
    if aoi >= 90:
        raise ValueError('angle of incidence must be less than 90 degrees')
    oneD_array(t_Ti_vec, 't_Ti_vec', ValueError)
    oneD_array(t_Ni_vec, 't_Ni_vec', ValueError)
    oneD_array(t_PMGI_vec, 't_PMGI_vec', ValueError)
    if len(t_Ti_vec) != len(t_Ni_vec) or len(t_Ni_vec) != len(t_PMGI_vec):
        raise ValueError('Ti, Ni, and PMGI thickness vectors must all ' +
                         'have same length.')
    real_positive_scalar(d0, 'd0', ValueError)
    scalar_integer(pol, 'pol', ValueError)
    if not any(pol == np.array([0, 1, 2])):
        raise ValueError('pol must equal 0, 1, or 2')
    if not type(flagOPD) == bool:
        raise TypeError('flagOPD must be a boolean.')
    if not type(SUBSTRATE) == str:
        raise TypeError('SUBSTRATE must be a string')
    if SUBSTRATE not in ('FS', 'FUSEDSILICA', 'NBK7', 'N-BK7', 'BK7'):
        raise ValueError('Unusable value of SUBSTRATE.')
    
    lam_nm = lam * 1.0e9  # m --> nm
    lam_um = lam * 1.0e6  # m --> microns
    lam_um2 = lam_um * lam_um
    theta = aoi * (np.pi/180.)  # deg --> rad
    
    localpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define Material Properties
    # ---------------------------------------------
    # Substrate properties
    if SUBSTRATE.upper() in ('FS', 'FUSEDSILICA'):
        fnFS = os.path.join(localpath, 'materialdata',
                'fused_silica_Sellmeier_coefficients_from_Corning_website.txt')
        fsCoef = np.loadtxt(fnFS, delimiter="\t", unpack=False, comments="#")
        A1 = fsCoef[0]
        A2 = fsCoef[1]
        A3 = fsCoef[2]
        B1 = fsCoef[3]
        B2 = fsCoef[4]
        B3 = fsCoef[5]
        n_substrate = np.sqrt(1 + A1*lam_um2/(lam_um2 - B1) +
                           A2*lam_um2/(lam_um2 - B2) +
                           A3*lam_um2/(lam_um2 - B3))
    else:
        raise ValueError('SUBSTRATE value not recognized.')
    
    # Dielectric properties
    npmgi = 1.524 + 5.176e-03/lam_um**2 + 2.105e-4/lam_um**4
    
    # Metal layer properties
    # Titanium base layer under the nickel
    Nmetal = len(t_Ni_vec)
    t_Ti_vec = t_Ti_vec * np.ones(Nmetal)
    fnTi = os.path.join(localpath, 'materialdata',
                            'titanium_data_from_PBJohnsonAndRWChristy1974.txt')
    titanium = np.loadtxt(fnTi, delimiter="\t", unpack=False, comments="#")
    lam_ti = titanium[:, 0]  # nm
    n_ti = titanium[:, 1]
    k_ti = titanium[:, 2]
    nti = np.interp(lam_nm, lam_ti, n_ti)
    kti = np.interp(lam_nm, lam_ti, k_ti)
    
    # Nickel
    fnNickel = os.path.join(localpath, 'materialdata',
                            'nickel_data_from_Palik_via_Bala_wvlNM_n_k.txt')
    vnickel = np.loadtxt(fnNickel, delimiter="\t", unpack=False, comments="#")
    lam_nickel = vnickel[:, 0]  # nm
    n_nickel = vnickel[:, 1]
    k_nickel = vnickel[:, 2]
    nnickel = np.interp(lam_nm, lam_nickel, n_nickel)
    knickel = np.interp(lam_nm, lam_nickel, k_nickel)
    
    # Compute the complex transmission
    tCoef = np.zeros((Nmetal, ), dtype=complex)  # initialize
    rCoef = np.zeros((Nmetal, ), dtype=complex)  # initialize

    for ii in range(Nmetal):
        dni = t_Ni_vec[ii]
        dti = t_Ti_vec[ii]
        dpm = t_PMGI_vec[ii]
        
        nvec = np.array([1, 1, npmgi, nnickel-1j*knickel, nti-1j*kti,
                         n_substrate], dtype=complex)
        dvec = np.array([d0-dpm-dni-dti, dpm, dni, dti])
        
        # Choose polarization
        if(pol == 2):  # Mean of the two
            [dummy1, dummy2, rr0, tt0] = solver(nvec, dvec, theta,
                                                lam, False)
            [dummy1, dummy2, rr1, tt1] = solver(nvec, dvec, theta,
                                                lam, True)
            rr = (rr0+rr1)/2.
            tt = (tt0+tt1)/2.
        elif(pol == 0 or pol == 1):
            [dumm1, dummy2, rr, tt] = solver(nvec, dvec, theta, lam,
                                             bool(pol))
        else:
            raise ValueError('Wrong input value for polarization.')

        # Choose phase convention
        if not flagOPD:
            tCoef[ii] = np.conj(tt)  # Complex field transmission coef
            rCoef[ii] = np.conj(rr)  # Complex field reflection coef
        else:  # OPD phase convention
            tCoef[ii] = tt  # Complex field transmission coeffient
            rCoef[ii] = rr  # Complex field reflection coeffient
    
    return tCoef, rCoef


def solver(n, d0, theta, lam, tetm=False):
    """
    Solve the thin film equations for the given materials.

    Parameters
    ----------
    n : array_like
        index of refraction for each layer.
        n(1) = index of incident medium
        n(N) = index of transmission medium
        then length(n) must be >= 2
    d0 : array_like
        thickness of each layer, not counting incident medium or transmission
        medium. length(d) = length(n)-2.
    theta : float
        angle of incidence in radians.
    lam : float
        wavelength. Units of lam must be same as for d0.
    tetm : bool, optional
        False => TE, True => TM. The default is False.

    Returns
    -------
    R : numpy ndarray
        normalized reflected intensity coefficient
    T : numpy ndarray
        normalized transmitted intensity coefficient
    rr : numpy ndarray
        complex field reflection coefficient
    tt : numpy ndarray
        complex field transmission coefficient

    """
    oneD_array(n, 'n', ValueError)
    oneD_array(d0, 'd0', ValueError)
    N = len(n)
    if not (len(d0) == N-2):
        raise ValueError('n and d size mismatch')
        pass
    real_nonnegative_scalar(theta, 'theta', ValueError)
    if theta >= np.pi/2.0:
        raise ValueError('angle of incidence must be less than 90 degrees')
    real_positive_scalar(lam, 'lam', ValueError)
    if not type(tetm) == bool:
        raise TypeError('tetm must be a boolean.')
        
    n = np.asarray(n)
    d0 = np.asarray(d0)

    d = np.hstack((0, d0.reshape(len(d0, )), 0))
        
    kx = 2*np.pi*n[0]*np.sin(theta)/lam
    # sign agrees with measurement convention:
    kz = -np.sqrt((2*np.pi*n/lam)**2 - kx*kx)
    
    if tetm:
        kzz = kz/(n*n)
    else:
        kzz = kz
    
    eep = np.exp(-1j*kz*d)
    eem = np.exp(1j*kz*d)
    
    i1 = np.arange(N-1)
    i2 = np.arange(1, N)
    tin = 0.5*(kzz[i1] + kzz[i2])/kzz[i1]
    ri = (kzz[i1] - kzz[i2])/(kzz[i1] + kzz[i2])
    
    A = np.eye(2, dtype=complex)
    for i in range(N-1):
        A = A @ np.array(tin[i]*np.array([[eep[i], ri[i]*eep[i]],
                                          [ri[i]*eem[i], eem[i]]]))
    
    rr = A[1, 0] / A[0, 0]
    tt = 1 / A[0, 0]
    
    # transmitted power flux (Poynting vector . surface) depends on index of
    # the substrate and angle
    R = np.abs(rr)**2
    if tetm:
        Pn = np.real((kz[-1]/(n[-1]**2)) / (kz[0]/(n[0]**2)))
    else:
        Pn = np.real((kz[-1]/kz[0]))
        pass
    
    T = Pn*np.abs(tt)**2
    tt = np.sqrt(Pn)*tt
    
    return [R, T, rr, tt]
