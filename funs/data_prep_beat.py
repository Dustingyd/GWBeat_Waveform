from __future__ import division

import lal
import lalsimulation
from lal.antenna import AntennaResponse
from lal import MSUN_SI, C_SI, G_SI

import os
import sys
import argparse
import time
import numpy as np
from six.moves import cPickle
from scipy.signal import filtfilt, butter
from scipy.optimize import brentq
from scipy import integrate, interpolate
import sympy as sp

import math
import scipy.integrate as integrate
import cmath
from scipy import special
from numpy import sqrt, pi, cos
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

import multiprocessing
from functools import partial

class SRparams:
    def __init__(self,M0,alpha0,as0,Ms0_0,Ms0_1,z,ra,dec,iota,phi,psi):
        self.M0 = M0
        self.alpha0 = alpha0
        self.as0 = as0
        self.Ms0_0 = Ms0_0
        self.Ms0_1 = Ms0_1
        self.z = z
        self.ra = ra
        self.dec = dec
        self.iota = iota
        self.phi = phi
        self.psi = psi


data_In = np.loadtxt('funs/DeltaMs1_R_M.csv',delimiter=",")
alpha0_In = np.unique(data_In[::,0])
as0_In = np.unique(data_In[::,1])
DeltaMs1_R_M = np.reshape(data_In[::,2], (len(alpha0_In), len(as0_In)))
interp = RegularGridInterpolator((alpha0_In, as0_In), DeltaMs1_R_M,       
                                 bounds_error=False, method='cubic')
def Delta_MsRM0_Dom(m0, alpha0, as0):
    if m0==1:
        return float(interp((alpha0,as0)))

data2_In = np.loadtxt('funs/M_times_f.csv',delimiter=",")
as0_In2 = data2_In[::,0]
M_times_f_Out = data2_In[::,1]
interp_M_times_f = interp1d(as0_In2, M_times_f_Out, kind='cubic')

def M_times_f(m0, as0):
    if m0==1:
        return float(interp_M_times_f(as0))

def KerrKGeqEigenvalueAsymR(nn, ll, mm, astar, alpha):
    # 计算主量子数
    nbar = nn + ll + 1
    # 计算表达式
    term1 = 1
    term2 = -alpha**2 / (2 * nbar**2)
    term3 = -alpha**4 / (8 * nbar**4)
    term4 = ((2 * ll - 3 * nbar + 1) * alpha**4) / (nbar**4 * (ll + 0.5))
    term5 = (2 * astar * mm * alpha**5) / (nbar**3 * ll * (ll + 0.5) * (ll + 1))
    
    # 返回计算结果
    return alpha * (term1 + term2 + term3 + term4 + term5)

def aStarC(m, alpha):
  """
  计算超辐射的 aStarC 参数

  参数：
    m: 超辐射模式
    alpha: M*\mu

  返回值：
    aStarC 值
  """
  M_omega = KerrKGeqEigenvalueAsymR(0, m, m, 0, alpha)
  if M_omega > m / 2 or alpha > m:
    return 1
  else:
    return (4 * m * M_omega) / (m**2 + (2 * M_omega)**2)

# def Delta_MsRM0_Dom(m0, alpha0, as0):
#   def M_omega(alpha, ass):
#     return KerrKGeqEigenvalueAsymR(0, m0, m0, ass, alpha)

#   def as1(x):
#     return  (4 * m0 * M_omega(x * alpha0, 0)) / (m0**2 + (2 * M_omega(x * alpha0, 0))**2)

#   def eq(x):
#     return m0 * (1 - x) - (as0 - as1(x) * x**2) * M_omega(alpha0 * x, 0) / x
      
#   z =  sp.Symbol('z')
#   try:
#       sol = sp.nsolve(eq(z) , z, 0.9)
#       if sol > 0 and sol <= 1:
#         return 1 - sol
#       else:
#         return 0
#   except (ValueError, NotImplementedError) as e:
#         return 0


def MsRM_Dom(m0, alpha0, as0, Ms0RM0):
    M1 = 1 - Delta_MsRM0_Dom(m0, alpha0, as0)
    sol = (1 - M1 + Ms0RM0)/M1
    return sol

def beta(n, l):
    numerator = math.factorial(n + 2 * l + 1)
    denominator = math.factorial(n) * (n + l + 1)**(2 * l + 4)
    return numerator / denominator

def MsRM_Sub(m0, alpha0, as0, Ms0RM0_Dom, Ms0RM0_Sub):
    M1 = 1 - Delta_MsRM0_Dom(m0, alpha0, as0)
    MsRM0_Dom = 1 - M1 + Ms0RM0_Dom
    MsRM0_Sub = Ms0RM0_Sub * (MsRM0_Dom / Ms0RM0_Dom)**(beta(1, m0) / beta(0, m0))
    return MsRM0_Sub / M1


def GWs_SR(M0, Ms2p, Ms3p, z, mu, theta, iniphi, t):
    """
    M0: 单位为太阳质量
    """
    H0 = 1
    GMsun = 1.17167 * (10**-23) #单位为 1/H0
    rg = 1
    omega2p = KerrKGeqEigenvalueAsymR(0, 1, 1, 0, mu)
    omega3p = KerrKGeqEigenvalueAsymR(1, 1, 1, 0, mu)

    omega1 = 2 * omega2p
    omega2 = 2 * omega3p
    omega3 = omega2p + omega3p
    omega4 = omega3p - omega2p

    N2p = Ms2p / omega2p
    N3p = Ms3p / omega3p

    U1 = 1/12 * (22 + 3j * np.pi) * math.sqrt(np.pi / 5) * rg**6 * mu**10
    U2 = (64 * (22 + 3j * np.pi) * math.sqrt(np.pi / 5) * rg**6 * mu**10) / 2187
    U3 = 4/81 * (22 + 3j * np.pi) * math.sqrt(np.pi / 5) * rg**6 * mu**10

    if N2p* N3p / (omega2p * omega3p) < 0:
        print(M0, Ms2p, Ms3p, z, mu, iniphi, t)
    
    def rc(zz):
        def integrand(zi):
            return 1 / math.sqrt((1 + zi)**3 * 3/10 + 7/10)
        integral = integrate.quad(integrand, 0, zz)[0]
        return integral / (H0 * (M0 * GMsun))  

    def h_plus_term(Ni, Nj, omegai, omegaj, U, omegaT, phi0):
        return math.sqrt(Ni* Nj / (omegai * omegaj)) * abs(U) / omegaT**2 * cos(omegaT*t - cmath.phase(U) + phi0)

    def h_cross_term(Ni, Nj, omegai, omegaj, U, omegaT, phi0):
        return math.sqrt(Ni* Nj / (omegai * omegaj)) * abs(U) / omegaT**2 * math.sin(omegaT*t - cmath.phase(U) + phi0)

    h_plus = -1 / (math.sqrt(2 * np.pi) * rc(z)) * 1/8 * math.sqrt(5/np.pi) * (cos(2*theta) + 3) * (
        2 * h_plus_term(N2p, N2p, omega2p, omega2p, U1, omega1, iniphi) +
        2 * h_plus_term(N3p, N3p, omega3p, omega3p, U2, omega2, iniphi) +
        4 * h_plus_term(N2p, N3p, omega2p, omega3p, U3, omega3, iniphi)
        )

    h_cross = -1 / (math.sqrt(2 * np.pi) * rc(z)) * 1/2 * math.sqrt(5/np.pi) * cos(theta) * (
        2 * h_cross_term(N2p, N2p, omega2p, omega2p, U1, omega1, iniphi) +
        2 * h_cross_term(N3p, N3p, omega3p, omega3p, U2, omega2, iniphi) +
        4 * h_cross_term(N2p, N3p, omega2p, omega3p, U3, omega3, iniphi)
        )
        
    return h_plus, h_cross

def gen_beat(t, M0, z, theta, iniphi, alpha0, as0, Ms0RM0_Dom, Ms0RM0_Sub):
    M1 = 1 - Delta_MsRM0_Dom(1, alpha0, as0) 
    Ms2p = MsRM_Dom(1, alpha0, as0, Ms0RM0_Dom) * M1
    Ms3p = MsRM_Sub(1, alpha0, as0, Ms0RM0_Dom, Ms0RM0_Sub) * M1
    t_unit = M0 * 4.92564 * (10**(-6))
    res = GWs_SR(M0, Ms2p, Ms3p, z, alpha0, theta, iniphi, t/t_unit/(1+z))
    return res

safe = 2    # define the safe multiplication scale for the desired time length

def tukey(M,alpha=0.5):
    """
    Tukey window code copied from scipy
    """
    n = np.arange(0, M)
    width = int(np.floor(alpha*(M-1)/2.0))
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]

    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = np.ones(n2.shape)
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
    w = np.concatenate((w1, w2, w3))

    return np.array(w[:M])

def gen_psd(fs,T_obs,op='AdvDesign',det='H1'):
    """
    generates noise for a variety of different detectors
    """
    N = T_obs * fs          # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)
    df = 1 / T_obs          # the frequency resolution
    psd = lal.CreateREAL8FrequencySeries(None, lal.LIGOTimeGPS(0), 0.0, df,lal.HertzUnit, N // 2 + 1)

    if det=='H1' or det=='L1':
        if op == 'AdvDesign':
            lalsimulation.SimNoisePSDAdVDesignSensitivityP1200087(psd, 10.0)
        elif op == 'AdvEarlyLow':
            lalsimulation.SimNoisePSDAdVEarlyLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvEarlyHigh':
            lalsimulation.SimNoisePSDAdVEarlyHighSensitivityP1200087(psd, 10.0)
        elif op == 'AdvMidLow':
            lalsimulation.SimNoisePSDAdVMidLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvMidHigh':
            lalsimulation.SimNoisePSDAdVMidHighSensitivityP1200087(psd, 10.0)
        elif op == 'AdvLateLow':
            lalsimulation.SimNoisePSDAdVLateLowSensitivityP1200087(psd, 10.0)
        elif op == 'AdvLateHigh':
            lalsimulation.SimNoisePSDAdVLateHighSensitivityP1200087(psd, 10.0)
        else:
            print('unknown noise option')
            exit(1)
    else:
        print('unknown detector - will add Virgo soon')
        exit(1)

    return psd

def get_snr(data,T_obs,fs,psd):
    """
    computes the snr of a signal given a PSD starting from a particular frequency index
    """

    N = T_obs*fs
    df = 1.0/T_obs
    dt = 1.0/fs
    fidx = 0

    win = tukey(N,alpha=1.0/8.0)
    idx = np.argwhere(psd>0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0/psd[idx]

    xf = np.fft.rfft(data*win)*dt
    SNRsq = 4.0*np.sum((np.abs(xf[fidx:])**2)*invpsd[fidx:])*df
    return np.sqrt(SNRsq)

def whiten_data(data,duration,sample_rate,psd,flag='td'):
    """
    Takes an input timeseries and whitens it according to a psd
    """

    if flag=='td':
        # FT the input timeseries - window first
        win = tukey(duration*sample_rate,alpha=1.0/8.0)
        xf = np.fft.rfft(win*data)
    else:
        xf = data

    # deal with undefined PDS bins and normalise
    idx = np.argwhere(psd>0.0)
    invpsd = np.zeros(psd.size)
    invpsd[idx] = 1.0/psd[idx]
    xf *= np.sqrt(2.0*invpsd/sample_rate)

    # Detrend the data: no DC component.
    xf[0] = 0.0

    if flag=='td':
        # Return to time domain.
        x = np.fft.irfft(xf)
        return x
    else:
        return xf

def make_beat(hp,hc,fs,ra,dec,psi,det,verbose):
    """
    turns hplus and hcross into a detector output
    applies antenna response and
    and applies correct time delays to each detector
    """

    # make basic time vector
    tvec = np.arange(len(hp))/float(fs)

    # compute antenna response and apply
    resp = AntennaResponse(det, ra, dec, psi,scalar=True, vector=True, times=0.0)
    Fp = resp.plus
    Fc = resp.cross
    ht = hp*Fp + hc*Fc     # overwrite the timeseries vector to reuse it

    # compute time delays relative to Earth centre
    frDetector =  lalsimulation.DetectorPrefixToLALDetector(det)
    tdelay = lal.TimeDelayFromEarthCenter(frDetector.location,ra,dec,0.0)
    if verbose:
        print('{}: computed {} Earth centre time delay = {}'.format(time.asctime(),det,tdelay))

    # interpolate to get time shifted signal
    ht_tck = interpolate.splrep(tvec, ht, s=0)
    hp_tck = interpolate.splrep(tvec, hp, s=0)
    hc_tck = interpolate.splrep(tvec, hc, s=0)
    tnew = tvec + tdelay
    new_ht = interpolate.splev(tnew, ht_tck, der=0,ext=1)
    new_hp = interpolate.splev(tnew, hp_tck, der=0,ext=1)
    new_hc = interpolate.splev(tnew, hc_tck, der=0,ext=1)

    return new_ht, new_hp, new_hc

def gen_signal(fs,T_obs,pars,dets=['H1'],verbose=True):
    """
    generates the signal of GW beats
    """
    N = T_obs * fs      # the total number of time samples
    dt = 1 / fs             # the sampling time (sec)

    # make waveform
    t_list = np.arange(0, T_obs, dt)

    # make waveform
    t_list = np.arange(0, T_obs, dt)
    orig_hp = []
    orig_hc = []

    for t in t_list:
        hp_n, hc_n = gen_beat(t, pars.M0, pars.z, pars.iota, pars.phi, pars.alpha0, pars.as0, pars.Ms0_0, pars.Ms0_1)
        orig_hp.append(hp_n)
        orig_hc.append(hc_n)
    # make aggressive window to cut out signal in central region
    # window is non-flat for 1/8 of desired Tobs
    # the window has dropped to 50% at the Tobs boundaries
    # win = np.zeros(N)
    # tempwin = tukey(int((16.0/15.0)*N/safe),alpha=1.0/8.0)
    # win[int((N-tempwin.size)/2):int((N-tempwin.size)/2)+tempwin.size] = tempwin
    # win = tukey(N,alpha=1.0/N)

    # loop over detectors
    psds = [gen_psd(fs,T_obs,op='AdvDesign',det=d) for d in dets]
    ndet = len(psds)
    ts = np.zeros((ndet,N))
    hp = np.zeros((ndet,N))
    hc = np.zeros((ndet,N))
    intsnr = []
    j = 0
    for det,psd in zip(dets,psds):

        # make signal - apply antenna and shifts
        ht_shift, hp_shift, hc_shift = make_beat(np.array(orig_hp),np.array(orig_hc),fs,pars.ra,pars.dec,pars.psi,det,verbose)

        # apply aggressive window to cut out signal in central region
        # window is non-flat for 1/8 of desired Tobs
        # the window has dropped to 50% at the Tobs boundaries
        ts[j,:] = ht_shift #* win
        hp[j,:] = hp_shift #* win
        hc[j,:] = hc_shift #* win

        # compute SNR of pre-whitened data
        intsnr.append(get_snr(ts[j,:],T_obs,fs,psd.data.data))

        j += 1

    if verbose:
        print('{}: computed the network SNR = {}'.format(time.asctime(),intsnr))

    return ts, hp, hc