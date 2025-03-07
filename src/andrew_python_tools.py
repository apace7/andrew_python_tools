#!/usr/bin/env python

import numpy as np
from scipy.stats import chi2
import corner
import os

import matplotlib.pyplot as plt

j_conver_solarkpc_to_gevcm = 2.25e-7
j_conver_solarkpc_to_gevcm_log = 6.64782
d_conver_solarkpc_to_gevcm = 8.55e-15
d_conver_solarkpc_to_gevcm_log = 14.06803
Gravity = 4.301e-6

zscore_limit_4param = chi2.isf(1.-0.9973580714594019, 4)
zscore_limit_3param = chi2.isf(1.-0.9973580714594019, 3)

def vel_combine(bhb, print_option=True):
    """
    combines velocities (or other repeated measurements)

    input:
    array with meausrement, error format

    output:
    0: weighted average
    1: varaince in error/weight
    2: weighted standard deviation
    3: p-value (assuming no variability)
    4: number of measurements
    5: velocity of best measurement (lowest error)
    6: velocity error of best measurement (lowest error)
    printed extra 
    3: weighted variance in the mean, sampling error
    4: chi squared value
    5: p-value (assuming no variability)
    7: probability of coming from chi^2 distribution

    """
#     if print_option:
#         print ("number of measurements", len(bhb.T) )
    if len(bhb.T) ==1:
        if print_option:
            print (bhb[0][0], bhb[1][0], bhb[1][0], bhb[1][0])
            return bhb[0][0], bhb[1][0], bhb[1][0], bhb[1][0]
    elif len(bhb.T) == 0:
        if print_option:
            print ("erorr")
        return
    else:
        if print_option:
            print (bhb)
    w = np.sum(1./bhb[1]**2.)
    mean = np.sum(bhb[0]/bhb[1]**2.)/w
    var2 = np.sqrt(1./w)
    new_err2 = np.sqrt(np.sum((bhb[0] - mean )**2./bhb[1]**2.)/(len(bhb.T-1))/w)
    std_dev = np.sqrt(np.sum((bhb[0] - mean )**2./bhb[1]**2.)*float(len(bhb.T))/(len(bhb.T)-1.)/w)

#     vh_mean = np.average(bhb[0], weights=1./bhb[1]**2)
    chi2_value = np.sum((bhb[0]-mean)**2/bhb[1]**2)
    pdf_chi2 = chi2.pdf(chi2_value, len(bhb.T)-1)
    cdf_chi2 = chi2.cdf(chi2_value, len(bhb.T)-1)
    sf_chi2 = chi2.sf(chi2_value, len(bhb.T)-1)
    ##sf = 	Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
#     ws2 = np.sum((1./bhb[1]**2.)**2)
#     new_thing =ws2 /w**2
#     temp = w/(w**2 - ws2 )
#     s2 = temp*np.sum((bhb[0] - nv )**2./(bhb[1]**2.))
#     s3 = float(len(bhb.T))/(len(bhb.T)-1.)*np.sum((bhb[0] - nv )**2./(bhb[1]**2.)**2)/w**2
    if print_option:
    #   print x, len(bhb.T)
        print (mean, var2, std_dev, new_err2, chi2_value, cdf_chi2, len(bhb.T), pdf_chi2, sf_chi2)
    # return mean, var2, std_dev, new_err2, sf_chi2, cdf_chi2, pdf_chi2, len(bhb.T)

    best_value = bhb[0][np.argmin(bhb[1])], bhb[1][np.argmin(bhb[1])]

    return mean, var2, std_dev, sf_chi2, len(bhb.T), best_value[0], best_value[1]

def chi_sq_variable(v):
    """
    computes Chi_k^2 for set of observations to check for variability.
    input:
    array of with
    v[0] = obs;
    v[1] = error

    output
    0: chi^2_k
    1: prob(chi^2_k), probability to draw this from chi^2 distribution.  Low numbers inticate variability
    2: avr for error testing
    """

    if len(v.T) < 2:
        return 1., 0, np.mean(v[0])
    w = np.sum(1./v[1]**2.)
    vmean = np.sum(v[0]/v[1]**2.)/w

    chi_sq = np.sum( ((v[0] - vmean)/v[1])**2  )/(len(v.T) - 1.)

    return chi_sq,  1. - chi2.cdf(chi_sq, len(v.T)), vmean

def out_data(a, d, length=False, delim='  ', first_line=''):
    os = open(a,'w')

    if first_line != '':
        os.write(first_line + '\n')
    if length:
        os.write(str(len(d))+'\n')

    for i in d:
    # 		t = for_output(i, delim=delim)
        t = ''
        if np.array(i).size == 1:
            t+= str(i)
        else:
            for j in i[:-1]:
                t += str(j) + delim
            t+= str(i[-1])
        t += '\n'
        os.write(t)
    os.close()

def coord_change(ra, dec, r,d, dist=180./np.pi*60., rotate=90, q=1.):
    """
    convert ra,dec coordinates to tangent place centered at r,d
    distance defaults to arcmin
    """
    ra = np.array(ra)
    dec = np.array(dec)
    a = np.pi/180.
    x =  dist*np.cos(a * dec)* np.sin(a*(ra - r))
    y = dist*(np.sin(a*dec) *np.cos(a*d) - np.cos(a*dec)*np.sin(a*d) * np.cos(a*(ra - r)))
    angle = (90.-rotate)*np.pi/180.
    xr = x*np.cos(angle) + y*np.sin(angle)
    yr = -x*np.sin(angle) + y*np.cos(angle)

    return x,y, np.sqrt(x**2 + y**2), xr, yr, np.sqrt(xr**2 + yr**2/q**2)

def dist_mod(mu, mu_e=0, mu_em=0, mu_ep=0):
    """
    computes distance in kpc from distance modulus
    if errors are included it does mcmc samples
    """
    def dm(x):
        return pow(10., x/5.+1.)/1000.

    if mu_e > 0:
        sample = []
        for i in range(100000):
            x = np.random.normal(mu, mu_e)
            sample.append(dm(x))
        a = corner.quantile(np.array(sample), [.5, .1587, .8413])

        return (a[0], a[0]-a[1], a[2]-a[0], (a[2]-a[1])/2.)
    else:
        return dm(mu)
def dist_mod_kpc(dist):
    return (np.log10(dist*1000.) -1.)*5.

def assign_good_star(spectra_stamp_v2, **kwargs):
    vlos_error = kwargs.get('vlos_error', 'vlos_error')
    vlos_skew = kwargs.get('vlos_skew', 'vlos_skew')
    vlos_kurtosis = kwargs.get('vlos_kurtosis', 'vlos_kurtosis')
    sn_ratio = kwargs.get('sn_ratio', 'sn_ratio')
    vel_qual = np.zeros(len(spectra_stamp_v2), dtype=int)
    for i in range(len(spectra_stamp_v2)):
        if spectra_stamp_v2[vlos_error][i]< 5 and spectra_stamp_v2[vlos_error][i] >0 and spectra_stamp_v2[sn_ratio][i] >0 and \
        abs(spectra_stamp_v2[vlos_kurtosis][i])<1 and abs(spectra_stamp_v2[vlos_skew][i])<1:
            vel_qual[i] = 3
        elif spectra_stamp_v2[vlos_error][i]< 5 and spectra_stamp_v2[vlos_error][i] >0 and spectra_stamp_v2[sn_ratio][i] >0 and \
        abs(spectra_stamp_v2[vlos_kurtosis][i])<10 and abs(spectra_stamp_v2[vlos_skew][i])<10:
            vel_qual[i] = 2
        elif spectra_stamp_v2[vlos_error][i]< 5 and spectra_stamp_v2[vlos_error][i] >0 and spectra_stamp_v2[sn_ratio][i] >0:
            vel_qual[i] = 1
        else:
            vel_qual[i] = 0

    return vel_qual
def vel_auto_combine(comb, **kwargs):
    vlos = kwargs.get('vlos', 'vlos')
    vlos_error = kwargs.get('vlos', 'vlos_error')
    source_id = kwargs.get('source_id', 'source_id')
    comb['vlos_unique'] = comb['vlos']
    comb['vlos_error_unique'] = comb['vlos_error']
    comb['num_vlos_good'] = np.zeros(len(comb), dtype=int)

    bad_array2 =[]
    for i in range(len(comb)):
        short_array = comb[comb[source_id] == comb[source_id][i]]
        if len(short_array)==1:
            comb['num_vlos_good'][i] = 1
        else:
            vel_comb = vel_combine(np.array((short_array[vlos], short_array[vlos_error])), print_option=False)
            comb['num_vlos_good'][i] = len(short_array)
            comb['vlos_unique'][i] = vel_comb[0]
            comb['vlos_error_unique'][i] = vel_comb[1] if vel_comb[1] > vel_comb[2] else vel_comb[2]
    return comb
def vel_feh_auto_combine(comb, **kwargs):
        vlos = kwargs.get('vlos', 'vlos')
        vlos_error = kwargs.get('vlos', 'vlos_error')
        feh = kwargs.get('feh', 'feh')
        feh_error = kwargs.get('feh_error', 'feh_error')
        source_id = kwargs.get('source_id', 'source_id')
        comb['vlos_unique'] = comb[vlos]
        comb['vlos_error_unique'] = comb[vlos_error]
        comb['num_vlos_good'] = np.zeros(len(comb), dtype=int)
        comb['feh_unique'] = comb[feh]
        comb['feh_error_unique'] = comb[feh_error]
        comb['num_feh_good'] = np.zeros(len(comb), dtype=int)

        bad_array2 =[]
        for i in range(len(comb)):
            short_array = comb[comb[source_id] == comb[source_id][i]]
            if len(short_array)==1:
                comb['num_vlos_good'][i] = 1
                comb['num_feh_good'][i] = 1
            else:
                vel_comb = vel_combine(np.array((short_array[vlos], short_array[vlos_error])), print_option=False)
                comb['num_vlos_good'][i] = len(short_array)
                comb['vlos_unique'][i] = vel_comb[0]
                comb['vlos_error_unique'][i] = vel_comb[1] if vel_comb[1] > vel_comb[2] else vel_comb[2]
                feh_comb = vel_combine(np.array((short_array[feh], short_array[feh_error])), print_option=False)
                comb['num_feh_good'][i] = len(short_array)
                comb['feh_unique'][i] = feh_comb[0]
                comb['feh_error_unique'][i] = feh_comb[1] # if vel_comb[1] > vel_comb[2] else vel_comb[2]
        return comb
############################################################

def lum(m_x, m_x_sun=4.83):
    return pow(10., -.4*(m_x - m_x_sun) )
def lum_inverse(lum, m_x_sun=4.83 ):
    return np.log10(lum)/(-.4)+m_x_sun

def legacy_viewer(ra,dec):
    print("https://www.legacysurvey.org/viewer/?ra="+str(ra)+"&dec="+str(dec)+"&layer=ls-dr10&zoom=16")


def evid(s):
    fName = s + '.stat'
    fName2 = s + '.evid'
    if os.path.exists(fName):
        x = np.loadtxt(fName, usecols=[1],unpack=True)
    elif os.path.exists(fName2):
        x = np.loadtxt(fName2, usecols=[1],unpack=True)
    else:
        print("file doesn't exist", s)
        return 0.
    return x

def corner_warp(n, ndim, **kwargs):

    d = np.loadtxt(n, usecols=range(ndim+1))
    d2 = np.array([i for i in d if i[0]>1e-8])
    cols = kwargs.get("cols", [])
    if len(cols)>0:
        d2_new = d2[:, cols]

    else:
        d2_new = d2.T[1:].T
    corn = corner.corner(d2_new,quantiles=[0.16, 0.50, 0.84], weights=d2.T[0],show_titles=True, title_kwargs={"fontsize": 10},plot_datapoints = False,bins=[20]*len(d2_new.T),levels=[ 0.393469, 0.864665, 0.988891],**kwargs)
    if kwargs.get('savefig'):
        plt.savefig(kwargs.get('savefig'))
    plt.show()

    temp = []
    for i in range(ndim):
        x = corner.quantile(d2.T[1:][i], [.5, .1587, .8413], weights=d2.T[0])
        print(x[0], x[0]-x[1], x[2]-x[0])
        temp.append(x)
    return temp


def line_bhb_bss(x):
    c = [-0.03684, 0.20021, 0.29969,
        0.94966, -1.50963, 1.11371]
    t =0
    for i in range(6):
        t+=c[i]*x**i
    return t
def line_bhb_bss_grz(x):
    c = [-0.11368, 0.66993, -0.12911,
        0.694766, -1.42272, 1.07163]
    t =0
    for i in range(6):
        t+=c[i]*x**i
    return t
x = np.arange(-0.4, 0.2, .001)
temp = np.transpose([x,line_bhb_bss(x)])
temp2 = np.array([
[temp[-1][0],temp[0][1]],
         [temp[0][0],temp[0][1]]])
box_bss = np.concatenate((temp, temp2))

def photometry_transform_sdss_to_des(gSDSS, rSDSS):
    gDES = gSDSS - 0.104*(gSDSS - rSDSS ) + 0.01
    rDES = rSDSS - 0.102*(gSDSS - rSDSS ) + 0.02
    return gDES, rDES

def betw(x, x1, x2): return (x >= x1) & (x < x2)

def j_factor_prediction(sigma, dist, rhalf, jo=17.87):
    ## http://arxiv.org/abs/1802.06811
    ## units: km/s, kpc, pc
    return np.log10(10**jo * (sigma/5.)**4 * (dist/100.)**(-2) * (rhalf/100.)**(-1))
def mhalf(rhalf, sigma, method='wolf'):
    return 930 *sigma*sigma*rhalf

def compute_zscore(data, **kwargs):
    parallax_mean_error = kwargs.get("parallax_mean_error",0.0002 )
    parallax_mean = kwargs.get("parallax_mean",0 )
    
    pmra_sigma = kwargs.get("pmra_sigma",0 )
    pmdec_sigma = kwargs.get("pmra_sigma",0 )
    pmra_mean_error = kwargs.get("pmra_mean_error",0 )
    pmdec_mean_error = kwargs.get("pmdec_mean_error",0 )
    pmra_mean = kwargs.get("pmra_mean",0 )
    pmdec_mean = kwargs.get("pmdec_mean",0 )
    
    rse_nplx = kwargs.get("rse_nplx", 1)
    rse_npmra = kwargs.get("rse_npmra", 1)
    rse_npmdec = kwargs.get("rse_npmdec", 1)
    
    vlos_mean = kwargs.get("vlos_mean",0 )
    vlos_mean_error = kwargs.get("vlos_mean_error",0 )
    sigma_vlos = kwargs.get("sigma_vlos",0 )
    
    col_vlos = kwargs.get("col_vlos",'vlos' )
    col_vlos_error = kwargs.get("col_vlos_error",'vlos_error' )
    zscore_option = kwargs.get('zscore_option', "PM_VLOS")
    parallax_error = kwargs.get('col_parallax_error', "parallax_error")
    pmra_error = kwargs.get('col_pmra_error', "pmra_error")
    pmdec_error = kwargs.get('col_pmdec_error', "pmdec_error")
    parallax_pmra_corr = kwargs.get('col_parallax_pmra_corr', "parallax_pmra_corr")
    parallax_pmdec_corr = kwargs.get('col_parallax_pmdec_corr', "parallax_pmdec_corr")
    pmra_pmdec_corr = kwargs.get('col_pmra_pmdec_corr', "pmra_pmdec_corr")
    parallax = kwargs.get('col_parallax', "parallax")
    pmra = kwargs.get('col_pmra', "pmra")
    pmdec = kwargs.get('col_pmdec', "pmdec")
    if zscore_option == "PM_VLOS":
        diffvec = np.array([data[parallax]-parallax_mean, data[pmra]-pmra_mean, data[pmdec]-pmdec_mean, data[col_vlos]-vlos_mean, ]).T

        dmat = np.zeros((4,4))
        dmat[0][0] = parallax_mean_error**2
        dmat[1][1] = pmra_mean_error**2 + pmra_sigma**2
        dmat[2][2] = pmdec_mean_error**2 + pmdec_sigma**2
        dmat[3][3] = vlos_mean_error**2 + sigma_vlos**2

        covmat = np.zeros((4,4))
        covmat[0][0] = data[parallax_error]**2*rse_nplx**2
        covmat[1][1] = data[pmra_error]**2*rse_npmra**2
        covmat[2][2] = data[pmdec_error]**2*rse_npmdec**2
        covmat[0][1] = data[parallax_error]*data[pmra_error]*data[parallax_pmra_corr]*rse_nplx*rse_npmra
        covmat[0][2] = data[parallax_error]*data[pmdec_error]*data[parallax_pmdec_corr]*rse_nplx*rse_npmdec
        covmat[1][2] = data[pmra_error]*data[pmdec_error]*data[pmra_pmdec_corr]*rse_npmra*rse_npmdec
        covmat[1][0] = covmat[0][1]
        covmat[2][0] = covmat[0][2]
        covmat[2][1] = covmat[1][2]
        covmat[3][3] = data[col_vlos_error]**2
        cinv = np.linalg.inv(covmat+dmat)

        zscore = np.dot(diffvec, np.dot(cinv,diffvec.T))
        return zscore
    elif zscore_option == "PM":
        diffvec = np.array([data[parallax]-parallax_mean, data[pmra]-pmra_mean, data[pmdec]-pmdec_mean,  ]).T
        dmat = np.zeros((3,3))
        dmat[0][0] = parallax_mean_error**2
        dmat[1][1] = pmra_mean_error**2 + pmra_sigma**2
        dmat[2][2] = pmdec_mean_error**2 + pmdec_sigma**2

        covmat = np.zeros((3,3))
        covmat[0][0] = data[parallax_error]**2*rse_nplx**2
        covmat[1][1] = data[pmra_error]**2*rse_npmra**2
        covmat[2][2] = data[pmdec_error]**2*rse_npmdec**2
        covmat[0][1] = data[parallax_error]*data[pmra_error]*data[parallax_pmra_corr]*rse_nplx*rse_npmra
        covmat[0][2] = data[parallax_error]*data[pmdec_error]*data[parallax_pmdec_corr]*rse_nplx*rse_npmdec
        covmat[1][2] = data[pmra_error]*data[pmdec_error]*data[pmra_pmdec_corr]*rse_npmra*rse_npmdec
        covmat[1][0] = covmat[0][1]
        covmat[2][0] = covmat[0][2]
        covmat[2][1] = covmat[1][2]
        cinv = np.linalg.inv(covmat+dmat)

        zscore = np.dot(diffvec, np.dot(cinv,diffvec.T))
        return zscore
    

def accret_line_BK2023(LZ):
    # https://ui.adsabs.harvard.edu/abs/2023MNRAS.tmp.2222B/abstract 
    if LZ < -.58:
        return -1.3
    elif LZ < 0.58:
        return -1.4 + 0.3*LZ**2
    else:
         return -1.325 + 0.075*LZ**2

def re_king(rc, rt):
    ## wolf et al 2010
    c = np.log10(rt/rc)
    a = [0.5439, 0.1044, 1.5618, -0.7559, .2572]
    re_rc = 0.
    for i in range(5):
        re_rc += a[i] * c**i 
    return rc * re_rc
