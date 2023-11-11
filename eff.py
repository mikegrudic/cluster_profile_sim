import streamlit as st
import pandas as pd
import numpy as np
import bokeh.plotting as bk
from scipy.integrate import cumtrapz
from bokeh.models import Range1d
import matplotlib.pyplot as plt
import emcee
from multiprocessing import Pool
from scipy.optimize import minimize
from numba import njit

plt.clf()

gamma = st.sidebar.slider(r'Slope $\gamma$',2.01,max_value=5., value=2.5)
mu_max = norm = 2*np.pi/(gamma - 2)
background = 10**st.sidebar.slider(r'$\log$ background',-3.,max_value=0., value=-1.) * mu_max
N = int(10**st.sidebar.slider(r'$\log$ N',min_value=1.,max_value=6., value=4.))
aperture = st.sidebar.slider(r'Aperture',0.,max_value=100., value=100.)

show_model = st.sidebar.checkbox("Show Model",True)
seed = st.sidebar.number_input("Seed",42)

np.random.seed(seed)
r = np.logspace(-2,np.log10(aperture),1000)

Rmax = r.max()
cdf = 1 - (1 + r**2)**(1-gamma/2)
r50 = np.interp(0.5,cdf,r)
sigma = norm * (1 + r**2)**(-gamma/2)
ecdf = cumtrapz(2*np.pi*r*(sigma + background), r,initial=0); ecdf_norm = ecdf.max()
fac =  np.trapz(r*sigma, r)/np.trapz(r*(sigma + background), r)
ecdf /= ecdf.max()
samples_within_r50 = 0
sample_size = N
while samples_within_r50 < N:
    samples = np.interp(np.sort(np.random.rand(sample_size)), ecdf, r)
    samples_within_r50 = (samples < 1).sum()
    sample_size *= 2
samples = samples#[samples < aperture]

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.loglog(r, sigma,color='red',ls='solid',label="Ground-truth cluster")
ax.plot([r50,r50],[0,1e6],color='red',label=r"Ground-truth $R_{\rm eff}$",ls='dashdot')
ax.loglog(r,np.repeat(background,len(r)),label="Background",ls='dotted',color='red')
rbins = np.logspace(-2,min(2,np.log10(aperture)),15)
r_avg = (rbins[1:]*rbins[:-1])**0.5
N = np.histogram(samples,rbins)[0]
sigma_histogram = N/np.diff(np.pi*rbins**2) * ecdf_norm / len(samples)
sigma_histogram[sigma_histogram==0] = np.nan
ax.errorbar(r_avg[N>3],sigma_histogram[N>3],sigma_histogram[N>3]/np.sqrt(N[N>3]),label="Observed Counts",capsize=1,lw=1,color='black')
ax.loglog(r, sigma+background,label="Total",ls='dashed',color='red')

if show_model:
#    @njit(parallel=True)
    def lossfunc(x):
        logmu0, logbackground, loga, gam = x
        mu, background, a = 10**logmu0, 10**logbackground, 10**loga
        sigma_model = mu * (1+(r_avg/a)**2)**(-gam/2) + background
        mask = (~np.isnan(sigma_histogram))*(sigma_histogram>0)#*(r_avg>0.1)
#        N_model = sigma_model * np.diff(np.pi*rbins**2)
        return np.sum(np.log((sigma_model/sigma_histogram)[mask])**2 * N[mask])

#     def lossfunc(x,prior=False):
#     #    print(x)
# #        if prior(x) == -np.inf: return -np.inf
#         logmu0, logbackground, loga, g = x
#         mu, background, a = 10**logmu0, 10**logbackground, 10**loga
#         norm = (background*(-2 + g)*Rmax**2)/2. + mu*(a**2 - a**g*(a**2 + Rmax**2)**(1 - g/2.))
#         lnprob = np.sum(np.log(samples * (mu/(1 + (samples/a)**2)**(-g/2) + background)) - np.log(norm))# + prior(x)
#         if np.isnan(lnprob): lnprob = -np.inf
#         return -lnprob


    p0 = (np.log10(mu_max),np.log10(background), 0., gamma)

    sol = minimize(lossfunc, np.array(p0)+0.01,tol=1e-4)
    x = sol.x
    H = sol.hess_inv
    ax.loglog(r, 10**x[0] *  (1+(r/10**x[2])**2)**(-x[3]/2) + 10**x[1],label="Best-fit Model",color='blue',ls="dashed")
    sigma_model = 10**x[0] *  (1+(r/10**x[2])**2)**(-x[3]/2)
    ax.loglog(r, sigma_model,label="Best-fit Cluster Profile",color='blue',ls="solid")

    Reff_model = 10**x[2]*(2**(2/(x[3]-2)) - 1 )**0.5
   # st.write(r"$\gamma=%g$"%x[3])
    ax.text(6,15,r"$\gamma_{\rm model}=%4.3g$"%(x[3]),fontsize=12,color="blue")

    if np.isnan(Reff_model):
        ax.text(1e1,1e1,r"$R_{\rm eff}$ undefined!",fontsize=12)
    ax.plot([Reff_model,Reff_model],[0,1e6],color='blue',label=r"Model $R_{\rm eff}$",ls='dashdot')


#ax.loglog(r_avg, 10**x[0] *  (1+(r_avg/10**x[2])**2)**(-x[3]/2),label="Model Background",color='darkblue',ls='dotted')

#    sigma_model = 

# # reverse-inference part
# def log_prob(x, samples):
# #    print(x)
#     if prior(x) == -np.inf: return -np.inf
#     logmu0, logbackground, loga, gamma = x

#     mu, background, a = 10**logmu0, 10**logbackground, 10**loga
#     norm = (background*(-2 + gamma)*Rmax**2)/2. + mu*(a**2 - a**gamma*(a**2 + Rmax**2)**(1 - gamma/2.))
#     #print(np.sum(np.log(samples * (mu/(1 + (samples/a)**2)**(-gamma/2) + background) / (gamma - 2)) - np.log(norm)) + prior(x))
#     lnprob = np.sum(np.log(samples * (mu/(1 + (samples/a)**2)**(-gamma/2) + background) / (gamma - 2)) - np.log(norm)) + prior(x)
#     if np.isnan(lnprob): lnprob = -np.inf
#     return lnprob
#     #return -0.5 * np.sum(ivar * x ** 2)

# def prior(x):
#     mu0, background, a, gamma = x
#     if gamma < 0 or gamma > 10: return -np.inf
#     if mu0 < -3 or mu0 > 10: return -np.inf
#     if background < -10 or background > mu0+3: return -np.inf
#     if a < -2 or a > 2: return -np.inf
#     return 1.


# ndim, nwalkers = 4, 100
# p0 = np.array([0.,0.,0.,2.5]) + 1e-2*np.random.normal(size=(nwalkers,ndim))

# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[samples])
# samples = sampler.run_mcmc(p0, 1000,progress=True)

ax.set(ylim=[10**int(np.log10(1e-4*mu_max)), 10**int(np.log10(mu_max)+1)],ylabel="$\mu$", xlabel=r"$R$",xlim=[1e-2,1e2])
ax.legend(loc=3)
st.pyplot(fig)