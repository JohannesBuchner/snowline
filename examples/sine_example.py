import argparse
import numpy as np
from numpy import pi, sin, log
import matplotlib.pyplot as plt

def main(args):

    np.random.seed(2)
    Ndata = args.ndata
    jitter_true = 0.1
    phase_true = 2.0
    period_true = 180
    amplitude_true = args.contrast / Ndata * jitter_true
    paramnames = ['amplitude', 'jitter', 'phase', 'period']
    ndim = len(paramnames)
    
    x = np.linspace(0, 360, 1000)
    y = amplitude_true * sin(x / period_true * 2 * pi + phase_true)
    
    if True:
        plt.plot(x, y)
        x = np.random.uniform(0, 360, Ndata)
        y = np.random.normal(amplitude_true * sin(x / period_true * 2 * pi + phase_true), jitter_true)
        plt.errorbar(x, y, yerr=jitter_true, marker='x', ls=' ')
        plt.savefig('testsine.pdf', bbox_inches='tight')
        plt.close()
    
    
    def loglike(params):
        amplitude, jitter, phase, period = params
        predicty = amplitude * sin(x / period * 2 * pi + phase)
        logl = (-0.5 * log(2 * pi * jitter**2) - 0.5 * ((predicty - y) / jitter)**2).sum()
        return logl
    
    def transform(x):
        z = np.empty_like(x)
        z[0] = 10**(x[0] * 4 - 2)
        z[1] = 10**(x[1] * 1 - 1.5)
        z[2] = 2 * pi * x[2]
        z[3] = 10**(x[3] * 4 - 1)
        #z[:,4] = 2 * pi / x[:,3]
        return z

    loglike(transform(np.ones(ndim)*0.5))
    
    from snowline import ReactiveImportanceSampler
    
    sampler = ReactiveImportanceSampler(paramnames, loglike, transform=transform)
    
    sampler.run()
    
    sampler.print_results()
    
    from getdist import MCSamples, plots

    samples_g = MCSamples(samples=sampler.results['samples'],
                           names=sampler.results['paramnames'],
                           label='Gaussian',
                           settings=dict(smooth_scale_2D=3), sampler='nested')

    mcsamples = [samples_g]
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.num_plot_contours = 3
    g.triangle_plot(mcsamples, filled=False, contour_colors=plt.cm.Set1.colors)
    plt.savefig('testsine_posterior.pdf', bbox_inches='tight')
    plt.close()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--contrast', type=int, default=100,
                        help="Signal-to-Noise level")
    parser.add_argument('--ndata', type=int, default=40,
                        help="Number of simulated data points")

    args = parser.parse_args()
    main(args)
