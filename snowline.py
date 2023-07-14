"""Calculates the Bayesian evidence and posterior samples of arbitrary monomodal models."""

from __future__ import print_function
from __future__ import division

import os
import sys
import logging
import warnings

import numpy as np
import scipy.spatial

from iminuit import Minuit
try:
    from iminuit.util import HesseFailedWarning
except ImportError:
    from iminuit.iminuit_warnings import HesseFailedWarning
from pypmc.sampler.importance_sampling import combine_weights
from pypmc.density.mixture import create_gaussian_mixture
from pypmc.density.gauss import Gauss
from pypmc.sampler.importance_sampling import ImportanceSampler
from pypmc.tools.convergence import ess
from pypmc.mix_adapt.variational import GaussianInference


__all__ = ['ReactiveImportanceSampler']
__author__ = """Johannes Buchner"""
__email__ = 'johannes.buchner.acad@gmx.com'
__version__ = '0.6.3'


# Some parts are from the nnest library by Adam Moss (https://github.com/adammoss/nnest)
def create_logger(module_name, log_dir=None, level=logging.INFO):
    """
    Set up the logging channel `module_name`.

    Append to ``debug.log`` in `log_dir` (if not ``None``).
    Write to stdout with output level `level`.

    If logging handlers are already registered, no new handlers are
    registered.
    """
    logger = logging.getLogger(str(module_name))
    first_logger = logger.handlers == []
    if log_dir is not None and first_logger:
        # create file handler which logs even debug messages
        handler = logging.FileHandler(os.path.join(log_dir, 'debug.log'))
        msgformat = '%(asctime)s [{}] [%(levelname)s] %(message)s'
        formatter = logging.Formatter(
            msgformat.format(module_name), datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    if first_logger:
        logger.setLevel(logging.DEBUG)
        # if it is new, register to write to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter('[{}] %(message)s'.format(module_name))
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


"""Square root of a small number."""
SQRTEPS = (float(np.finfo(np.float64).eps))**0.5


# Some parts are from the Nestle library by Kyle Barbary (https://github.com/kbarbary/nestle)
def resample_equal(samples, weights, N=None, rstate=None):
    """Resample the samples so that the final samples all have equal weight.

    Each input sample appears in the output array either
    `floor(weights[i] * N)` or `ceil(weights[i] * N)` times, with
    `floor` or `ceil` randomly selected (weighted by proximity).

    Parameters
    ----------
    samples : `~numpy.ndarray`
        Unequally weight samples returned by the nested sampling algorithm.
        Shape is (N, ...), with N the number of samples.
    weights : `~numpy.ndarray`
        Weight of each sample. Shape is (N,).
    N : int
        Number of samples to draw. if None, len(weights) is used.

    Returns
    -------
    equal_weight_samples : `~numpy.ndarray`
        Samples with equal weights, same shape as input samples.

    Examples
    --------
    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
    >>> nestle.resample_equal(x, w)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [ 3.,  3.]])

    Notes
    -----
    Implements the systematic resampling method described in
    `this PDF <http://people.isy.liu.se/rt/schon/Publications/HolSG2006.pdf>`_.
    Another way to sample according to weights would be::

        N = len(weights)
        new_samples = samples[np.random.choice(N, size=N, p=weights)]

    However, the method used in this function is less "noisy".

    """
    if abs(np.sum(weights) - 1.) > SQRTEPS:  # same tol as in np.random.choice.
        raise ValueError("weights do not sum to 1 (%g)" % np.sum(weights))

    if rstate is None:
        rstate = np.random

    if N is None:
        N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (rstate.random() + np.arange(N)) / N

    idx = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1

    rstate.shuffle(idx)
    return samples[idx]

def _make_initial_proposal(optu, cov):
    # 1) find the middle between the estimate and the full prior
    stdevs = np.diag(cov)**0.5
    # mid_stdevs = np.exp((np.log(stdevs) + np.log(1)) / 2)
    mid_stdevs = stdevs**0.5
    # use that as a wide, uncorrelated gaussian proposal
    verywidecov = np.diag(mid_stdevs**2)
    # 2) blow up the current covariance
    widecov = cov * 100**2
    # 3) narrow the current covariance
    narrowcov = cov / 100**2
    # combine
    return [optu, optu, optu, optu], [cov, widecov, verywidecov, narrowcov], [0.7, 0.1, 0.1, 0.1]

def _make_proposal(samples, weights, optu, cov, invcov):
    # split samples into 3 equally large groups, by L
    w1, w2 = np.percentile(weights[weights>0], [33, 66])

    means = [optu]
    covs = [cov]
    chunk_weights = [1]
    # for each group (top: L1 < L, mid: L1 > L > L2, bottom: L < L2)

    cov_guess = cov
    for mask in weights >= w1, ~np.logical_or(weights >= w2, weights <= w1), weights <= w2:
        mask = np.logical_and(mask, weights > 0)
        if not mask.any():
            continue
        # assume H as distance metric
        # find most distant point from ML (u)
        dists = scipy.spatial.distance.cdist(samples[mask, :], [optu], 'mahalanobis', VI=invcov).flatten()
        # maximum size of clusters:

        handled = np.zeros(len(dists), dtype=bool)

        # repeat recursively until no points left
        while not handled.all():
            samples_todo = samples[mask, :][~handled, :]

            # find most distant point, which is used as the center
            i = dists[~handled].argmax()
            # add all points within distance until peak is included
            d = dists[~handled][i]
            #    but include at most a distance of maxdistance

            dists_todo = scipy.spatial.distance.cdist(samples_todo, [samples_todo[i, :]], 'mahalanobis', VI=invcov).flatten()
            selected = dists_todo <= d
            cluster = samples_todo[selected]
            #print("  accreted %d (of %d to do)" % (len(cluster), (~handled).sum()), 'from', samples_todo[i, :])
            handled[~handled] = selected

            if len(cluster) < cluster.shape[1]:
                continue

            # print(np.diag(np.var(cluster, axis=0)))
            # cov_guess = np.diag(np.var(cluster, axis=0))
            try:
                cov_local = np.cov(cluster, rowvar=0)
                # check that it is positive-definite
                np.linalg.cholesky(cov_local)
                if not np.all(np.linalg.eigvals(cov_local) > 0):
                    continue
            except np.linalg.LinAlgError:
                cov_local = cov_guess
                # reject, too few points in cluster
                continue
            
            assert np.isfinite(cluster).all(), cluster[~np.isfinite(cluster)]
            assert np.isfinite(cov_local).all(), (cov_local, cov_local[np.isfinite(cov_local)])
            means.append(np.mean(cluster, axis=0))
            covs.append(cov_local)
            chunk_weights.append(1)

    chunk_weights = np.asarray(chunk_weights) / np.sum(chunk_weights)

    mix = create_gaussian_mixture(means, covs, weights=chunk_weights)
    return mix


class ReactiveImportanceSampler(object):
    """Sampler with reactive exploration strategy.

    Storage & resume capable, optionally MPI parallelised.
    """

    def __init__(self,
                 param_names,
                 loglike,
                 transform=None,
                 resume=True,
                 run_num=None,
                 num_test_samples=2,
                 ):
        """Initialise importance sampler.

        Parameters
        -----------
        param_names: list of str, names of the parameters.
            Length gives dimensionality of the sampling problem.

        loglike: function
            log-likelihood function.
            Receives multiple parameter vectors, returns vector of likelihood.
        transform: function
            parameter transform from unit cube to physical parameters.
            Receives multiple cube vectors, returns multiple parameter vectors.

        log_dir: str
            where to store output files
        resume: continue previous run if available.

        num_test_samples: int
            test transform and likelihood with this number of
            random points for errors first. Useful to catch bugs.
        """
        self.paramnames = param_names
        x_dim = len(self.paramnames)

        self.sampler = 'reactive-importance'
        self.x_dim = x_dim

        self.use_mpi = False
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.mpi_size = self.comm.Get_size()
            self.mpi_rank = self.comm.Get_rank()
            if self.mpi_size > 1:
                self.use_mpi = True
                self._setup_distributed_seeds()
        except Exception:
            self.mpi_size = 1
            self.mpi_rank = 0

        self.log = self.mpi_rank == 0

        if self.log:
            self.logger = create_logger('snowline')

        self.ncall = 0
        self._set_likelihood_function(transform, loglike, num_test_samples)

    def _setup_distributed_seeds(self):
        if not self.use_mpi:
            return
        seed = 0
        if self.mpi_rank == 0:
            seed = np.random.randint(0, 1000000)

        seed = self.comm.bcast(seed, root=0)
        if self.mpi_rank > 0:
            # from http://arxiv.org/abs/1005.4117
            seed = int(abs(((seed * 181) * ((self.mpi_rank - 83) * 359)) % 104729))
            # print('setting seed:', self.mpi_rank, seed)
            np.random.seed(seed)

    def _set_likelihood_function(self, transform, loglike, num_test_samples, make_safe=False):
        """Store the transform and log-likelihood functions.

        Tests with `num_test_samples` whether they work and give the correct output.

        if make_safe is set, make functions safer by accepting misformed
        return shapes and non-finite likelihood values.
        """
        # do some checks on the likelihood function
        # this makes debugging easier by failing early with meaningful errors

        for i in range(num_test_samples):
            # test with num_test_samples random points
            u = np.random.uniform(size=self.x_dim)
            p = transform(u) if transform is not None else u
            assert np.shape(p) == (self.x_dim,), ("Error in transform function: returned shape is %s, expected %s" % (p.shape, self.x_dim))
            logl = loglike(p)
            assert np.logical_and(u > 0, u < 1).all(), ("Error in transform function: u was modified!")
            assert np.shape(logl) == (), ("Error in loglikelihood function: returned shape is %s, expected %s" % (np.shape(logl), self.x_dim))
            assert np.isfinite(logl), ("Error in loglikelihood function: returned non-finite number: %s for input u=%s p=%s" % (logl, u, p))
        
        self.ncall += num_test_samples
        self.loglike = loglike

        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def run(self,
            num_global_samples=10000,
            num_gauss_samples=1000,
            max_ncalls=100000,
            min_ess=400,
            max_improvement_loops=4,
            heavytail_laplaceapprox=True,
            verbose=True):
        """Sample at least *min_ess* effective samples have been drawn.

        The steps are:

        1. Draw *num_global_samples* from prior. The highest likelihood point is chosen.
        2. Optimize to find maximum likelihood point.
        3. Estimate local covariance with finite differences.
        4. Importance sample from Laplace approximation (with *num_gauss_samples*).
        5. Construct Gaussian mixture model from samples
        6. Simplify Gaussian mixture model with Variational Bayes
        7. Importance sample from mixture model

        Steps 5-7 are repeated (*max_improvement_loops* times).
        Steps 2-3 are performed with MINUIT, Steps 3-6
        are performed with pypmc.

        Parameters
        ----------
        min_ess: float
            Number of effective samples to draw
        max_ncalls: int
            Maximum number of likelihood function evaluations
        max_improvement_loops: int
            Number of times the proposal should be improved

        num_gauss_samples: int
            Number of samples to draw from initial Gaussian likelihood approximation before
            improving the approximation.
        num_global_samples: int
            Number of samples to draw from the prior to
        heavytail_laplaceapprox: bool
            If False, use laplace approximation as initial gaussian proposal
            If True, use a gaussian mixture, including the laplace approximation
            but also wider gaussians.
        verbose: bool
            whether to print summary information to stdout
        """
        self.laplace_approximate(
            num_global_samples=num_global_samples,
            verbose=verbose)

        results = None
        for results in self.run_iter(
            num_gauss_samples=num_gauss_samples,
            max_ncalls=max_ncalls,
            min_ess=min_ess,
            max_improvement_loops=max_improvement_loops,
            heavytail_laplaceapprox=heavytail_laplaceapprox,
            verbose=verbose,
        ):
            pass
        if verbose and max_improvement_loops > 0:
            self.print_results()
        return results

    def _collect_samples(self, sampler, mixes=None, all=False):
        if self.mpi_size > 1:
            if self.mpi_rank == 0:
                if all:
                    samples = np.vstack([history_item[:] for history_item in sampler.samples_list])
                    weights = np.vstack([combine_weights(
                        [samples[:] for samples in sampler.samples_list[i]],
                        [np.where(weights[:][:, 0] > 0, weights[:][:, 0], 0) for weights in sampler.weights_list[i]],
                        mixes)[:][:, 0] for i in range(self.mpi_size)])
                else:
                    samples = np.vstack([history_item[-1] for history_item in sampler.samples_list])
                    weights = np.vstack([history_item[-1] for history_item in sampler.weights_list])[:, 0]
            else:
                samples = None
                weights = None
            samples = self.comm.bcast(samples)
            weights = self.comm.bcast(weights).flatten()
        else:
            if all:
                weights = combine_weights(
                    [samples[:] for samples in sampler.samples],
                    [np.where(weights[:][:, 0] > 0, weights[:][:, 0], 0) for weights in sampler.weights],
                    mixes)[:][:, 0]
                samples = sampler.samples[:]
            else:
                samples = sampler.samples[-1]
                weights = sampler.weights[-1].flatten()
        assert len(samples) == len(weights), (samples.shape, weights.shape)
        return samples, np.where(weights > 0, weights, 0)

    def run_iter(
            self,
            num_gauss_samples=1000,
            max_ncalls=100000,
            min_ess=400,
            max_improvement_loops=4,
            heavytail_laplaceapprox=True,
            verbose=True,
    ):
        """
        Iterative version of run(). See documentation there.
        Returns current samples on each iteration.
        """
        paramnames = self.paramnames
        loglike = self.loglike
        transform = self.transform

        ndim = len(paramnames)
        optu, cov, invcov = self.optu, self.cov, self.invcov
        # for numerical stability, use 1e260, so that we can go down be 1e-100, 
        # but up by 1e600
        self.Loffset = self.optL #+ 600

        # first iteration: create a single gaussian and importance-sample
        if self.log:
            self.logger.info("Initiating gaussian importance sampler")

        def log_target(u):
            """ log-posterior to sample from """
            if (u > 1).any() or (u < 0).any():
                return -np.inf
            p = transform(u)
            L = loglike(p)
            return L - self.Loffset

        if not heavytail_laplaceapprox:
            initial_proposal = Gauss(optu, cov)
        else:
            # make a few gaussians, in case the fit errors were too narrow
            means, covs, weights = _make_initial_proposal(optu, cov)
            initial_proposal = create_gaussian_mixture(means, covs, weights)

        mixes = [initial_proposal]
        
        N = num_gauss_samples
        Nhere = N // self.mpi_size
        if self.mpi_size > 1:
            SequentialIS = ImportanceSampler
            from pypmc.tools.parallel_sampler import MPISampler
            sampler = MPISampler(
                SequentialIS, target=log_target,
                proposal=initial_proposal, prealloc=Nhere)
        else:
            sampler = ImportanceSampler(
                target=log_target, proposal=initial_proposal, prealloc=Nhere)

        if self.log:
            self.logger.info("    sampling %d ..." % N)
        np.seterr(over="warn")
        sampler.run(Nhere)
        self.ncall += Nhere * self.mpi_size

        samples, weights = self._collect_samples(sampler)
        assert weights.sum() > 0, 'All samples have weight zero.'

        vbmix = None
        for it in range(max_improvement_loops):
            ess_fraction = ess(weights)
            if self.log:
                self.logger.info("    sampling efficiency: %.3f%%" % (ess_fraction * 100))

            if it % 3 == 0:
                if self.log:
                    self.logger.info("Optimizing proposal (from scratch) ...")
                mix = _make_proposal(
                    samples, weights, optu, cov, invcov)
                vb = GaussianInference(
                    samples, weights=weights,
                    initial_guess=mix, W0=np.eye(ndim) * 1e10)
                vb_prune = 0.5 * len(vb.data) / vb.K
            else:
                if self.log:
                    self.logger.info("Optimizing proposal (from previous) ...")
                prior_for_proposal_update = vb.posterior2prior()
                prior_for_proposal_update.pop('alpha0')
                vb = GaussianInference(samples, initial_guess=vbmix,
                                       weights=weights,
                                       **prior_for_proposal_update)

            if self.log:
                self.logger.info('    running variational Bayes ...')
            vb.run(1000, rel_tol=1e-8, abs_tol=1e-5, prune=vb_prune, verbose=False)
            vbmix = vb.make_mixture()
            if self.log:
                self.logger.info('    reduced from %d to %d components' % (
                    len(mix.components), len(vbmix.components)))

            sampler.proposal = vbmix

            if self.log:
                self.logger.info("Importance sampling %d ..." % N)
            sampler.run(N // self.mpi_size)
            self.ncall += (N // self.mpi_size) * self.mpi_size
            mixes.append(vbmix)

            samples, weights = self._collect_samples(sampler)
            ess_fraction = ess(weights)
            if self.log:
                self.logger.debug("    sampling efficiency: %.3f%%" % (ess_fraction * 100))
                self.logger.debug("    obtained %.0f new effective samples" % (ess_fraction * len(weights)))

            samples, weights = self._collect_samples(sampler, all=True, mixes=mixes)
            ess_fraction = ess(weights)
            Ndone = ess_fraction * len(weights)

            result = self._update_results(samples, weights)
            if Ndone >= min_ess:
                if self.log:
                    self.logger.info("Status: Have %d total effective samples, done." % Ndone)
                yield result
                break
            elif self.ncall > max_ncalls:
                if self.log:
                    self.logger.info("Status: Have %d total effective samples, reached max number of calls." % Ndone)
                yield result
                break
            else:
                N = int(1.4 * min(max_ncalls - self.ncall, N))
                if self.log:
                    self.logger.info("Status: Have %d total effective samples, sampling %d next." % (Ndone, N))
                yield result

    def init_globally(self, num_global_samples=10000):
        """ Sample *num_global_samples* points from the prior 
        and store the best point. """

        ndim, loglike, transform = self.x_dim, self.loglike, self.transform

        if self.log:
            self.logger.debug("global sampling for starting point ...")

        if self.use_mpi:
            if self.mpi_rank == 0:
                active_u = np.random.uniform(size=(num_global_samples, ndim))
            else:
                active_u = np.empty((num_global_samples, ndim), dtype=np.float64)
            u = self.comm.bcast(active_u, root=0)

            if self.mpi_rank == 0:
                chunks = [[] for _ in range(self.mpi_size)]
                for i, chunk in enumerate(u):
                    chunks[i % self.mpi_size].append(chunk)
            else:
                chunks = None
            data = self.comm.scatter(chunks, root=0)
            active_p = [transform(ui) for ui in data]
            active_L = [float(loglike(pi)) for pi in active_p]

            recv_active_p = self.comm.gather(active_p, root=0)
            recv_active_p = self.comm.bcast(recv_active_p, root=0)
            p = np.concatenate(recv_active_p, axis=0)

            recv_active_L = self.comm.gather(active_L, root=0)
            recv_active_L = self.comm.bcast(recv_active_L, root=0)
            L = np.concatenate(recv_active_L, axis=0)
        else:
            u = np.random.uniform(size=(num_global_samples, ndim))
            p = [transform(ui) for ui in u]
            L = [float(loglike(pi)) for pi in p]

        i = np.argmax(L)

        self.cov = np.eye(ndim) * 0.04
        self.invcov = np.linalg.inv(self.cov)
        self.optu = u[i]
        self.optp = p[i]
        self.optL = L[i]

        self.ncall += num_global_samples

    def laplace_approximate(self, num_global_samples=400, verbose=True):
        """ Find maximum and derive a Laplace approximation there.
        
        Parameters
        ----------
        num_global_samples: int
            Number of samples to draw from the prior to find a good
            starting point (see `init_globally`).
        verbose: bool
            If true, print out maximum likelihood value and point

         """
        
        if not hasattr(self, 'optu'):
            self.init_globally(num_global_samples=num_global_samples)

        # starting point is:
        startu = np.copy(self.optu)
        ndim = len(startu)

        # this part is not parallelised.
        if self.mpi_rank == 0:
            
            # choose a jump distance that does not go over the space border
            # because Minuit does not support that.
            deltau = 0.9999 * np.min([np.abs(startu - 1), np.abs(startu)], axis=0)
            deltau[deltau > 0.04] = 0.04
            assert deltau.shape == startu.shape

            def negloglike(u):
                """ negative log-likelihood to minimize """
                p = self.transform(u)
                return -self.loglike(p)

            if self.log:
                self.logger.info("    starting optimization from: %s", startu)
                self.logger.info("    error: %s", deltau)
            if hasattr(Minuit, 'from_array_func'):
                m = Minuit.from_array_func(
                    negloglike, startu, errordef=0.5,
                    error=deltau, limit=[(0, 1)] * ndim)
            else:
                m = Minuit(negloglike, startu)
                m.errordef = Minuit.LIKELIHOOD
                m.errors = deltau
                m.limits = np.array([(0, 1)] * ndim)
            m.migrad()

            if hasattr(m, 'fval'):
                optL = -m.fval
            else:
                optL = -m.get_fmin().val
            if verbose:
                print("Maximum likelihood: L = %.1f at:" % optL)
            optu = [max(1e-10, min(1 - 1e-10, m.values[i])) for i in range(ndim)]
            optp = np.asarray(self.transform(np.asarray(optu)))
            umax = [max(1e-6, min(1 - 1e-6, m.values[i] + m.errors[i])) for i in range(ndim)]
            umin = [max(1e-6, min(1 - 1e-6, m.values[i] - m.errors[i])) for i in range(ndim)]
            pmax = np.asarray(self.transform(np.asarray(umax)))
            pmin = np.asarray(self.transform(np.asarray(umin)))
            perr = (pmax - pmin) / 2
            self.logger.info("    optimization finished at L=%.1f: %s" % (optL, optp))

            for name, med, sigma in zip(self.paramnames, optp, perr):
                if sigma > 0:
                    i = max(0, int(-np.floor(np.log10(sigma))) + 1)
                else:
                    i = 3
                fmt = '%%.%df' % i
                fmts = '\t'.join(['    %-20s' + fmt + " +- " + fmt])
                if verbose:
                    print(fmts % (name, med, sigma))

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    m.hesse()
                    hesse_failed = getattr(m, 'hesse_failed', False)
                except:
                    hesse_failed = True
                if not hesse_failed:
                    hesse_failed = any((issubclass(warning.category, HesseFailedWarning) for warning in w))
                if not hesse_failed:
                    hesse_failed = not getattr(m, 'has_covariance', True)
                # check if full rank matrix:
                if not hesse_failed:
                    if hasattr(m, 'np_matrix'):
                        cov = m.np_matrix()
                    else:
                        cov = np.asarray(m.covariance)
                    if cov.shape != (ndim, ndim):
                        self.logger.debug("    hesse failed, not full rank")
                        del cov
                        hesse_failed = True
                else:
                    self.logger.debug("    hesse failed")

            if not hesse_failed:
                self.logger.info("    using correlated errors ...")
                invcov = np.linalg.inv(cov)
            
            if hesse_failed:
                self.logger.info("    using uncorrelated errors ...")
                cov = np.diag(np.clip(perr, a_min=1e-10, a_max=1)**2)
                invcov = np.linalg.inv(cov)
            assert cov.shape == (ndim, ndim), (cov.shape, ndim)
            assert invcov.shape == (ndim, ndim), (invcov.shape, ndim)
            
            if hasattr(m, 'nfcn'):
                self.ncall += m.nfcn
            elif hasattr(m, 'ncalls_total'):
                self.ncall += m.ncalls_total
            else:
                self.ncall += m.ncalls
        else:
            cov = np.empty((ndim, ndim))
            invcov = np.empty((ndim, ndim))
            optu = np.empty(ndim)
            optp = np.empty(ndim)
            optL = np.empty(1)

        if self.use_mpi:
            # inform other processes about the results.
            cov = self.comm.bcast(cov)
            invcov = self.comm.bcast(invcov)
            optu = self.comm.bcast(optu)
            optp = self.comm.bcast(optp)
            optL = self.comm.bcast(optL)
            self.ncall += self.comm.bcast(self.ncall)

        self.invcov, self.cov = invcov, cov
        self.optu, self.optp, self.optL = optu, optp, optL
        
        return self._update_results_laplace()

    def _update_results_laplace(self, num_draws=100000):
        ndim = len(self.optu)
        # draw samples
        samples_u_laplace = np.random.multivariate_normal(self.optu, self.cov, size=num_draws)
        within_cube = np.logical_and(samples_u_laplace > 0, samples_u_laplace < 1).all(axis=1)
        samples_u_within_cube = samples_u_laplace[within_cube,:]
        samples_u = samples_u_within_cube[:10000]
        eqsamples = np.asarray([self.transform(u) for u in samples_u])
        if len(eqsamples) > 1:
            posterior=dict(
                mean=eqsamples.mean(axis=0).tolist(),
                stdev=eqsamples.std(axis=0).tolist(),
                median=np.percentile(eqsamples, 50, axis=0).tolist(),
                errlo=np.percentile(eqsamples, 15.8655, axis=0).tolist(),
                errup=np.percentile(eqsamples, 84.1345, axis=0).tolist(),
            )
        else:
            # if all samples are outside the cube, just return MLE
            # because we cannot estimate the transformed covariance
            posterior = dict(
                mean=self.optp,
                stdev=np.zeros_like(self.optp),
                median=self.optp,
                errlo=self.optp,
                errup=self.optp,
            )
        # estimate ln(Z) using multivariate normal formula
        sign, logdet = np.linalg.slogdet(self.cov)
        logvol = 0.5 * (np.log(2 * np.pi) * ndim + logdet)
        # correct for border by fraction of samples drawn outside cube
        border_correction = np.log((within_cube.sum() + 0.1) / num_draws)
        logZ = self.optL + logvol + border_correction
        
        # we do not have an error estimate available, nor ESS
        self.results = dict(
            z=np.exp(logZ),
            zerr=0.0,
            logz=logZ,
            logzerr=0.0,
            ess=0.0,
            paramnames=self.paramnames,
            ncall=int(self.ncall),
            samples=eqsamples,
            posterior=posterior
        )
        return self.results

    def _update_results(self, samples, weights):
        if self.log:
            self.logger.info('Likelihood function evaluations: %d', self.ncall)

        integral_estimator = weights.sum() / len(weights)
        integral_uncertainty_estimator = np.sqrt((weights**2).sum() / len(weights) - integral_estimator**2) / np.sqrt(len(weights) - 1)

        logZ = np.log(integral_estimator)
        logZerr = np.log(integral_estimator + integral_uncertainty_estimator) - logZ
        ess_fraction = ess(weights)

        # get a decent accuracy based on the weights, and not too few samples
        Nsamples = int(max(400, ess_fraction * len(weights) * 40))
        eqsamples_u = resample_equal(samples, weights / weights.sum(), N=Nsamples)
        eqsamples = np.asarray([self.transform(u) for u in eqsamples_u])

        results = dict(
            z=integral_estimator * np.exp(self.Loffset),
            zerr=integral_uncertainty_estimator * np.exp(self.Loffset),
            logz=logZ + self.Loffset,
            logzerr=logZerr,
            ess=ess_fraction,
            paramnames=self.paramnames,
            ncall=int(self.ncall),
            posterior=dict(
                mean=eqsamples.mean(axis=0).tolist(),
                stdev=eqsamples.std(axis=0).tolist(),
                median=np.percentile(eqsamples, 50, axis=0).tolist(),
                errlo=np.percentile(eqsamples, 15.8655, axis=0).tolist(),
                errup=np.percentile(eqsamples, 84.1345, axis=0).tolist(),
            ),
            samples=eqsamples,
        )
        self.results = results
        return results

    def print_results(self):
        """Give summary of marginal likelihood and parameters."""
        if self.log:
            print()
            print('logZ = %(logz).3f +- %(logzerr).3f' % self.results)
            print()
            for i, p in enumerate(self.paramnames):
                v = self.results['samples'][:, i]
                sigma = v.std()
                med = v.mean()
                if sigma == 0:
                    i = 3
                else:
                    i = max(0, int(-np.floor(np.log10(sigma))) + 1)
                fmt = '%%.%df' % i
                fmts = '\t'.join(['    %-20s' + fmt + " +- " + fmt])
                print(fmts % (p, med, sigma))

    def plot(self, **kwargs):
        if self.log:
            import corner
            corner.corner(
                self.results['samples'],
                labels=self.results['paramnames'],
                show_titles=True)
