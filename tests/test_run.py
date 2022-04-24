import numpy as np
from snowline import ReactiveImportanceSampler

def get_funccalls(loglike, sampler):
    if sampler.mpi_size > 1:
        ncalls = sampler.comm.gather(loglike.ncalls, root=0)
        if sampler.mpi_rank == 0:
            print("ncalls on the different MPI ranks:", ncalls)
        return sum(sampler.comm.bcast(ncalls, root=0))
    else:
        return loglike.ncalls

def test_laplace_only():
    np.random.seed(2)

    paramnames = ['Hinz', 'Kunz']
    def loglike(z):
        assert len(z) == len(paramnames)
        a = -0.5 * (((z - 0.5) / 0.01)**2).sum() + -0.5 * ((z[0] - z[1])/0.01)**2
        loglike.ncalls += 1
        return a
    loglike.ncalls = 0

    def transform(x):
        assert len(x) == len(paramnames)
        return 10. * x - 5.
    

    sampler = ReactiveImportanceSampler(paramnames, loglike, transform=transform)
    print('ncall (after init):', sampler.ncall)
    ncalls = get_funccalls(loglike, sampler)
    assert sampler.ncall == ncalls, (sampler.ncall, ncalls)

    sampler.init_globally()
    print('ncall (after global sampling):', sampler.ncall)
    ncalls = get_funccalls(loglike, sampler)
    assert sampler.ncall == ncalls, (sampler.ncall, ncalls)

    results = sampler.laplace_approximate()
    print('ncall (after laplace):', sampler.ncall)
    ncalls2 = get_funccalls(loglike, sampler)
    assert sampler.ncall == ncalls2, (sampler.ncall, ncalls2)

    assert np.isclose(results['logz'], -12.527302073226371, atol=0.01), results['logz']
    assert np.isclose(np.exp(results['logz']), results['z']), (results['logz'], results['z'])
    assert results['paramnames'] == paramnames
    assert np.allclose(results['posterior']['mean'], [0.5, 0.5], atol=0.01)
    assert np.allclose(results['posterior']['median'], [0.5, 0.5], atol=0.01)
    assert (np.asarray(results['posterior']['errlo']) <= np.asarray(results['posterior']['median'])).all()
    assert (np.asarray(results['posterior']['errup']) >= np.asarray(results['posterior']['median'])).all()
    assert len(results['samples']) > 0

    print(results)
    sampler.plot()


def test_run():
    np.random.seed(1)

    paramnames = ['Hinz', 'Kunz']
    def loglike(z):
        assert len(z) == len(paramnames)
        a = -0.5 * (((z - 0.5) / 0.01)**2).sum() + -0.5 * ((z[0] - z[1])/0.01)**2
        loglike.ncalls += 1
        return a
    loglike.ncalls = 0

    def transform(x):
        assert len(x) == len(paramnames)
        return 10. * x - 5.
    

    sampler = ReactiveImportanceSampler(paramnames, loglike, transform=transform)
    r = sampler.run()
    print('ncall (after run):', sampler.ncall)
    assert sampler.ncall == r['ncall'], (sampler.ncall, r['ncall'])
    ncalls = get_funccalls(loglike, sampler)
    assert sampler.ncall == ncalls, (sampler.ncall, ncalls)

    results = r
    assert np.isclose(results['logz'], -12.527302073226371, atol=0.01), results['logz']
    assert np.isclose(np.exp(results['logz']), results['z']), (results['logz'], results['z'])
    assert results['paramnames'] == paramnames
    assert np.allclose(results['posterior']['mean'], [0.5, 0.5], atol=0.01)
    assert np.allclose(results['posterior']['median'], [0.5, 0.5], atol=0.01)
    assert (np.asarray(results['posterior']['errlo']) <= np.asarray(results['posterior']['median'])).all()
    assert (np.asarray(results['posterior']['errup']) >= np.asarray(results['posterior']['median'])).all()
    assert len(results['samples']) > 0
    sampler.plot()


def test_rosen():
    np.random.seed(1)

    paramnames = ['Hinz', 'Kunz', 'Fuchs', 'Gans', 'Hofer']
    def loglike(theta):
        assert len(theta) == len(paramnames)
        a = theta[:-1]
        b = theta[1:]
        return -2 * (100 * (b - a**2)**2 + (1 - a)**2).sum()

    def transform(u):
        assert len(u) == len(paramnames)
        return u * 20 - 10
    
    sampler = ReactiveImportanceSampler(paramnames, loglike, transform=transform)
    sampler.run(min_ess=1000)
    

if __name__ == '__main__':
    test_run()
    test_rosen()
