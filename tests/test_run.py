import numpy as np
from snowline import ReactiveImportanceSampler


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
    #sampler.laplace_approximate()
    
    ncalls = loglike.ncalls
    if sampler.mpi_size > 1:
        ncalls = sampler.comm.gather(ncalls, root=0)
        if sampler.mpi_rank == 0:
            print("ncalls on the different MPI ranks:", ncalls)
        ncalls = sum(sampler.comm.bcast(ncalls, root=0))
    assert abs(sampler.ncall - ncalls) == 2 * sampler.mpi_size, (sampler.ncall, ncalls)

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
    #test_run()
    test_rosen()
