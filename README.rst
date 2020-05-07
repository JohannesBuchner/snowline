=========
snowline
=========

Fit and compare models very quickly. MCMC-free.

.. image:: https://img.shields.io/pypi/v/snowline.svg
        :target: https://pypi.python.org/pypi/snowline

.. image:: https://api.travis-ci.org/JohannesBuchner/snowline.svg?branch=master&status=started
        :target: https://travis-ci.org/github/JohannesBuchner/snowline

.. image:: https://img.shields.io/badge/docs-published-ok.svg
        :target: https://johannesbuchner.github.io/snowline/
        :alt: Documentation Status

About
-----

Posterior distributions and corner plots without MCMC?
No dealing with convergence criteria?

Yes!

Tailored for low-dimensional (d<10) problems with a single mode,
this package automatically finds the best fit and uses the local covariance matrix
as a Laplace Approximation. Then Importance Sampling and Variational Bayes come 
in to improve from a single-gaussian approximation to more complex shapes.
This allows sampling efficiently in some problems, and provides a estimate
for the marginal likelihood.

This package is built on top the excellent (i)minuit and pypmc packages.

You can help by testing snowline and reporting issues. Code contributions are welcome.
See the `Contributing page <https://johannesbuchner.github.io/snowline/contributing.html>`_.

Features
---------

* Pythonic. pip installable.
* Easy to program for: Sanity checks with meaningful errors
* Fast
* MPI support

Usage
^^^^^

Read the full documentation at:

https://johannesbuchner.github.io/snowline/


Licence
^^^^^^^

GPLv3 (see LICENCE file). If you require another license, please contact me.

Icon made by `Vecteezy <https://www.vecteezy.com/free-vector/hill>`_.
