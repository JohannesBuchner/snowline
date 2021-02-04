.. _install:
.. highlight:: shell

============
Installation
============


Stable release
--------------

To install snowline, run this command in your terminal:

.. code-block:: console

    $ pip install snowline

This is the preferred method to install snowline, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


If you get an error installing pypmc, try installing 
from the `pypmc source repository <https://github.com/fredRos/pypmc/>`_:

.. code-block:: console

    $ pip install cython
    $ pip install git+git://github.com/fredRos/pypmc
    $ pip install snowline


From sources
------------

The sources for snowline can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/JohannesBuchner/snowline

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/JohannesBuchner/snowline/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/JohannesBuchner/snowline
.. _tarball: https://github.com/JohannesBuchner/snowline/tarball/master
