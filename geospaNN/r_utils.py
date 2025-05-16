"""
functions for running BRISC in R

Imports for rpy2 should not be top level. If the R home is not set up
on import it will fail, particularly seems to be in issue in Windows
(see function setup_r_home).
"""

import os
from typing import Optional
from pathlib import Path

import numpy as np
from packaging.version import Version


def setup_r_home(r_path: Optional[str] = None, r_version: Optional[str] = None):
    """
    set up the R_HOME in os enivron
    """
    #
    if "R_HOME" in os.environ or os.name != "nt":
        return
    #
    r_path = "C:/Program Files/R" if r_path is None else r_path
    #
    if r_version:
        if Path(r_version).exists():
            os.environ["R_HOME"] = str(r_version)
            return
        r_version = (
            f"R-{r_version}" if not r_version.startswith("R-") else r_version
        )
        if Path(r_path, r_version).exists():
            os.environ["R_HOME"] = str(r_version)
            return

        raise ValueError(f"R version {r_version}, could not be found")
    #
    valid_versions = [
        (file_path, Version(file_path.name.replace("R-", "")))
        for file_path in Path(r_path).iterdir()
        if file_path.is_dir() and file_path.name.startswith("R-")
    ]
    #
    if not valid_versions:
        raise ValueError(f"No valid R installation found in: {r_path}")
    # Get the directory with the highest version
    os.environ["R_HOME"] = str(max(valid_versions, key=lambda x: x[1])[0])


def ensure_r_packages_installed(packnames="BRISC"):
    """
    install R packages in the current environment
    """
    # pylint: disable=import-outside-toplevel
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.vectors import StrVector

    # Check if R home
    setup_r_home()
    if "R_HOME" not in os.environ:
        raise ValueError("R_HOME not set in environment")

    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    #
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))


# Ensure R packages are installed at runtime
ensure_r_packages_installed()


# pylint: disable=invalid-name
# want to keep the same name as R (and arg order)
def BRISC_estimation(coords, y, x=None, **kwargs):
    """
    Run BRISC estimation in R

    Again top level imports do not always work if R_HOME is not set.
    """
    # pylint: disable=import-outside-toplevel
    from rpy2.robjects import r, numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    from rpy2.rinterface_lib import openrlib

    brisc = importr("BRISC")
    #
    with localconverter(numpy2ri.converter) + openrlib.rlock:
        if x is None:
            res = brisc.BRISC_estimation(coords, y, **kwargs)
        else:
            res = brisc.BRISC_estimation(coords, y, x, **kwargs)
    #
    r("gc()")
    #
    beta = np.array(res["Beta"])
    theta_hat = np.array(res["Theta"])
    sigma_sq, tau_sq, phi = theta_hat
    theta_hat[1] = phi
    theta_hat[2] = max(tau_sq / sigma_sq, 1e-03)

    return beta, theta_hat
