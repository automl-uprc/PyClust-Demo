import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector


def install_r_packages(packnames: list, lib: str) -> None:
    """Installs the appropriate R packages for autoClust"""
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    if lib == None:
        utils.install_packages(StrVector(packnames))
    else:
        utils.install_packages(StrVector(packnames), lib=lib)