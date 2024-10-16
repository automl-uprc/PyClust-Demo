import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.vectors import IntVector
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpackages

def install_r_packages(packnames: list, lib: str) -> None:
    """ Installs the appropriate R packages for autoClust """
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    if lib is None:
        utils.install_packages(StrVector(packnames))
    else:
        utils.install_packages(StrVector(packnames), lib=lib)


def compute_internal_cvi(X: object, labels: object, cvi_lib_loc=None):
    rpy2.robjects.numpy2ri.activate()
    clusterCrit = ""
    fpc = ""

    # Import R libraries. The location of the r packages can be set manually, thus the parameter cvi_lib_loc
    # is to declare such path to packages.
    try:
        if cvi_lib_loc is not None:
            fpc = importr('fpc', lib_loc=cvi_lib_loc)
            clusterCrit = importr("clusterCrit", lib_loc=cvi_lib_loc)
        else:
            fpc = importr('fpc')
            clusterCrit = importr("clusterCrit")

    except Exception as e:
        install_r_packages(("fpc", "clusterCrit"), lib=cvi_lib_loc)

    base = importr("base")
    r_labels = base.match(labels, base.unique(labels))
    r_data = np.array(X)

    nr, nc = X.shape
    Br = ro.r.matrix(r_data, nrow=nr, ncol=nc)
    test = clusterCrit.intCriteria(Br, IntVector(r_labels), "all")
    int_cvi = dict(zip(test.names, map(list, list(test))))
    int_cvi = dict(map(lambda x: (x[0], float(x[1][0])), int_cvi.items()))

    cdbw = fpc.cdbw(Br, r_labels)[1][0]
    int_cvi["cdbw"] = cdbw

    # Correction for silhouette score.
    if str(int_cvi["silhouette"]) == "nan":
        int_cvi["silhouette"] = sc(X, labels)

    return int_cvi