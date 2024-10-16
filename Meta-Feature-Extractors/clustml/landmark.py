
from sklearn.cluster import MeanShift, DBSCAN, OPTICS
import sys

sys.path.append(r"C:\Users\giann\OneDrive\Έγγραφα\GitHub\cvi")
import pyclust_eval.cvi
import pyclust_eval


class CVIMF:
    def __init__(self):
        self.algorithms = {"meanshift": MeanShift().fit_predict, "dbscan": DBSCAN().fit_predict,
                            "optics": OPTICS().fit_predict}
        self.X = None
        pass

    def calculate_cvi(self,X, cvi, algorithm):
        """

        :param cvi:
        :type cvi:
        :param algorithm:
        :type algorithm:
        :param y:
        :type y:
        :return:
        :rtype:
        """
        self.X = X
        if hasattr(CVIMF, algorithm + "_labels"):
            pass
        else:
            setattr(CVIMF, algorithm + "_labels", self.algorithms[algorithm](self.X))

        cvi_value = getattr(pyclust_eval.cvi, cvi)(self.X, getattr(CVIMF, algorithm + "_labels"))
        setattr(CVIMF, algorithm + f"_{cvi}", cvi_value)
        return cvi_value
