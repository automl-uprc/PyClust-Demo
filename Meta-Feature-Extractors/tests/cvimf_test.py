from clustml.landmark import CVIMF
from sklearn.datasets import make_blobs
from clustml import MFExtractor

X, y = make_blobs()

cvimf = CVIMF()
cvimf.calculate_cvi(X,"tau", algorithm="meanshift")
cvimf.calculate_cvi(X,"tau", algorithm="dbscan")
from sklearn.cluster import DBSCAN
db = DBSCAN()
db.fit_predict(X)

mfe = MFExtractor(df=X)
mfe.search_mf(category="landmark", search_type="names")
mfe.calculate_mf(category="similarity-vector")




cvimf.meanshift_labels
dir(cvimf)
cvimf.meanshift_labels

# landmark
from clustml.landmark import CVIMF
from sklearn.datasets import make_blobs
from clustml import MFExtractor

X, y = make_blobs()
mfe = MFExtractor(X)
mfe.search_mf(category="landmark")
mfe.calculate_mf(category="landmark")
