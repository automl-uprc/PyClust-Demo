import numpy as np
import pandas as pd

from clustml._mf_extractor import MFExtractor

df = np.random.rand(5, 5)
mf = MFExtractor(df)
mf.calculate_mf()
mf.meta_features



df = pd.DataFrame(df)
df.loc[:, 0]