import numpy as np
import sys
m = np.load(sys.argv[1], allow_pickle=False)
for k in m.files: print(k, m[k].shape, m[k].dtype)