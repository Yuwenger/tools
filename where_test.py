import numpy as np
arr = np.random.randn(1,2,1,1)
c =  arr[0,1,:,:]
print c
print c.shape
print np.where(c>-1)
