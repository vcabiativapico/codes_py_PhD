import numpy as np
import time

A = np.random.randint(-10, high=20,size=(283,151,601))
B = np.random.randint(-10, high=20,size=(283,151,601))
c = np.random.randint(-5,high=5,size=(151,601))

start_time = time.time()
summsumm = []
for k in range(A.shape[0]):
    summ = 0
    for j in range(A.shape[1]):
        for l in range(A.shape[2]):
            summ += A[k, j, l] * B[k, j, l] * c[j, l]
    summsumm.append(summ)
print("Time in seconds, regular sum : {}s".format(time.time()-start_time))
print(summsumm)

start_time = time.time()
summ_vectorisation = np.sum(np.sum(A[:,:,:]*B[:,:,:]*c[np.newaxis,:,:],axis=1),axis=1)

print("Time in seconds, vectorisation sum : {}s".format(time.time()-start_time))
print(summ_vectorisation)