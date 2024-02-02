import numpy as np
import time
from tqdm import tqdm



A = np.random.randint(-10, high=20,size=(283,151,601))
Ai = np.random.randint(-10, high=20,size=(283,151,601))
B = np.random.randint(-10, high=20,size=(283,151,601))
Bi = np.random.randint(-10, high=20,size=(283,151,601))
c = np.random.randint(-5,high=5,size=(151,601))
ci = np.random.randint(-5,high=5,size=(151,601))

Acomp = np.zeros((283,151,601),dtype='complex')
Bcomp = np.zeros((283,151,601),dtype='complex')
ccomp = np.zeros((151,601),dtype='complex')

for i in tqdm(range(283)):
    for k in range(151):
        for l in range(601):
            Acomp[i,k,l] = complex(A[i,k,l], Ai[i,k,l])
            Bcomp[i,k,l] = complex(B[i,k,l], Bi[i,k,l])
            ccomp[k,l] = complex(c[k,l], ci[k,l])

start_time = time.time()
summsumm = []
for k in range(A.shape[0]):
    summ = 0
    for j in range(A.shape[1]):
        for l in range(A.shape[2]):
            summ += Acomp[k, j, l] * Bcomp[k, j, l] * ccomp[j, l]
    summsumm.append(summ)
print("Time in seconds, regular sum : {}s".format(time.time()-start_time))
print(summsumm)

start_time = time.time()
summ_vectorisation = np.sum(np.sum(Acomp[:,:,:]*Bcomp[:,:,:]*ccomp[np.newaxis,:,:],axis=1),axis=1)

print("Time in seconds, vectorisation sum : {}s".format(time.time()-start_time))
print(summ_vectorisation)