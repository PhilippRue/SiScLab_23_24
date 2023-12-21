
import numpy as np
import subprocess


times = []
for N in [16, 32, 64, 128]:
  for M in [100, 1000, 5000, 10000, 20000]:
    job = f'./test.exe -n {N} -m {M} | grep "timings"'
    t = subprocess.check_output(job, shell=True)
    times += [[N, M] + t.split()[-5:-1]]
    print('done', N, M)
times = np.array(times, dtype=float)

np.savetxt("times_gpu.csv", times, delimiter="\t")


