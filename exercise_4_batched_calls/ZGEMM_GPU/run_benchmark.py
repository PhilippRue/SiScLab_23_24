import numpy as np
import subprocess


times = []
for N in [16, 32, 64, 128]:
  for M in [100, 1000, 5000]:
    job = f'./test.exe -n {N} -m {M} | grep "timings batch strided"'
    t = subprocess.check_output(job, shell=True)
    times += [[N, M] + t.split()[-2:]]
    print('done', N, M)
times = np.array(times, dtype=float)

np.savetxt("times.csv", times, delimiter="\t")


