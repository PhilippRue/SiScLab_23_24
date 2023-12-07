
import numpy as np
import subprocess


times = []
for O in [1, 2, 4, 8, 16, 32]:
  for N in [16, 32, 64, 128]:
    for M in [100, 1000, 5000]:
      job = f'OMP_NUM_THREADS={O} ./test.exe -n {N} -m {M} | grep "timings batch strided"'
      t = subprocess.check_output(job, shell=True)
      times += [[O, N, M] + t.split()[-2:]]
      print('done', O, N, M)
times = np.array(times, dtype=float)

np.savetxt("times.csv", times, delimiter="\t")


