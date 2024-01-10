
print('start benchmark script')

import numpy as np
import subprocess

print('run benchmark')

times = []
for N in [64]:
  for M in [1000]:
    for O in [1, 2, 4, 8]:
      for NO in [8, 16, 32, 64, 128, 256]:
        job = f'OMP_NUM_THREADS={O} ./test.exe -n {N} -m {M} -o {NO}| grep "timings"'
        print(job)
        t = subprocess.check_output(job, shell=True, text=True, encoding='utf-8')
        times += [[N, M, O, NO] + t.split(':')[1].split()] # [t.split(':')[1].split()[0]]]
        print('done', N, M, O, NO, flush=True)
times = np.array(times, dtype=float)

print('save timings to file')
np.savetxt("times_gpu_parallel.csv", times, delimiter="\t")

print('end benchmark script')
