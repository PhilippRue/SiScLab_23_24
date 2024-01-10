
print('start benchmark script')

import numpy as np
import subprocess

print('run benchmark')

times = []
for N in [100]:
  for M in [1000]:
    for O in [1, 2, 4, 8, 16]:
      for NO in [16, 128, 256]:
        job = f'OMP_NUM_THREADS={O} ./test.exe -n {N} -m {M} -o {NO}| grep "timings"'
        print(job)
        t = subprocess.check_output(job, shell=True, text=True, encoding='utf-8')
        times += [[N, M, O, NO] + [t.split(':')[1].split()[0]]]
        print('done', N, M, O, NO)
times = np.array(times, dtype=float)

print('save timings to file')
np.savetxt("times_gpu.csv", times, delimiter="\t")

print('end benchmark script')
