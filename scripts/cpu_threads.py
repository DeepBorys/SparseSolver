# SparseSolver — Copyright (C) 2026 Jerzy Slaby (DeepBorys).
# Uses the postflop-solver engine by Wataru Inariba; see NOTICE.
# SPDX-License-Identifier: AGPL-3.0-or-later
# Modified for SparseSolver inference and benchmarks; see NOTICE and LICENSE.

"""Available solver workers, independent of single-threaded BLAS/OpenMP settings."""
import os,subprocess

def available_threads():
 limits=[]
 try:limits.append(len(os.sched_getaffinity(0)))
 except (AttributeError,OSError):pass
 env=os.environ.copy()
 # nproc respects these variables; they deliberately equal1 for NN/BLAS,
 # but must not reduce the number of independent CFR workers.
 env.pop('OMP_NUM_THREADS',None);env.pop('OMP_THREAD_LIMIT',None)
 try:
  n=int(subprocess.check_output(['nproc'],env=env,text=True).strip())
  if n>0:limits.append(n)
 except (OSError,ValueError,subprocess.CalledProcessError):pass
 return max(1,min(limits))if limits else max(1,os.cpu_count()or 1)

def validate_threads(n):
 available=available_threads()
 if n<1 or n>available:raise ValueError(f'Requested {n} workers; available CPUs: {available}. Omit --threads to use all available CPUs.')
