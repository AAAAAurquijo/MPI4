# Parallel Word Counting in a Text Corpus with MPI

## 2. Problem Description

Given a list of query words (`dataset/consulta.txt`) and a corpus of text files (`dataset/file_XXXX.txt`), the program must count how many times each query word appears in the whole corpus and print the 10 most frequent ones.

This lab compares three solutions on the same dataset:

1. **sequential baseline** (provided).
2. **MPI version 1** (`mpi1.py`): Implements broadcast of the query words and static file distribution.
3. **MPI version 2** (`mpi2.py`): a second version designed to reduce load imbalance using dynamic work distribution.

All versions must produce the same global result as the sequential baseline.

## 3. Environment and Execution Instructions

**Environment**

- Operating system: Windows 11 
- Container runtime: Docker Desktop version 29.9.2 
- Container image: `augustosalazar/slim-mpi:2` 
- CPU: 12th Gen Intel(R) Core(TM) i5-12500H (2.50 GHz)
- RAM: 16GB 
- Number of processes `p`: 13 (value reported by `Detectar_procesos_mpi`) 

**Repository contents**

| File | Description |
|---|---|
| `generator.py` | Generates the dataset (as provided, unchanged) |
| `baseline_secuencial.py` | Sequential baseline (as provided, unchanged) |
| `run_all.sh` | Provided script (unchanged) |
| `mpi1.py` | MPI version 1: broadcast + static file distribution |
| `mpi2.py` | MPI version 2: dynamic distribution (coordinator–workers) |
| `README.md` | This report |

**Dataset generation** (run once, from the folder that contains the scripts):

```bash
docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 python /app/generator.py
```

### a. Experimental Plan

1. Run the sequential baseline and record `T_seq` and its global output.
```bash
docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py
```
2. Determine `p` with `Detectar_procesos_mpi` (my result: `p = 13`).
```bash
docker run --rm -v"%cd%:/app" augustosalazar/slim-mpi:2 python /app/detectar_procesos_mpi.py
```
3. Run MPI version 1 with `p`, at least 3 runs. Record, for each run: total execution time, local time of every process, load imbalance `LI` and relative spread `RS`.
```bash
docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 mpiexec --allow-run-as-root --use-hwthread-cpus --oversubscribe -n p python /app/mpi1.py
```
4. Compute the mean total time `T_p`, speedup `S_p = T_seq / T_p`, and efficiency `E_p = S_p / p`.
5. Implement MPI version 2 to reduce the imbalance observed in version 1, and repeat steps 3 and 4 with the same `p`.
```bash
docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 mpiexec --allow-run-as-root --use-hwthread-cpus --oversubscribe -n p python /app/mpi2.py
```
6. Compare both versions in total time, speedup, efficiency, and load balance, and analyse the results.

Metrics used (`t_i` is the local counting time of process `i`):

- `LI = max(t_i) / mean(t_i)` (1.0 = perfect balance)
- `RS = (max(t_i) − min(t_i)) / mean(t_i)` (0 = perfect balance)
- `S_p = T_seq / T_p`, `E_p = S_p / p`

### b. Sequential baseline (what it does and how)

`baseline_secuencial.py` reads `consulta.txt` into a set of lowercase words, then walks through every `file_*.txt` in `dataset/`, splits each line with `split()`, lowercases the tokens, and increments a `Counter` for each token that belongs to the query set. At the end it prints the execution time, the number of files, tokens and occurrences, saves `baseline_results.csv`, and shows the top 10. It runs in a single process.

### c. MPI version 1 (what it does and how)

`mpi1.py` follows the steps of the assignment:

1. Rank 0 reads `consulta.txt`.
2. Rank 0 broadcasts the query words to all processes (`comm.bcast`).
3. Rank 0 lists the `file_*.txt` files (sorted, so the assignment is repeatable).
4. **Static distribution:** file `i` of the sorted list is assigned to process `i % p` (`files[i::p]`), and each process receives its list with `comm.scatter`.
5. Each process counts the query words in its own files, using the same counting rule as the baseline, and measures its local time (only the counting loop, without communication).
6. Partial results are collected in rank 0 with `comm.gather`.
7. Rank 0 adds up the partial counters, prints a per-process report (files and local time), `LI`, `RS`, and the top 10, and saves `mpi1_results.csv`.

The total time is measured by rank 0 from a common start (after a `Barrier`) until the global result is built.

### d. Test procedure

1. **Correctness:** for every MPI run, the top 10 and the totals (files, tokens, occurrences) must match the sequential output.
2. **Timing:** run each version at least 3 times with the same `p` and the same dataset, and record the values printed by the program.
3. **Computation:** mean total time, speedup, efficiency, mean `LI` and mean `RS`.

MPI version 2 is run in the same way.

## 4. Experimental Plan Execution

### a. Sequential baseline timing

One run of the baseline on the generated dataset:

| Metric | Value |
|---|---|
| `T_seq` | **62.981996 s** |
| Files processed | 3000 |
| Tokens read | 44,951,458 |
| Occurrences found | 3,631,778 |

Top 10 (reference output):

| Word | Count |
|---|---:|
| a | 785774 |
| para | 392156 |
| sus | 228913 |
| otros | 105530 |
| ante | 99832 |
| unos | 88794 |
| otra | 83901 |
| vosotros | 61617 |
| mios | 58420 |
| tuya | 56635 |

All MPI runs below (version 1 and version 2) printed exactly this top 10 and the same totals.

### b. MPI version 1 timing results (using `p = 13`)

| Run | Total time (s) | LI | RS | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.066088 | 1.050 | 0.135 | 3.92 | 0.302 |
| 2 | 14.285211 | 1.058 | 0.143 | 4.41 | 0.339 |
| 3 | 15.384609 | 1.058 | 0.160 | 4.09 | 0.315 |
| **Mean** | **15.245** | **1.055** | **0.146** | **4.13** | **0.318** |

The mean-based values are `T_p = 15.245 s`, `S_p = 62.981996 / 15.245 = 4.13` and `E_p = 4.13 / 13 = 0.318`.

### c. Load imbalance evidence (per-process local times and LI)

Local counting time per process, MPI version 1 (seconds):

| Rank | Files | Run 1 | Run 2 | Run 3 |
|---:|---:|---:|---:|---:|
| 0 | 231 | 7.183 | 6.270 | 6.134 |
| 1 | 231 | 7.412 | 6.478 | 6.343 |
| 2 | 231 | **6.563** | **5.716** | **5.469** |
| 3 | 231 | 7.228 | 6.320 | 6.117 |
| 4 | 231 | **7.533** | **6.608** | **6.443** |
| 5 | 231 | 7.146 | 6.150 | 6.071 |
| 6 | 231 | 7.126 | 6.175 | 6.063 |
| 7 | 231 | 7.322 | 6.414 | 6.283 |
| 8 | 231 | 7.292 | 6.380 | 6.195 |
| 9 | 231 | 7.190 | 6.296 | 6.131 |
| 10 | 230 | 7.169 | 6.147 | 6.052 |
| 11 | 230 | 6.809 | 5.864 | 5.663 |
| 12 | 230 | 7.277 | 6.387 | 6.219 |
| **LI** | | **1.050** | **1.058** | **1.058** |
| **RS** | | 0.135 | 0.143 | 0.160 |

Observations:

- Every process received 230 or 231 files, so the file count was balanced. The differences in time come from the cost of the files themselves or from the machine, not from the number of files.
- The pattern repeats in the three runs: rank 4 is always the slowest and rank 2 is always the fastest, which points to a systematic cause rather than random noise.
- In every run the slowest process is 5–6 % above the average (`LI ≈ 1.05`), and the gap between the slowest and the fastest process is 0.9–1.0 s (`RS` between 0.135 and 0.160).

### d. MPI version 2: how it corrects the imbalance, and its results (total execution time, speedup, efficiency, LI)

**How it works:** `mpi2.py` replaces the fixed assignment with dynamic distribution in a coordinator–workers scheme:

1. Rank 0 reads `consulta.txt` and broadcasts the query words (same as version 1).
2. Rank 0 becomes the coordinator, that means it sorts the files from largest to smallest (by size) and hands them out one at a time.
3. Each worker sends a request to rank 0, receives a file path, counts it, and then asks again. When there are no files left, the coordinator answers `None` and the worker stops.
4. At the end, partial results are collected with `comm.gather` and rank 0 builds the global result.

A process that finishes early simply asks for more work, so no worker waits idle while files remain, and starting with the largest files avoids leaving a big file for the end. The cost of this design is that rank 0 does not count, so with `p = 13` there are 12 workers. That is why `LI` and `RS` for this version are computed over the 12 workers.

**Results (using `p = 13` and 12 workers):**

| Run | Total time (s) | LI | RS | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 36.291049 | 1.001 | 0.003 | 1.74 | 0.133 |
| 2 | 18.669380 | 1.001 | 0.002 | 3.37 | 0.260 |
| 3 | 21.340803 | 1.002 | 0.005 | 2.95 | 0.227 |
| **Mean** | **25.434** | **1.001** | **0.003** | **2.48** | **0.190** |
| Median | 21.341 | – | – | 2.95 | 0.227 |

Per-worker files and local times, MPI version 2:

| Rank | Files (run 1) | Time (run 1) | Files (run 2) | Time (run 2) | Files (run 3) | Time (run 3) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 249 | 17.677 | 252 | 5.530 | 249 | 6.067 |
| 2 | 249 | 17.704 | 249 | 5.533 | 250 | 6.038 |
| 3 | 257 | 17.669 | 250 | 5.525 | 249 | 6.053 |
| 4 | 249 | 17.696 | 250 | 5.531 | 249 | 6.065 |
| 5 | 253 | 17.687 | 250 | 5.527 | 250 | 6.057 |
| 6 | 246 | 17.721 | 251 | 5.531 | 251 | 6.061 |
| 7 | 250 | 17.707 | 251 | 5.524 | 249 | 6.052 |
| 8 | 253 | 17.685 | 248 | 5.522 | 253 | 6.061 |
| 9 | 255 | 17.701 | 249 | 5.524 | 250 | 6.052 |
| 10 | 245 | 17.706 | 250 | 5.523 | 252 | 6.061 |
| 11 | 247 | 17.707 | 251 | 5.529 | 250 | 6.068 |
| 12 | 247 | 17.693 | 249 | 5.531 | 248 | 6.062 |

**Comparison of both versions (`p = 13`):**

| Metric | Version 1 (static) | Version 2 (dynamic) |
|---|---:|---:|
| Mean total time (s) | 15.245 | 25.434 |
| Median total time (s) | 15.385 | 21.341 |
| Best run (s) | 14.285 | 18.669 |
| Speedup (mean) | 4.13 | 2.48 |
| Efficiency (mean) | 0.318 | 0.190 |
| LI (mean) | 1.055 | 1.001 |
| RS (mean) | 0.146 | 0.003 |

* The comparison above uses the mean as the assignment asks, and the median is shown to illustrate how much this single run affects it. The conclusion, even the best run of version 2 (18.67 s) is slower than the slowest run of version 1 (16.07 s).

## 5. Analysis

### a. Parallel performance:

**Did version 1 improve on the sequential baseline? how close was its speedup to the ideal (S_p = p)?** 
R/ Yes, the mean time dropped from 62.98 s to 15.25 s, a speedup of 4.13. However, the ideal speedup with 13 processes is 13, so version 1 reached about 32 % of the ideal (efficiency 0.318).

**What explains the gap?** 
R/ Two observations from the measurements:

1. **A large part of the total time is not the counting loop.** In each run, the total time minus the slowest process's local time was 8.53 s, 7.68 s and 8.94 s (mean 8.38 s), about half of the total. This time covers everything outside the counting loop: listing the files, the broadcast, the scatter, the gather, waiting for rank 0, and building the result. It does not shrink when more processes are added. With the ideal counting time `62.98 / 13 = 4.84 s`, adding this ~8.4 s gives about 13.2 s, which is close to the measured 14.3–16.1 s.
2. **Counting itself was slower per file when running in parallel.** The sum of local times of the 13 processes was 93.3 s, 81.2 s and 79.2 s, more than the 63.0 s of the whole sequential run. That means 26–31 ms per file against 21 ms per file in the baseline. The slowest process needed 6.4–7.5 s instead of the ideal 4.84 s. This is consistent with the 13 processes competing for the same cores, memory, and file access, but this was not measured directly.

### b. Load balance

**What evidence of load imbalance did version 1 show (per-process local times and LI)** Local times ranged from 5.47 s to 7.53 s, with `LI` between 1.050 and 1.058 and `RS` between 0.135 and 0.160. All processes had 230–231 files, and the same ranks (4 slowest, 2 fastest) repeated in every run. The imbalance existed, but it was small.

**Did version 2 reduce it enough to produce a real performance improvement?** It reduced the imbalance almost to zero (`LI` went from 1.055 in version 1 to 1.001 in version 2, `RS` from 0.146 to 0.003, workers processed between 245 and 257 files, adapting to their own speed), but this didn't produce a performance improvement. The mean total time worsened from 15.25 s to 25.43 s (median: 15.39 s to 21.34 s). Reasons supported by the data:

- Version 1 was already nearly balanced, so the maximum benefit of perfect balance was about 5 %.
- Version 2 gives up a worker. The coordinator does not count, so counting capacity falls from 13 to 12 processes, a loss of 7.7 %, which is larger than the maximum 5 % gain.
- Extra time outside the counting loop: In version 2 the total minus the slowest local time was 18.57 s, 13.14 s and 15.27 s, compared with 7.7–8.9 s in version 1. Possible causes, which were not measured are: reading the size of all 3000 files in the coordinator to sort them (on a bind-mounted folder), 3000 request/response message pairs through a single coordinator, and a coordinator process that waits for messages while competing for CPU in an oversubscribed run.
- One anomalous run (run 1) inflated the mean.

In short, `LI` was lowered but the cost of the technique was higher than the benefit it could bring on this dataset. A good `LI` is necessary but not sufficient for a good total time.

### c. Limitations

- Only 3 runs per version. This is the minimum requested. The anomalous run 1 of version 2 (36.29 s) shows that the variability is large and that the mean of version 2 is not reliable. Conclusions about the size of the difference between versions are weak, the direction (version 1 faster) is robust because it holds for every individual run.
- The experiments ran in Docker Desktop on Windows, with the dataset in a folder shared with the host, so file access goes through a virtualization layer and may vary between runs.
- `p` came from `Detectar_procesos_mpi` and the programs run with `--oversubscribe`, so the number of physical cores available to the container is not known precisely, so contention between processes may affect the local times.
- `LI` and `RS` are not computed over identical sets. Version 1 uses 13 processes and version 2 uses the 12 workers, and local times exclude communication and waiting.

## 6. Conclusions

For this dataset I would recommend the static distribution of MPI version 1. With 13 processes it reduced the time from 62.98 s to a mean of 15.25 s (speedup 4.13, efficiency 0.32) with an already small imbalance (`LI = 1.055`), while the dynamic version 2 achieved a nearly perfect balance (`LI = 1.001`) but was slower (mean 25.43 s, median 21.34 s; speedup 2.48, efficiency 0.19) because the potential gain from balancing was at most ~5 %, it loses one counting process to the coordinator (13 → 12), and it adds extra time outside the counting loop. Since both versions are limited mainly by time outside the counting loop and by contention between processes (about half of the total time in version 1), the next improvement should target those factors rather than the load balance, dynamic distribution would be worth reconsidering for datasets with very uneven file sizes.