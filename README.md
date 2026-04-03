# Progetto High Performance Computing 2025/2026
**Autore:** Simone Brunelli
**Matricola:** 0001021672
**Corso:** High Performance Computing — Ingegneria e Scienze Informatiche
**Università di Bologna, campus di Cesena, AA 2025/2026**

---

## Contenuto dell'archivio

```
src/
├── omp-k-means.c           # versione parallela OpenMP
├── cuda-k-means.cu         # versione parallela CUDA
├── inputgen.c              # generatore di input casuali
├── hpc.h                   # header per la misurazione dei tempi
└── Makefile                # Makefile per compilazione e test
```

---

## Dipendenze

- `gcc` con supporto OpenMP (`-fopenmp`)
- `nvcc` — NVIDIA CUDA Toolkit (versione ≥ 11.0)

Su Ubuntu/Debian:
```bash
sudo apt install gcc
# CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
```

---

## Compilazione

### Con Makefile (consigliato)

```bash
make all
```

### Compilazione manuale

```bash
# OpenMP
gcc -std=c99 -Wall -Wpedantic -fopenmp omp-k-means.c -o omp-k-means

# CUDA
nvcc -O2 -std=c++11 cuda-k-means.cu -o cuda-k-means

# Generatore di input
gcc -std=c99 -Wall -Wpedantic inputgen.c -o inputgen
```

---

## Esecuzione

### Versione OpenMP
```bash
OMP_NUM_THREADS=4 ./omp-k-means K input_file output_file

# Esempio con 4 thread
OMP_NUM_THREADS=4 ./omp-k-means 5 demo.in output.txt
```

### Versione CUDA
```bash
./cuda-k-means K input_file output_file

# Esempio
./cuda-k-means 5 demo.in output.txt
```

### Generatore di input
```bash
./inputgen n_points n_dims n_clusters > output.in

# Esempio: 10000 punti, 3 dimensioni, 8 cluster
./inputgen 10000 3 8 > custom.in
```

---

## Target Makefile

| Target | Descrizione |
|--------|-------------|
| `make all` | Compila tutti i sorgenti disponibili |
| `make test` | Esegue tutte le versioni su `demo.in` e confronta i risultati |
| `make test-file FILE=input.in` | Esegue tutte le versioni su un file a scelta |
| `make bench` | Benchmark completo su tutti i dataset con tabella riepilogativa |
| `make clean` | Rimuove eseguibili e file temporanei |

### Parametri configurabili

```bash
make bench K=10 NP=8               # K cluster, NP thread
make test-file FILE=avila.in K=5 NP=4
```

---

## Note

- La versione CUDA non ha vincoli fissi su K e D: i centroidi vengono
  caricati in shared memory se K×D×4 ≤ 48 KB, altrimenti si usa la
  global memory con cache L2.
- La versione OpenMP usa `OMP_NUM_THREADS` per controllare il numero
  di thread; se non impostata, usa tutti i core disponibili.
- Le funzioni `init_centroids()`, `read_input()` e `save_results()`
  non sono parallelizzate come indicato nelle specifiche del progetto.
- Entrambe le versioni usano lo stesso seed (srand(123)) e gli stessi
  parametri di convergenza (TOL=1e-5, MAXITER=100), garantendo
  risultati riproducibili e confrontabili.
