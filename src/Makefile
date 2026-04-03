## Makefile per K-Means: seriale, OpenMP, MPI e CUDA.
##
## Struttura delle cartelle (create automaticamente):
##   movie/          contiene demo.mp4 e i file temporanei (png, txt)
##   results/serial/ contiene l'output della versione seriale
##   results/omp/    contiene l'output della versione OpenMP
##   results/mpi/    contiene l'output della versione MPI
##   results/cuda/   contiene l'output della versione CUDA
##
## Target disponibili:
##
##   all        compila tutto (seriale + OpenMP + MPI + CUDA + inputgen)
##   demo       genera movie/demo.mp4 (versione seriale)
##   test       esegue le versioni disponibili su demo.in e confronta
##   bench      esegue le versioni disponibili su tutti i file di input
##   clean      rimuove eseguibili e file temporanei
##
## Last modified on 2025-12-03 by Moreno Marzolla
## Modified for OpenMP, MPI and CUDA by Simone Brunelli

SHELL=/bin/bash

CC=gcc
MPICC=mpicc
NVCC=nvcc
CFLAGS+=-std=c99 -Wall -Wpedantic
OMPFLAGS=-fopenmp
NVCCFLAGS=-O2 -std=c++11

MPIRUN=mpirun --mca btl_vader_single_copy_mechanism none --mca orte_base_help_aggregate 0 -n

MOVIE_DIR=movie
SERIAL_DIR=results/serial
OMP_DIR=results/omp
MPI_DIR=results/mpi
CUDA_DIR=results/cuda

INPUTS=demo.in iris.in avila.in letter-recognition.in \
       test-N50000-D80.in test-N100000-D30.in test-N150000-D20.in

K=5
NP=4

GREEN=\033[0;32m
RED=\033[0;31m
CYAN=\033[0;36m
BOLD=\033[1m
RESET=\033[0m

.PHONY: all demo test test-file bench clean

all:
	mkdir -p $(MOVIE_DIR) $(SERIAL_DIR) $(OMP_DIR) $(MPI_DIR) $(CUDA_DIR)
	@## Compila solo se i sorgenti esistono
	@[ -f k-means.c ]       && $(CC) $(CFLAGS) k-means.c -o k-means             || echo "k-means.c non trovato, salto"
	@[ -f omp-k-means.c ]   && $(CC) $(CFLAGS) $(OMPFLAGS) omp-k-means.c -o omp-k-means || echo "omp-k-means.c non trovato, salto"
	@## mpi-k-means non consegnato: compilare manualmente se disponibile
	@[ -f cuda-k-means.cu ] && $(NVCC) $(NVCCFLAGS) cuda-k-means.cu -o cuda-k-means || echo "cuda-k-means.cu non trovato, salto"
	@[ -f inputgen.c ]      && $(CC) $(CFLAGS) inputgen.c -o inputgen            || echo "inputgen.c non trovato, salto"

demo: k-means
	mkdir -p $(MOVIE_DIR)
	$(CC) $(CFLAGS) -DMAKE_MOVIE k-means.c -o k-means
	\rm -f $(MOVIE_DIR)/centroids_*.txt $(MOVIE_DIR)/clusters_*.txt $(MOVIE_DIR)/img_*.png
	cd $(MOVIE_DIR) && ../k-means $(K) ../demo.in demo.out && \rm -f demo.out
	cd $(MOVIE_DIR) && ../generate_frames.sh
	ffmpeg -pattern_type glob -stream_loop 5 -y -r 1 -i "$(MOVIE_DIR)/img_*.png" -vcodec mpeg4 -r 1 $(MOVIE_DIR)/demo.mp4

demo.in: inputgen
	./inputgen 20 2 50 > $@

test:
	mkdir -p $(SERIAL_DIR) $(OMP_DIR) $(MPI_DIR) $(CUDA_DIR)
	@echo "================================================================"
	@echo -e " $(BOLD)Test su demo.in  (K=$(K), thread/processi=$(NP))$(RESET)"
	@echo "================================================================"
	@## Seriale (riferimento)
	@if [ -x ./k-means ]; then \
		echo "=== Seriale ==="; \
		./k-means $(K) demo.in $(SERIAL_DIR)/demo.out 2>&1 | tee $(SERIAL_DIR)/demo.log; \
	else \
		echo -e "$(CYAN)[SKIP] k-means non disponibile$(RESET)"; \
	fi
	@## OpenMP
	@if [ -x ./omp-k-means ]; then \
		echo "=== OpenMP ($(NP) thread) ==="; \
		OMP_NUM_THREADS=$(NP) ./omp-k-means $(K) demo.in $(OMP_DIR)/demo.out 2>&1 | tee $(OMP_DIR)/demo.log; \
	else \
		echo -e "$(CYAN)[SKIP] omp-k-means non disponibile$(RESET)"; \
	fi
	@## MPI
	@if [ -x ./mpi-k-means ]; then \
		echo "=== MPI ($(NP) processi) ==="; \
		$(MPIRUN) $(NP) ./mpi-k-means $(K) demo.in $(MPI_DIR)/demo.out 2>&1 | tee $(MPI_DIR)/demo.log; \
	else \
		echo -e "$(CYAN)[SKIP] mpi-k-means non disponibile$(RESET)"; \
	fi
	@## CUDA
	@if [ -x ./cuda-k-means ]; then \
		echo "=== CUDA ==="; \
		./cuda-k-means $(K) demo.in $(CUDA_DIR)/demo.out 2>&1 | tee $(CUDA_DIR)/demo.log; \
	else \
		echo -e "$(CYAN)[SKIP] cuda-k-means non disponibile$(RESET)"; \
	fi
	@## Riepilogo
	@echo ""
	@echo "================================================================"
	@echo -e " $(BOLD)Riepilogo$(RESET)"
	@echo "================================================================"
	@printf "$(BOLD)%-12s %-12s %-12s$(RESET)\n" "Versione" "Tempo" "Correttezza"
	@echo "----------------------------------------------------------------"
	@## Seriale (baseline, nessun confronto)
	@if [ -f $(SERIAL_DIR)/demo.log ]; then \
		T=$$(grep "Elapsed time" $(SERIAL_DIR)/demo.log | awk '{print $$3}'); \
		printf "%-12s %-12s %-12s\n" "Seriale" "$$T" "-"; \
	fi
	@## Funzione di confronto con il seriale
	@check_vs_serial() { \
		local label=$$1 log=$$2 out=$$3; \
		local T; T=$$(grep "Elapsed time" $$log 2>/dev/null | awk '{print $$3}'); \
		if [ -f $(SERIAL_DIR)/demo.out ] && [ -f $$out ]; then \
			if diff <(grep -v '^#' $(SERIAL_DIR)/demo.out) <(grep -v '^#' $$out) > /dev/null 2>&1; then \
				printf "%-12s %-12s $(GREEN)%-12s$(RESET)\n" "$$label" "$$T" "OK"; \
			else \
				printf "%-12s %-12s $(RED)%-12s$(RESET)\n"   "$$label" "$$T" "ERRORE"; \
			fi; \
		elif [ -f $$log ]; then \
			printf "%-12s %-12s %-12s\n" "$$label" "$$T" "N/A"; \
		fi; \
	}; \
	[ -f $(OMP_DIR)/demo.log  ] && check_vs_serial "OpenMP" $(OMP_DIR)/demo.log  $(OMP_DIR)/demo.out; \
	[ -f $(MPI_DIR)/demo.log  ] && check_vs_serial "MPI"    $(MPI_DIR)/demo.log  $(MPI_DIR)/demo.out; \
	[ -f $(CUDA_DIR)/demo.log ] && check_vs_serial "CUDA"   $(CUDA_DIR)/demo.log $(CUDA_DIR)/demo.out
	@echo "================================================================"

test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Uso: make test-file FILE=input.in [K=5] [NP=4]"; \
		exit 1; \
	fi
	@[ -f $(FILE) ] || { echo "ERRORE: file $(FILE) non trovato"; exit 1; }
	mkdir -p $(SERIAL_DIR) $(OMP_DIR) $(MPI_DIR) $(CUDA_DIR)
	@BASE=$$(basename $(FILE) .in); \
	echo "================================================================"; \
	echo " Test su $(FILE)  (K=$(K), thread=$(NP))"; \
	echo "================================================================"; \
	[ -x ./k-means ]      && { echo "[Seriale]"; ./k-means $(K) $(FILE) $(SERIAL_DIR)/$$BASE.out 2>&1 | tee $(SERIAL_DIR)/$$BASE.log; } || echo "[SKIP] k-means"; \
	[ -x ./omp-k-means ]  && { echo "[OpenMP $(NP) thread]"; OMP_NUM_THREADS=$(NP) ./omp-k-means $(K) $(FILE) $(OMP_DIR)/$$BASE.out 2>&1 | tee $(OMP_DIR)/$$BASE.log; } || echo "[SKIP] omp-k-means"; \
	[ -x ./mpi-k-means ]  && { echo "[MPI $(NP) processi]"; $(MPIRUN) $(NP) ./mpi-k-means $(K) $(FILE) $(MPI_DIR)/$$BASE.out 2>&1 | tee $(MPI_DIR)/$$BASE.log; } || echo "[SKIP] mpi-k-means"; \
	[ -x ./cuda-k-means ] && { echo "[CUDA]"; ./cuda-k-means $(K) $(FILE) $(CUDA_DIR)/$$BASE.out 2>&1 | tee $(CUDA_DIR)/$$BASE.log; } || echo "[SKIP] cuda-k-means"; \
	echo ""; \
	echo "================================================================"; \
	echo " Riepilogo $(FILE)"; \
	echo "================================================================"; \
	printf "$(BOLD)%-12s %-12s %-12s$(RESET)\n" "Versione" "Tempo" "Correttezza"; \
	echo "----------------------------------------------------------------"; \
	if [ -f $(SERIAL_DIR)/$$BASE.log ]; then \
		T=$$(grep "Elapsed time" $(SERIAL_DIR)/$$BASE.log | awk '{printf "%.3f", $$3*1000}'); \
		printf "%-12s %-12s %-12s\n" "Seriale" "$$T ms" "-"; \
	fi; \
	for VER in omp mpi cuda; do \
		DIR=results/$$VER; \
		LABEL=$$(echo $$VER | tr '[:lower:]' '[:upper:]'); \
		[ -f $$DIR/$$BASE.log ] || continue; \
		T=$$(grep "Elapsed time" $$DIR/$$BASE.log | awk '{print $$3}'); \
		if [ -f $(SERIAL_DIR)/$$BASE.out ] && [ -f $$DIR/$$BASE.out ]; then \
			if diff <(grep -v '^#' $(SERIAL_DIR)/$$BASE.out) <(grep -v '^#' $$DIR/$$BASE.out) > /dev/null 2>&1; then \
				RES="$(GREEN)OK$(RESET)"; \
			else \
				RES="$(RED)ERRORE$(RESET)"; \
			fi; \
		else \
			RES="N/A"; \
		fi; \
		printf "%-12s %-12s " "$$LABEL" "$$T"; \
		echo -e "$$RES"; \
	done; \
	echo "================================================================"

bench:
	mkdir -p $(SERIAL_DIR) $(OMP_DIR) $(MPI_DIR) $(CUDA_DIR)
	@echo "================================================================"
	@echo -e " $(BOLD)BENCHMARK - K=$(K), thread/processi=$(NP)$(RESET)"
	@echo "================================================================"
	@for INPUT in $(INPUTS); do \
		[ -f $$INPUT ] || continue; \
		BASE=$$(basename $$INPUT .in); \
		echo ""; \
		echo "--- Input: $$INPUT ---"; \
		[ -x ./k-means ]      && { echo "[Seriale]";          ./k-means $(K) $$INPUT $(SERIAL_DIR)/$$BASE.out 2>&1 | tee $(SERIAL_DIR)/$$BASE.log; } || echo "[SKIP] k-means"; \
		[ -x ./omp-k-means ]  && { echo "[OpenMP $(NP) thread]"; OMP_NUM_THREADS=$(NP) ./omp-k-means $(K) $$INPUT $(OMP_DIR)/$$BASE.out 2>&1 | tee $(OMP_DIR)/$$BASE.log; } || echo "[SKIP] omp-k-means"; \
		[ -x ./mpi-k-means ]  && { echo "[MPI $(NP) processi]";  $(MPIRUN) $(NP) ./mpi-k-means $(K) $$INPUT $(MPI_DIR)/$$BASE.out 2>&1 | tee $(MPI_DIR)/$$BASE.log; } || echo "[SKIP] mpi-k-means"; \
		[ -x ./cuda-k-means ] && { echo "[CUDA]";             ./cuda-k-means $(K) $$INPUT $(CUDA_DIR)/$$BASE.out 2>&1 | tee $(CUDA_DIR)/$$BASE.log; } || echo "[SKIP] cuda-k-means"; \
	done
	@echo ""
	@echo "================================================================"
	@echo -e " $(BOLD)Riepilogo tempi e correttezza$(RESET)"
	@echo "================================================================"
	@printf "$(BOLD)%-30s %-10s %-10s %-10s %-10s %-8s %-8s %-8s$(RESET)\n" \
		"Input" "Ser(ms)" "OMP(ms)" "MPI(ms)" "CUDA(ms)" "OMP" "MPI" "CUDA"
	@echo "----------------------------------------------------------------"
	@for INPUT in $(INPUTS); do \
		[ -f $$INPUT ] || continue; \
		BASE=$$(basename $$INPUT .in); \
		T_SER=$$(grep  "Elapsed time" $(SERIAL_DIR)/$$BASE.log 2>/dev/null | awk '{printf "%.3f", $$3*1000}'); \
		T_OMP=$$(grep  "Elapsed time" $(OMP_DIR)/$$BASE.log    2>/dev/null | awk '{print $$3}'); \
		T_MPI=$$(grep  "Elapsed time" $(MPI_DIR)/$$BASE.log    2>/dev/null | awk '{print $$3}'); \
		T_CUDA=$$(grep "Elapsed time" $(CUDA_DIR)/$$BASE.log   2>/dev/null | awk '{print $$3}'); \
		OMP_STR="-"; MPI_STR="-"; CUDA_STR="-"; \
		[ -f $(SERIAL_DIR)/$$BASE.out ] && [ -f $(OMP_DIR)/$$BASE.out ]  && \
			{ diff <(grep -v '^#' $(SERIAL_DIR)/$$BASE.out) <(grep -v '^#' $(OMP_DIR)/$$BASE.out)  > /dev/null 2>&1 && OMP_STR="$(GREEN)OK$(RESET)"  || OMP_STR="$(RED)ERR$(RESET)"; }; \
		[ -f $(SERIAL_DIR)/$$BASE.out ] && [ -f $(MPI_DIR)/$$BASE.out ]  && \
			{ diff <(grep -v '^#' $(SERIAL_DIR)/$$BASE.out) <(grep -v '^#' $(MPI_DIR)/$$BASE.out)  > /dev/null 2>&1 && MPI_STR="$(GREEN)OK$(RESET)"  || MPI_STR="$(RED)ERR$(RESET)"; }; \
		[ -f $(SERIAL_DIR)/$$BASE.out ] && [ -f $(CUDA_DIR)/$$BASE.out ] && \
			{ diff <(grep -v '^#' $(SERIAL_DIR)/$$BASE.out) <(grep -v '^#' $(CUDA_DIR)/$$BASE.out) > /dev/null 2>&1 && CUDA_STR="$(GREEN)OK$(RESET)" || CUDA_STR="$(RED)ERR$(RESET)"; }; \
		printf "%-30s %-10s %-10s %-10s %-10s " "$$BASE" "$$T_SER" "$$T_OMP" "$$T_MPI" "$$T_CUDA"; \
		echo -e "$$OMP_STR     $$MPI_STR     $$CUDA_STR"; \
	done
	@echo "================================================================"

clean:
	\rm -f k-means omp-k-means mpi-k-means cuda-k-means inputgen *.o *~ \
	       centroids_*.txt clusters_*.txt img_*.png demo.out demo.mp4
	\rm -rf $(MOVIE_DIR) results/

valgrind: k-means omp-k-means
	@echo "================================================================"
	@echo " VALGRIND - controllo memory leak"
	@echo "================================================================"
	@echo ""
	@echo "--- Seriale ---"
	valgrind --leak-check=full --error-exitcode=1 ./k-means $(K) demo.in /dev/null
	@echo ""
	@echo "--- OpenMP (1 thread) ---"
	OMP_NUM_THREADS=1 valgrind --leak-check=full --error-exitcode=1 ./omp-k-means $(K) demo.in /dev/null
	@echo ""
	@echo "================================================================"
	@echo " Fine valgrind"
	@echo "================================================================"
