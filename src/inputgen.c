/****************************************************************************
 *
 * inputgen.c - Generate random input for the K-Means algorithm.
 *
 * Copyright (C) 2025 Moreno Marzolla
 * <https://unibo.it/sitoweb/moreno.marzolla/>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------------
 *
 * This program generates a random input for the K-Means algorithm.
 *
 * To compile:
 *
 * gcc -std=c99 -Wall -Wpedantic inputgen.c -o inputgen
 *
 * To execute:
 *
 * ./inputgen n_points n_dims n_clusters
 *
 * The program generates `n_points` that are distributed over
 * `n_clusters` hyper-rectangles in `n_dims` dimensions. Each
 * hyper-rectangle contains approximately `n_points / n_clusters`
 * random points.
 *
 * If `n_points` is not an integer multiple of `n_clusters`, then the
 * program assigns `n_points / n_clusters` (rounded down) points to
 * each rectangle; one extra point is assigned to the first `n_points
 * % n_clusters` clusters.
 *
 * Example: n_points=1000, n_clusters=6
 * n_points/n_clusters = 166
 * n_points%n_clusters = 4
 *
 * cluster 0: 167 points
 * cluster 1: 167 points
 * cluster 2: 167 points
 * cluster 3: 167 points
 * cluster 4: 166 points
 * cluster 5: 166 points
 *
 ****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

double rand01(void) { return rand() / (double)RAND_MAX; }

double randab(double a, double b) { return a + rand01() * (b - a); }

/* Generate `n` random points inside a `D`-dimensional box with center
   `center` and side `2*r`. */
void gen_points(float* center, int D, float r, int n) {
  for (int i = 0; i < n; i++) {
    for (int d = 0; d < D; d++) {
      printf("%f ", center[d] + randab(-r, r));
    }
    printf("\n");
  }
}

void init_center(float* p, int D) {
  for (int d = 0; d < D; d++) {
    p[d] = randab(0, 200);
  }
}

int main(int argc, char* argv[]) {
  int n_points, n_dims, n_clusters;
  int points_per_cluster, remainder;

  if (argc != 4) {
    fprintf(stderr, "Usage: %s n_points n_dims n_clusters\n", argv[0]);
    return EXIT_FAILURE;
  }

  srand(17); /* Deterministic initialization of the PRNG. */

  n_points = atoi(argv[1]);
  n_dims = atoi(argv[2]);
  n_clusters = atoi(argv[3]);
  points_per_cluster = n_points / n_clusters; /* rounded down */
  remainder = n_points % n_clusters;

  float* center = (float*)malloc(n_dims * sizeof(*center));

  for (int c = 0; c < n_clusters; c++) {
    const double side = randab(10, 30);
    const int this_cluster_points = points_per_cluster + (c < remainder);
    init_center(center, n_dims);
    gen_points(center, n_dims, side, this_cluster_points);
  }

  free(center);
  return EXIT_SUCCESS;
}
