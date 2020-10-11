/* min-dist-bf.c (Roland Teodorowitsch; 17 Sep. 2020)
 * Compilation: gcc min-dist-bf.c -o min-dist-bf -fopenmp -lm
 * Note: Includes some code from the sequential solution of the
 *       "Closest Pair of Points" problem from the
 *       14th Marathon of Parallel Programming avaiable at
 *       http://lspd.mackenzie.br/marathon/19/points.zip
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#define SIZE 100000
#define START 10000
#define STEP  10000

#define EPS 0.00000000001

typedef struct {
    double x;
    double y;
} point_t;

point_t points[SIZE];

unsigned long long llrand() {
    unsigned long long r = 0;
    for (int i = 0; i < 5; ++i)
        r = (r << 15) | (rand() & 0x7FFF);
    return r & 0xFFFFFFFFFFFFFFFFULL;
}

void points_generate(point_t *points, int size, int seed) {
    int p, i, found;
    double x, y;
    srand(seed);
    p = 0;
    while (p<size) {
        x = ((double)(llrand() % 20000000000) - 10000000000) / 1000.0;
        y = ((double)(llrand() % 20000000000) - 10000000000) / 1000.0;
        if (x >= -10000000.0 && x <= 10000000.0 && y >= -10000000.0 && y <= 10000000.0) {
            points[p].x = x;
            points[p].y = y;
            p++;
        }
    }
}

double points_distance_sqr(point_t *p1, point_t *p2) {
    double dx, dy;
    dx = p1->x - p2->x;
    dy = p1->y - p2->y;
    return dx*dx + dy*dy;
}

double points_min_distance_bf(point_t *points, int size) { /* bf = brute-force */
    int i, j;
    double min_d, d;
    min_d = DBL_MAX;
    for (i=0; i< size-1; ++i) {
        for (j=i+1; j<size; ++j) {
            d = points_distance_sqr(points+i,points+j);
            if (d < min_d)
                min_d = d;
        }
    }
    return sqrt(min_d);
}

int main() {
    int i;
	double start, finish;
    
    points_generate(points,SIZE,0);
    for (int i=START; i<=SIZE; i+=STEP) {
        start = omp_get_wtime();  
        printf("%.6lf\n", points_min_distance_bf(points,i));
        finish = omp_get_wtime();  
        fprintf(stderr,"%d %lf\n",i,finish-start);
    }
    return 0;
}
