/* min-dist-dc.c (Roland Teodorowitsch; 17 Sep. 2020)
 * Compilation: gcc min-dist-dc.c -o min-dist-dc -fopenmp -lm
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

#define SIZE 10000000
#define START 1000000
#define STEP  1000000

#define EPS 0.000000001
#define BRUTEFORCESSIZE 200

typedef struct {
    double x;
    double y;
} point_t;

point_t points[SIZE];
point_t border[SIZE];

unsigned long long llrand() {
    unsigned long long r = 0;
    for (int i = 0; i < 5; ++i)
        r = (r << 15) | (rand() & 0x7FFF);
    return r & 0xFFFFFFFFFFFFFFFFULL;
}

int equals(double a, double b) {
    double diff = a-b;
    if (diff < 0.0) diff = -diff;
    if (diff < EPS)
        return 1;
    return 0;
}

int points_have_equals(point_t *points, int size) {
    int i, j;
    for (i=0; i<size-1; ++i) {
        j=i+1;
        if (equals(points[i].x,points[j].x) && equals(points[i].y,points[j].y))
           return 1;
    }
    return 0;
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

int points_qsort_comparator(const void *v1, const void *v2) {
    const point_t *p1 = (point_t *)v1;
    const point_t *p2 = (point_t *)v2;
    if (equals(p1->x,p2->x)) {
        if (equals(p1->y,p2->y))
            return 0;
        if (p1->y > p2->y)
            return 1;
        return -1;
    }
    if (p1->x > p2->x)
        return 1;
    return -1;
}

int points_qsort_comparator_y(const void *v1, const void *v2) {
    const point_t *p1 = (point_t *)v1;
    const point_t *p2 = (point_t *)v2;
    double diff = p1->y - p2->y;
    if (diff < 0.0) diff = -diff;
    if (diff < EPS)
        return p1->x < p2->x;
    return p1->y < p2->y;
}

double points_distance_sqr(point_t *p1, point_t *p2) {
    double dx, dy;
    dx = p1->x - p2->x;
    dy = p1->y - p2->y;
    return dx*dx + dy*dy;
}

double points_min_distance_dc(point_t *points, point_t *border, int l, int r) { /* recursive divide and conquer */
    double minDist = DBL_MAX;
    double dist;
    int i, j;
    if(r-l+1 <= BRUTEFORCESSIZE){
        for(i=l; i<=r; i++){
            for(j = i+1; j<=r; j++) {
                dist = points_distance_sqr(points+i, points+j);
                if(dist<minDist){
                    minDist = dist;
                }
            }
        }
        return minDist;
    }

    int m = (l+r)/2;
    double dL = points_min_distance_dc(points,border,l,m);
    double dR = points_min_distance_dc(points,border,m,r);
    minDist = (dL < dR ? dL : dR);

    int k = l;
    for(i=m-1; i>=l && abs(points[i].x-points[m].x)<minDist; i--){
        border[k++] = points[i];
    }
    for(i=m+1; i<=r && abs(points[i].x-points[m].x)<minDist; i++){
        border[k++] = points[i];
    }

    if (k-l <= 1) return minDist;

    qsort((void *)border+l, k-l, sizeof(point_t), points_qsort_comparator_y);

    for(i=l; i<k; i++){
        for(j=i+1; j<k && border[j].y - border[i].y < minDist; j++){
            dist = points_distance_sqr(border+i, border+j);
            if (dist < minDist){
                minDist = dist;
            }
        }
    }
    return minDist;
}

int main() {
    int i;
	double start, finish;
    
    points_generate(points,SIZE,11);
    qsort((void *)points, SIZE, sizeof(point_t), points_qsort_comparator);
    if (points_have_equals(points,SIZE)) {
        fprintf(stderr,"Sorry, invalid point set generated...\n");
        return 1;
    }
    for (int i=START; i<=SIZE; i+=STEP) {
        start = omp_get_wtime();  
        printf("%.6lf\n", sqrt(points_min_distance_dc(points,border,0,i-1)));
        finish = omp_get_wtime();  
        fprintf(stderr,"%d %lf\n",i,finish-start);
    }
    return 0;
}