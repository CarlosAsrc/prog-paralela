/* min-dist-dc3.cpp (Roland Teodorowitsch; 29 out. 2020)
 * Compilation: mpic++ -o min-dist-dc-mpi min-dist-dc3.cpp -lm
 * Note: Includes some code from the sequential solution of the
 *       "Closest Pair of Points" problem from the
 *       14th Marathon of Parallel Programming avaiable at
 *       http://lspd.mackenzie.br/marathon/19/points.zip
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <mpi.h>

#define SIZE 10000000
#define START 1000000
#define STEP 1000000

#define EPS 0.00000000001
#define BRUTEFORCESSIZE 200

using namespace std;

typedef struct
{
    double x;
    double y;
} point_t;

point_t points[SIZE];
point_t border[SIZE];

unsigned long long llrand()
{
    unsigned long long r = 0;
    int i;
    for (i = 0; i < 5; ++i)
        r = (r << 15) | (rand() & 0x7FFF);
    return r & 0xFFFFFFFFFFFFFFFFULL;
}

void points_generate(point_t *points, int size, int seed)
{
    int p, i, found;
    double x, y;
    srand(seed);
    p = 0;
    while (p < size)
    {
        x = ((double)(llrand() % 20000000000) - 10000000000) / 1000.0;
        y = ((double)(llrand() % 20000000000) - 10000000000) / 1000.0;
        if (x >= -10000000.0 && x <= 10000000.0 && y >= -10000000.0 && y <= 10000000.0)
        {
            points[p].x = x;
            points[p].y = y;
            p++;
        }
    }
}

bool compX(const point_t &a, const point_t &b)
{
    if (a.x == b.x)
        return a.y < b.y;
    return a.x < b.x;
}

bool compY(const point_t &a, const point_t &b)
{
    if (a.y == b.y)
        return a.x < b.x;
    return a.y < b.y;
}

double points_distance_sqr(point_t *p1, point_t *p2)
{
    double dx, dy;
    dx = p1->x - p2->x;
    dy = p1->y - p2->y;
    return dx * dx + dy * dy;
}

double points_min_distance_dc(point_t *point, point_t *border, int l, int r)
{

    const int nitems = 2;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype mpi_point_t;
    MPI_Aint offsets[2];

    offsets[0] = offsetof(point_t, x);
    offsets[1] = offsetof(point_t, y);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_point_t);
    MPI_Type_commit(&mpi_point_t);

    int menor, menor1, menor2;
    int tamanho, tamanho1, tamanho2;
    int vetor[SIZE];
    int pai, filho1, filho2, p, id;
    MPI_Status status;
    double minDist = DBL_MAX;
    double dist;
    int i, j;

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    if (id == 0)
    {
        // NODO RAIZ
        tamanho = r - l + 1;
        // DIVIDE 1
        filho1 = id * 2 + 1;
        tamanho1 = tamanho / 2;
        MPI_Send((void *)&tamanho1, 1, MPI_INT, filho1, 0, MPI_COMM_WORLD);
        MPI_Send((void *)&point[0], tamanho1, mpi_point_t, filho1, 0, MPI_COMM_WORLD);
        MPI_Send((void *)&border[0], tamanho1, mpi_point_t, filho1, 0, MPI_COMM_WORLD);

        // DIVIDE 2
        filho2 = filho1 + 1;
        tamanho2 = tamanho - tamanho1;
        MPI_Send((void *)&tamanho2, 1, MPI_INT, filho2, 0, MPI_COMM_WORLD);
        MPI_Send((void *)&point[tamanho1], tamanho2, mpi_point_t, filho2, 0, MPI_COMM_WORLD);
        MPI_Send((void *)&border[tamanho1], tamanho2, mpi_point_t, filho2, 0, MPI_COMM_WORLD);

        // CONQUISTA
        MPI_Recv((void *)&menor1, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        MPI_Recv((void *)&menor2, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        minDist = (menor1 < menor1 ? menor1 : menor2);
        return minDist;
    }
    else
    {

        MPI_Recv((void *)&tamanho, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        MPI_Recv((void *)&point[0], tamanho, mpi_point_t, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv((void *)&border[0], tamanho, mpi_point_t, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pai = status.MPI_SOURCE;

        if (id >= p / 2)
        {
            // NODO FOLHA
            for (i = l; i < tamanho - 1; i++)
            {
                for (j = i + 1; j <= tamanho - 1; j++)
                {
                    dist = points_distance_sqr(point + i, point + j);
                    if (dist < minDist)
                    {
                        minDist = dist;
                    }
                }
            }
            MPI_Send((void *)&minDist, 1, MPI_DOUBLE, pai, 0, MPI_COMM_WORLD);
        }
        else
        {
            // NODO INTERMEDIÁRIO

            // DIVIDE 1
            filho1 = id * 2 + 1;
            tamanho1 = tamanho / 2;
            MPI_Send((void *)&tamanho1, 1, MPI_INT, filho1, 0, MPI_COMM_WORLD);
            MPI_Send((void *)&point[0], tamanho1, mpi_point_t, filho1, 0, MPI_COMM_WORLD);
            MPI_Send((void *)&border[0], tamanho1, mpi_point_t, filho1, 0, MPI_COMM_WORLD);

            // DIVIDE 2
            filho2 = filho1 + 1;
            tamanho2 = tamanho - tamanho1;
            MPI_Send((void *)&tamanho2, 1, MPI_INT, filho2, 0, MPI_COMM_WORLD);
            MPI_Send((void *)&point[tamanho1], tamanho2, mpi_point_t, filho2, 0, MPI_COMM_WORLD);
            MPI_Send((void *)&border[tamanho1], tamanho2, mpi_point_t, filho2, 0, MPI_COMM_WORLD);

            // CONQUISTA
            MPI_Recv((void *)&menor1, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            MPI_Recv((void *)&menor2, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            minDist = (menor1 < menor1 ? menor1 : menor2);

            int m = (l + tamanho - 1) / 2;
            int k = l;
            for (i = m - 1; i >= l && fabs(point[i].x - point[m].x) < minDist; i--)
                border[k++] = point[i];
            for (i = m + 1; i <= tamanho - 1 && fabs(point[i].x - point[m].x) < minDist; i++)
                border[k++] = point[i];

            if (k - l <= 1)
                return minDist;

            sort(&border[l], &border[l] + (k - l), compY);

            for (i = l; i < k; i++)
            {
                for (j = i + 1; j < k && border[j].y - border[i].y < minDist; j++)
                {
                    dist = points_distance_sqr(border + i, border + j);
                    if (dist < minDist)
                        minDist = dist;
                }
            }

            MPI_Send((void *)&minDist, 1, MPI_DOUBLE, pai, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Type_free(&mpi_point_t);
    return 0;
}


int main(int argc, char *argv[])
{
    int i, p, id;
    double elapsed_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    if (id == 0)
    {
        points_generate(points, SIZE, 11);
        sort(&points[0], &points[SIZE], compX);
    }

    for (i = START; i <= SIZE; i += STEP)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        elapsed_time = -MPI_Wtime();
        if (id == 0)
            printf("%.6lf\n", sqrt(points_min_distance_dc(points, border, 0, i - 1)));
        else
        {
            points_min_distance_dc(points, border, 0, i - 1);
        }
        elapsed_time += MPI_Wtime();
        fprintf(stderr, "%d %lf\n", i, elapsed_time);
    }
    
    MPI_Finalize();
    return 0;
}