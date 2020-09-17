#include <cstdio>
#include <cmath>
#include "ladsolver.h"

/**
 *   standard partition process
 *   Consider the last element as pivot, and moves all smaller
 *   element to left of it and greater elements to right of it
 *   The actual movement is recorded in the index table idx[] 
 */
 
int Partition(const double* v, int* idx, int l, int r) 
{
    double last_element = v[idx[r]];
    int i = l, tmp;
    for (int j = l; j < r; j++) 
    {
        if (v[idx[j]] <= last_element)
        { 
            // swap i-th and j-th element
            tmp = idx[i];
            idx[i] = idx[j];
            idx[j] = tmp;
            i++; 
        }
    }
    // swap i-th and r-th element
    tmp = idx[i];
    idx[i] = idx[r];
    idx[r] = tmp;
    return i;
}

/**
 *   quick select routine for weighted median
 */
  
int QuickSelect(const double* u, const double* v, int* idx, int l, int r, double th)
{ 
    int k = Partition(v, idx, l, r);
    double sum = 0.0;
    for (int i = l;i < k; ++i)
    {
        sum += u[idx[i]];
    }
    if (sum < th && sum + u[idx[k]] >= th)
        return idx[k];
    if (sum >= th)
        return QuickSelect(u, v, idx, l, k - 1, th);
    return QuickSelect(u, v, idx, k + 1, r, th - sum - u[idx[k]]);
}

/**
 *   weighted median
 *   given values v[0],v[1],...,v[n-1] and corresponding non-negative weights 
 *   u[0],u[1],...,u[n-1], return the index t such that v[t] minimizes ∑ u[k]*|v-v[k]|
 */

int WeightedMedian(const double* u, const double* v, int n)
{
    int* idx = (int*) malloc(n*sizeof(int));
    double halfsum = 0.0;
    for (int i = 0;i < n; ++i)
    {
        idx[i] = i;
        halfsum += u[i];
    }
    halfsum *= 0.5;
    int t = QuickSelect(u, v, idx, 0, n - 1, halfsum);
    free(idx);
    return t;
}

/**
 *   solve least absolute deviation (LAD) problem
 *   given n-by-r array X and n-by-1 array y, the function will find an
 *   r-by-1 array b such that b minimizes the residual ||X*b - y||_1
 */

int LADSolver(int n, int r, const double* X, const double* y, double* b)
{
    double* A = (double*) malloc((n + r)*(r + 1)*sizeof(double));
    double* g = (double*) malloc(r*sizeof(double));
    double* h = (double*) malloc(r*sizeof(double));
    double* u = (double*) malloc(n*sizeof(double));
    double* v = (double*) malloc(n*sizeof(double));

    int iter = 0, max_iter = 20;
    const double eps = 1.0e-12;
    double gmin, Aij;
    int i, j, p, q;

    // initialize matrix A
    // A = [X y]
    //     [I 0]
    for (i = 0;i < n; ++i)
    {
        for (j = 0;j < r; ++j)
        {
            A[i*(r + 1) + j] = X[i*r + j];
        }
        A[i*(r + 1) + r] = y[i];
    }
    for (i = 0;i < r; ++i)
    {
        for (j = 0;j <= r; ++j)
        {
            A[(n + i)*(r + 1) + j] = 0.0;
        }
        A[(n + i)*(r + 1) + i] = 1.0;
    }

    // start iteration
    while (iter < max_iter)
    {
        iter++;
        // compute directional derivatives
        for (j = 0;j < r; ++j)
        {   
            g[j] = 0.0;
            h[j] = 0.0;
        }
        for (i = 0;i < n; ++i)
        {
            Aij = A[i*(r + 1) + r];
            if (fabs(Aij) < eps)
            {
                for (j = 0;j < r; ++j)
                {
                    g[j] += fabs(A[i*(r + 1) + j]);
                }
            }
            else 
            {
                if (Aij > 0)
                {
                    for (j = 0;j < r; ++j)
                    {
                        h[j] += A[i*(r + 1) + j];
                    }
                }
                else
                {
                    for (j = 0;j < r; ++j)
                    {
                        h[j] -= A[i*(r + 1) + j];
                    }
                }
            }
        }
        for (j = 0;j < r; ++j)
        {   
            double sum = 1.e-12;
            for (i = 0; i < n; ++i)
            {
                sum += fabs(A[i*(r + 1) + j]);
            }
            g[j] -= fabs(h[j]);
            g[j] /= sum; // (Bloomfield and Steiger's heuristic)
        }

        // determine the steepest direction
        gmin = g[0];
        p = 0;
        for (j = 1;j < r; ++j)
        {
            if (g[j] < gmin)
            {
                p = j;
                gmin = g[j];
            }
        }

        // if all the directional derivatives are positive,
        // then the optimality is achieved
        if (gmin >= 0)
            break;

        for (i = 0; i < n; ++i)
        {
            u[i] = fabs(A[i*(r + 1) + p]);
            v[i] = (u[i] < eps)?0.0:(A[i*(r + 1) + r]/A[i*(r + 1) + p]);
        }

        // find index of median of v weighted by u
        q = WeightedMedian(u, v, n);

        // pivot on (q, p)

        /**
         * 
         *   [Aij  Aip] → [ Aij-Aqj*Aip/Aqp   Aip/Aqp]  
         *   [Aqj  Aqp] → [       0              1   ]
         * 
         */
         
        Aij = A[q*(r + 1)+p];
        for (i = 0; i < n+r; ++i)
        {
            A[i*(r + 1)+p] /= Aij;
        }
        for (j = 0; j < r + 1; ++j)
        {
            if (j == p)
                continue;
            Aij = A[q*(r + 1) + j];
            for (i = 0; i < n + r; ++i)
            {
                A[i*(r + 1) + j] -= Aij * A[i*(r + 1) + p];
            }
        }
    }
    //printf((iter < max_iter)?(" BR converges in %2d steps.\n"):(" BR fails to converge in %2d steps.\n"), iter);

    // output optimal solution
    for (j = 0;j < r; ++j)
    {
        b[j] = -A[(n + j)*(r + 1) + r];
    }

    free(A);
    free(g);
    free(h);
    free(u);
    free(v);

    return 0;
}
