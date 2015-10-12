#include "mex.h"

void quickSort(double *x, int l, int r);
int  partition(double *x, int l, int r);


/* ------------------------------------------------------------------ */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
/* ------------------------------------------------------------------ */
{  const mxArray *b;
   mxArray *x;
   double   tau, normB, z;
   double  *ptrX, *ptrB;
   double   csb; /* Cumulative sum of b */
   double   alpha, alphaPrev;
   int      i, n;

   /* Expected arguments are (b, tau) */
    b    = prhs[0];
    tau  = mxGetScalar(prhs[1]);

    /* Get length of vector */
    n = mxGetDimensions(b)[0];
 
    /* Check quick exit condition (small tau) */
    if (tau < 1e-14)
    {  plhs[0] = mxCreateNumericMatrix(n,1,mxDOUBLE_CLASS,0);
       return ;
    }

    /* Copy input vector for sorting and set as output vector */
    x = mxDuplicateArray(b);
    ptrX = mxGetPr(x);
    ptrB = mxGetPr(b);
    plhs[0] = x;

    /* Compute norm of b (remember, entries are assumed nonnegative) */
    normB = 0;
    for (i=0; i < n; i++) normB += ptrB[i]; 

    /* Check quick exit condition (solution is b) */
    if (normB <= tau)
    {  return ;
    }

    

    /* Sort vector x */
    quickSort(ptrX,0,n-1);

    /* Initialize */
    csb       = -1 * tau;
    alphaPrev = 0;

    for (i = 0; i < n; i++)
    {  csb  += ptrX[i];
       alpha = csb / (i + 1);

       /* We are done as soon as the constraint can be satisfied */
       /* without exceding the current minimum value of b        */
       if (alpha >= ptrX[i]) break;

       alphaPrev = alpha;
    }

    /* Set the solution by applying soft-thresholding with the */
    /* previous value of alpha                                 */
    for (i = 0; i < n; i++)
    {  z = ptrB[i];
       ptrX[i] = (z < alphaPrev) ? 0 : z - alphaPrev; 
    }

    return;
}


/* ------------------------------------------------------------------ */
void quickSort(double *x, int l, int r)
/* ------------------------------------------------------------------ */
{  int j;

   if (l < r)
   {
      j = partition(x, l, r);
      quickSort(x, l,   j-1);
      quickSort(x, j+1, r  );
   }
}


/* ------------------------------------------------------------------ */
int  partition(double *x, int l, int r)
/* ------------------------------------------------------------------ */
{  double pivot, t;
   int    i, j;

   pivot = x[l];
   i     = l;
   j     = r+1;
		
   while(1)
   {
      do ++i; while(x[i] >= pivot && i <= r);
      do --j; while(x[j] <  pivot          );
      if (i >= j) break;

      /* Swap elements i and j */
      t    = x[i];
      x[i] = x[j];
      x[j] = t;
   }

   /* Swap elements l and j*/
   t    = x[l];
   x[l] = x[j];
   x[j] = t;

   return j;
}
