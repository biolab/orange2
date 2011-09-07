// This code is derived from code in the Rational Fortran file dmarss.r which is
// part of the R and S mda package by Hastie and Tibshirani.
// Comments containing "TODO" mark known issues
//
// See the R earth documentation for descriptions of the principal data structures.
// See also www.milbo.users.sonic.net.
//
// Stephen Milborrow Feb 2007 Petaluma
//
//-----------------------------------------------------------------------------
// ...
//-----------------------------------------------------------------------------
// References:
//
// HastieTibs: Trevor Hastie and Robert Tibshirani
//      S library mda version 0.3.2 dmarss.r Ratfor code
//      Modifications for R by Kurt Hornik, Friedrich Leisch, Brian Ripley
//
// FriedmanMars: Multivariate Adaptive Regression Splines (with discussion)
//      Annals of Statistics 19/1, 1--141, 1991
//
// FriedmanFastMars: Friedman "Fast MARS"
//      Dep. of Stats. Stanford, Tech Report 110, May 1993
//
// Miller: Alan Miller (2nd ed. 2002) Subset Selection in Regression
//
//-----------------------------------------------------------------------------
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// A copy of the GNU General Public License is available at
// http://www.r-project.org/Licenses
//
//-----------------------------------------------------------------------------

/*
    This file is part of Orange.

    Copyright 1996-2011 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 	 Changes to earth.c from earth R package:
 	 - Added defines for STANDALONE, USING_BLAS, _DEBUG
 	 - Removed  #include <crtdbg.h> for windows
 	 - Fix defines for ISNAN and FINITE to work on non MSC compilers
 	 - Removed debugging code for windows
 	 - Removed definitions of bool, true false
 	 - Define _C_ as "C" for all compilers
 	 - Define c linkage for error, xerbla
 	 - Replaced POS_INF static global variable with numeric_limits<double>::infinity()
 	 - Added #include <limits>
 	 - Changed include of earth.h to earth.ppp and moved it before the module level defines
 	 - Changed EvalSubsetsUsingXtX to return an error code if lin. dep. terms in bx

	- TODO: Move global vars inside the functions using them (most are local)
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <limits>

#include "earth.ppp"

#define STANDALONE 1
#define USING_BLAS 1
#define _DEBUG 0

#if !STANDALONE
#define USING_R 1
#endif // STANDALONE


#if _MSC_VER && _DEBUG
    #include <crtdbg.h> // microsoft malloc debugging library
#endif

#if _MSC_VER            // microsoft
    #define _C_ "C"
    #if _DEBUG          // debugging enabled?
        // disable warning: too many actual params for macro (for malloc1 and calloc1)
        #pragma warning(disable: 4002)
    #endif
#else
    #define _C_ "C"
//    #ifndef bool
//        typedef int bool;
//        #define false 0
//        #define true  1
//    #endif
#endif

#if USING_R             // R with gcc
    #include "R.h"
    #include "Rinternals.h" // needed for Allowed function handling
    #include "allowed.h"
    #define printf Rprintf
    #define FINITE(x) R_FINITE(x)
    #define ASSERT(x)   \
        if (!(x)) error("internal assertion failed in file %s line %d: %s\n", \
                        __FILE__, __LINE__, #x)
#else
    #define warning printf
    extern "C" { void error(const char *args, ...); }
	#ifdef _MSC_VER
		#define ISNAN(x)  _isnan(x)
		#define FINITE(x) _finite(x)
	#else
		#define ISNAN(x)  isnan(x)
		#define FINITE(x) finite(x)
	#endif // _MSC_VER

    #define ASSERT(x)   \
        if (!(x)) error("internal assertion failed in file %s line %d: %s\n", \
                        __FILE__, __LINE__, #x)
#endif

//#include "earth.h"

extern _C_ int dqrdc2_(double *x, int *ldx, int *n, int *p,
                        double *tol, int *rank,
                        double *qraux, int *pivot, double *work);

extern _C_ int dqrsl_(double *x, int *ldx, int *n, int *k,
                        double *qraux, double *y,
                        double *qy, double *qty, double *b,
                        double *rsd, double *xb, int *job, int *info);

extern _C_ void dtrsl_(double *t, int *ldt, int *n, double *b, int *job, int *info);

extern _C_ void daxpy_(const int *n, const double *alpha,
                        const double *dx, const int *incx,
                        double *dy, const int *incy);

extern _C_ double ddot_(const int *n,
                        const double *dx, const int *incx,
                        const double *dy, const int *incy);

#define sq(x)       ((x) * (x))
#ifndef max
#define max(a,b)    (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b)    (((a) < (b)) ? (a) : (b))
#endif

#define INLINE      inline
#define USE_BLAS    1     // 1 is faster (tested on Windows XP Pentium with R BLAS)
                          // also, need USE_BLAS to use bxOrthCenteredT

#define FAST_MARS   1     // 1 to use techniques in FriedmanFastMars (see refs)

#define IOFFSET     1     // printfs only: 1 to convert 0-based indices to 1-based in printfs
                          // use 0 for C style indices in messages to the user

static const char   *VERSION    = "version 3.2-0"; // change if you modify this file!
static const double BX_TOL      = 0.01;
static const double QR_TOL      = 0.01;
static const double MIN_GRSQ    = -10.0;
static const double ALMOST_ZERO = 1e-10;
static const int    ONE         = 1;        // parameter for BLAS routines
#if _MSC_VER                                // microsoft compiler
static const double ZERO        = 0.0;
//static const double POS_INF     = (1.0 / ZERO);
static const double POS_INF  	= std::numeric_limits<double>::infinity();
#else
//static const double POS_INF     = (1.0 / 0.0);
static const double POS_INF  	= std::numeric_limits<double>::infinity();
#endif
static const int    MAX_DEGREE  = 100;

// Poor man's array indexing -- not pretty, but pretty useful.
//
// Note that we use column major ordering. C programs usually use row major
// ordering but we don't here because the functions in this file are called
// by R and call Fortran routines which use column major ordering.

#define Dirs_(iTerm,iPred)      Dirs[(iTerm) + (iPred)*(nMaxTerms)]
#define Cuts_(iTerm,iPred)      Cuts[(iTerm) + (iPred)*(nMaxTerms)]

#define bx_(iCase,iTerm)                bx             [(iCase) + (iTerm)*(nCases)]
#define bxOrth_(iCase,iTerm)            bxOrth         [(iCase) + (iTerm)*(nCases)]
#define bxOrthCenteredT_(iTerm,iCase)   bxOrthCenteredT[(iTerm) + (iCase)*(nMaxTerms)]
#define x_(iCase,iPred)                 x              [(iCase) + (iPred)*(nCases)]
#define xOrder_(iCase,iPred)            xOrder         [(iCase) + (iPred)*(nCases)]
#define y_(iCase,iResp)                 y              [(iCase) + (iResp)*(nCases)]
#define Residuals_(iCase,iResp)         Residuals      [(iCase) + (iResp)*(nCases)]
#define ycboSum_(iTerm,iResp)           ycboSum        [(iTerm) + (iResp)*(nMaxTerms)]
#define Betas_(iTerm,iResp)             Betas          [(iTerm) + (iResp)*(nUsedCols)]

// Global copies of some input parameters.  These stay constant for the entire MARS fit.
static double TraceGlobal;      // copy of Trace parameter
static int nMinSpanGlobal;      // copy of nMinSpan parameter

static void FreeBetaCache(void);
static char *sFormatMemSize(const unsigned MemSize, const bool Align);

//-----------------------------------------------------------------------------
// malloc and its friends are redefined (a) so under Microsoft C using
// crtdbg.h we can easily track alloc errors and (b) so FreeR() doesn't
// re-free any freed blocks and (c) so out of memory conditions are
// immediately detected.
// So DON'T USE free, malloc, and calloc.  Use free1, malloc1, and calloc1 instead.

// free1 is a macro so we can zero p
#define free1(p) { if (p) free(p); p = NULL; }

#if _MSC_VER && _DEBUG  // microsoft C and debugging enabled?

#define malloc1(size) _malloc_dbg((size), _NORMAL_BLOCK, __FILE__, __LINE__)
#define calloc1(num, size) \
                      _calloc_dbg((num), (size), _NORMAL_BLOCK, __FILE__, __LINE__)
#else
static void *malloc1(size_t size, const char *args, ...)
{
    void *p = malloc(size);
    if (!p || TraceGlobal == 1.5) {
        if (args == NULL)
            printf("malloc %s\n", sFormatMemSize(size, true));
        else {
            char s[100];
            va_list p;
            va_start(p, args);
            vsprintf(s, args, p);
            va_end(p);
            printf("malloc %s: %s\n", sFormatMemSize(size, true), s);
        }
        fflush(stdout);
    }
    if (!p)
        error("Out of memory (could not allocate %s)", sFormatMemSize(size, false));
    return p;
}

static void *calloc1(size_t num, size_t size, const char *args, ...)
{
    void *p = calloc(num, size);
    if (!p || TraceGlobal == 1.5) {
        if (args == NULL)
            printf("calloc %s\n", sFormatMemSize(size, true));
        else {
            char s[100];
            va_list p;
            va_start(p, args);
            vsprintf(s, args, p);
            va_end(p);
            printf("calloc %s: %s\n", sFormatMemSize(size, true), s);
        }
        fflush(stdout);
    }
    if (!p)
        error("Out of memory (could not allocate %s)", sFormatMemSize(size, false));
    return p;
}
#endif

// After calling this, on program termination we will get a report if there are
// writes outside the borders of allocated blocks or if there are non-freed blocks.

#if _MSC_VER && _DEBUG          // microsoft C and debugging enabled?
static void InitMallocTracking(void)
{
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_WNDW);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
    int Flag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    Flag |= (_CRTDBG_ALLOC_MEM_DF|_CRTDBG_DELAY_FREE_MEM_DF|_CRTDBG_LEAK_CHECK_DF);
    _CrtSetDbgFlag(Flag);
}
#endif

//-----------------------------------------------------------------------------
// These are malloced blocks.  They unfortunately have to be declared globally so
// under R if the user interrupts we can free them using on.exit(.C("FreeR"))

static int    *xOrder;      // local to FindTerm
static bool   *WorkingSet;  // local to FindTerm and EvalSubsets
static double *xbx;         // local to FindTerm
static double *CovSx;       // local to FindTerm
static double *CovCol;      // local to FindTerm
static double *ycboSum;     // local to FindTerm (used to be called CovSy)
static double *bxOrth;      // local to ForwardPass
static double *yMean;       // local to ForwardPass
static double *Weights;     // local to ForwardPass and EvalSubsetsUsingXtx

// Transposed and mean centered copy of bxOrth, for fast update in FindKnot.
// It's faster because there is better data locality as iTerm increases, so
// better L1 cache use.  This is used only if USE_BLAS is true.

static double *bxOrthCenteredT; // local to ForwardPass

static double *bxOrthMean;      // local to ForwardPass
static int  *nFactorsInTerm;    // local to Earth or ForwardPassR
static int  *nUses;             // local to Earth or ForwardPassR
#if USING_R
static int *iDirs;              // local to ForwardPassR
static bool *BoolFullSet;       // local to ForwardPassR
#endif
#if FAST_MARS
static void FreeQ(void);
#endif

#if USING_R
void FreeR(void)                // for use by R
{
    free1(WorkingSet);
    free1(CovSx);
    free1(CovCol);
    free1(ycboSum);
    free1(xOrder);
    free1(bxOrthMean);
    free1(bxOrthCenteredT);
    free1(bxOrth);
    free1(yMean);
    free1(Weights);
    free1(BoolFullSet);
    free1(iDirs);
    free1(nUses);
    free1(nFactorsInTerm);
    FreeBetaCache();
#if FAST_MARS
    FreeQ();
#endif
}
#endif

//-----------------------------------------------------------------------------
static char *sFormatMemSize(const unsigned MemSize, const bool Align)
{
    static char s[100];
    double Size = (double)MemSize;
    if(Size >= 1e9)
        sprintf(s, Align? "%6.3f GB": "%.3g GB", Size / 1e9);
    else if(Size >= 1e6)
        sprintf(s, Align? "%6.0f MB": "%.3g MB", Size / 1e6);
    else if(Size >= 1e3)
        sprintf(s, Align? "%6.0f kB": "%.3g kB", Size / 1e3);
    else
        sprintf(s, Align? "%6.0f  B": "%g B", Size);
    return s;
}

//-----------------------------------------------------------------------------
// Gets called periodically to service the R framework.
// Will not return if the user interrupts.

#if USING_R

static INLINE void ServiceR(void)
{
    R_FlushConsole();
    R_CheckUserInterrupt();     // may never return
}

#endif

//-----------------------------------------------------------------------------
#if FAST_MARS

typedef struct tQueue {
    int     iParent;            // parent term
    double  RssDelta;
    int     nTermsForRssDelta;  // number of terms when RssDelta was calculated
    double  AgedRank;
} tQueue;

static tQueue *Q;       // indexed on iTerm (this Q is used for queue updates)
static tQueue *SortedQ; // indexed on iParent rank (this Q is used to get next iParent)
static int    nQMax;    // number of elements in Q

static void InitQ(const int nMaxTerms)
{
    int i;
    nQMax = 0;
    Q       = (tQueue *)malloc1(nMaxTerms * sizeof(tQueue),
                            "Q\t\t\tnMaxTerms %f sizeof(tQueue) %d",
                            nMaxTerms, sizeof(tQueue));
    SortedQ = (tQueue *)malloc1(nMaxTerms * sizeof(tQueue),
                            "SortedQ\t\tnMaxTerms %f sizeof(tQueue) %d",
                            nMaxTerms, sizeof(tQueue));
    for (i = 0; i < nMaxTerms; i++) {
        Q[i].iParent = i;
        Q[i].nTermsForRssDelta = -99;   // not strictly needed, nice for debugging
        Q[i].RssDelta = -1;
        Q[i].AgedRank = -1;
    }
}

static void FreeQ(void)
{
    free1(SortedQ);
    free1(Q);
}

static void PrintSortedQ(int nFastK)     // for debugging
{
    printf("\n\nSortedQ  QIndex Parent nTermsForRssDelta AgedRank  RssDelta\n");
    for (int i = 0; i < nQMax; i++) {
        printf("            %3d    %3d   %15d    %5.1f  %g\n",
            i+IOFFSET,
            SortedQ[i].iParent+IOFFSET,
            SortedQ[i].nTermsForRssDelta+IOFFSET,
            SortedQ[i].AgedRank,
            SortedQ[i].RssDelta);
        if (i == nFastK-1)
            printf("FastK %d ----------------------------------------------------\n",
                nFastK);
    }
}

// Sort so highest RssDeltas are at low indices.
// Secondary sort key is iParent.  Not strictly needed, but removes
// possible differences in qsort implementations (which "sort"
// identical keys unpredictably).

static int CompareQ(const void *p1, const void *p2)     // for qsort
{
    double Diff = ((tQueue*)p2)->RssDelta - ((tQueue*)p1)->RssDelta;
    if (Diff < 0)
        return -1;
    else if (Diff > 0)
        return 1;

    // Diff is 0, so sort now on iParent

    int iDiff = ((tQueue*)p1)->iParent - ((tQueue*)p2)->iParent;
    if (iDiff < 0)
        return -1;
    else if (iDiff > 0)
        return 1;
    return 0;
}

// Sort so lowest AgedRanks are at low indices.
// If AgedRanks are the same then sort on RssDelta and iParent.

static int CompareAgedQ(const void *p1, const void *p2) // for qsort
{
    double Diff = ((tQueue*)p1)->AgedRank - ((tQueue*)p2)->AgedRank;
    if (Diff < 0)
        return -1;
    else if (Diff > 0)
        return 1;

    // Diff is 0, so sort now on RssDelta

    Diff = ((tQueue*)p2)->RssDelta - ((tQueue*)p1)->RssDelta;
    if (Diff < 0)
        return -1;
    else if (Diff > 0)
        return 1;

    // Diff is still 0, so sort now on iParent

    int iDiff = ((tQueue*)p1)->iParent - ((tQueue*)p2)->iParent;
    if (iDiff < 0)
        return -1;
    else if (iDiff > 0)
        return 1;
    return 0;
}

static void AddTermToQ(
    const int iTerm,        // in
    const int nTerms,       // in
    const double RssDelta,  // in
    const bool Sort,        // in
    const int nMaxTerms,    // in
    const double FastBeta)  // in: ageing Coef, 0 is no ageing, FastMARS recommends 1
{
    ASSERT(iTerm < nMaxTerms);
    ASSERT(nQMax < nMaxTerms);
    Q[nQMax].nTermsForRssDelta = nTerms;
    Q[nQMax].RssDelta = max(Q[iTerm].RssDelta, RssDelta);
    nQMax++;
    if (Sort) {
        memcpy(SortedQ, Q, nQMax * sizeof(tQueue));
        qsort(SortedQ, nQMax, sizeof(tQueue), CompareQ);         // sort on RssDelta
        if (FastBeta != 0) {
            for (int iRank = 0; iRank < nQMax; iRank++)
                SortedQ[iRank].AgedRank =
                    iRank + FastBeta * (nTerms - SortedQ[iRank].nTermsForRssDelta);
            qsort(SortedQ, nQMax, sizeof(tQueue), CompareAgedQ); // sort on aged rank
        }
    }
}

static void UpdateRssDeltaInQ(const int iParent, const int nTermsForRssDelta,
                              const double RssDelta)
{
    ASSERT(iParent == Q[iParent].iParent);
    ASSERT(iParent < nQMax);
    Q[iParent].nTermsForRssDelta = nTermsForRssDelta;
    Q[iParent].RssDelta = RssDelta;
}

static int GetNextParent(   // returns -1 if no more parents
    const bool InitFlag,    // use true to init, thereafter false
    const int  nFastK)
{
    static int iQ;          // index into sorted queue
    int iParent = -1;
    if (InitFlag) {
        if (TraceGlobal == 6)
            printf("\n|Considering parents ");
        iQ = 0;
    } else {
        if (iQ < min(nQMax, nFastK)) {
            iParent = SortedQ[iQ].iParent;
            iQ++;
        }
        if (TraceGlobal == 6 && iParent >= 0)
            printf("%d [%g] ", iParent+IOFFSET, SortedQ[iQ].RssDelta);
    }
    return iParent;
}

#endif // FAST_MARS

//-----------------------------------------------------------------------------
// Order() gets the sort indices of vector x, so x[sorted[i]] <= x[sorted[i+1]].
// Ties may be reordered. The returned indices are 0 based (as in C not as in R).
//
// This function is similar to the R library function rsort_with_index(),
// but is defined here to minimize R dependencies.
// Informal tests show that this is faster than rsort_with_index().

static const double *pxGlobal;

static int Compare(const void *p1, const void *p2)  // for qsort
{
    const int i1 = *(int *)p1;
    const int i2 = *(int *)p2;
    double Diff = pxGlobal[i1] - pxGlobal[i2];
    if (Diff < 0)
        return -1;
    else if (Diff > 0)
        return 1;
    else
        return 0;
}

static void Order(int sorted[],                     // out: vector with nx elements
                  const double x[], const int nx)   // in: x is a vector with nx elems
{
    for (int i = 0; i < nx; i++)
        sorted[i] = i;
    pxGlobal = x;
    qsort(sorted, nx, sizeof(int), Compare);
}


//-----------------------------------------------------------------------------
// Get order indices for an x array of dimensions nRows x nCols.
//
// Returns an nRows x nCols integer array of indices, where each column
// corresponds to a column of x.  See Order() for ordering details.
//
// Caller must free the returned array.

static int *OrderArray(const double x[], const int nRows, const int nCols)
{
    int *xOrder = (int *)malloc1(nRows * nCols * sizeof(int),
                            "xOrder\t\tnRows %d nCols %d sizeof(int) %d",
                            nRows, nCols, sizeof(int));

    for (int iCol = 0; iCol < nCols; iCol++) {
        Order(xOrder + iCol*nRows, x + iCol*nRows, nRows);
#if USING_R
        if (nRows > 10000)
            ServiceR();
#endif
    }
    return xOrder;
}

//-----------------------------------------------------------------------------
// return the number of TRUEs in the boolean vector UsedCols

static int GetNbrUsedCols(const bool UsedCols[], const int nLen)
{
    int nTrue = 0;

    for (int iCol = 0; iCol < nLen; iCol++)
        if (UsedCols[iCol])
            nTrue++;

    return nTrue;
}

//-----------------------------------------------------------------------------
// Copy used columns in x to *pxUsed and return the number of used columns
// UsedCols[i] is true for each each used column index in x
// Caller must free *pxUsed

static int CopyUsedCols(double **pxUsed,                // out
                    const double x[],                   // in: nCases x nCols
                    const int nCases, const int nCols,  // in
                    const bool UsedCols[])              // in
{
    int nUsedCols = GetNbrUsedCols(UsedCols, nCols);
    double *xUsed = (double *)malloc1(nCases * nUsedCols * sizeof(double),
                        "xUsed\t\t\tnCases %d nUsedCols %d sizeof(double) %d",
                        nCases, nUsedCols, sizeof(double));

    int iUsed = 0;
    for (int iCol = 0; iCol < nCols; iCol++)
        if (UsedCols[iCol]) {
            memcpy(xUsed + iUsed * nCases,
                x + iCol * nCases, nCases * sizeof(double));
            iUsed++;
        }
    *pxUsed = xUsed;
    return nUsedCols;
}

//-----------------------------------------------------------------------------
// Print a summary of the model, for debug tracing

#if STANDALONE
static void PrintSummary(
    const int    nMaxTerms,         // in
    const int    nTerms,            // in: number of cols in bx, some may be unused
    const int    nPreds,            // in: number of predictors
    const int    nResp,             // in: number of cols in y
    const bool   UsedCols[],        // in: specifies used colums in bx
    const int    Dirs[],            // in
    const double Cuts[],            // in
    const double Betas[],           // in: if NULL will print zeroes
    const int    nFactorsInTerm[])  // in: number of hinge funcs in basis term
{
    printf("   nFacs       Beta\n");

    int nUsedCols = GetNbrUsedCols(UsedCols, nTerms);
    int iUsed = -1;
    for (int iTerm = 0; iTerm < nTerms; iTerm++) {
        if (UsedCols[iTerm]) {
            iUsed++;
            printf("%2.2d  %2d    ", iTerm, nFactorsInTerm[iTerm]);
            for (int iResp = 0; iResp < nResp; iResp++)
                printf("%9.3g ", (Betas? Betas_(iUsed, iResp): 0));
            printf("| ");
            }
        else {
            printf("%2.2d  --    ", iTerm);
            for (int iResp = 0; iResp < nResp; iResp++)
                printf("%9s ", "--");
            printf("| ");
        }
        int iPred;
        for (iPred = 0; iPred < nPreds; iPred++)
            if (Dirs_(iTerm,iPred) == 0)
                printf(" . ");
            else
                printf("%2d ", Dirs_(iTerm,iPred));

        printf("|");

        for (iPred = 0; iPred < nPreds; iPred++)
            if (Dirs_(iTerm,iPred) == 0)
                printf("    .    ");
            else if (Dirs_(iTerm,iPred) == 2)
                printf("  linear ");
            else
                printf("%8.3g ", Cuts_(iTerm,iPred));

        printf("\n");
    }
    printf("\n");
}
#endif // STANDALONE

//-----------------------------------------------------------------------------
// Set Diags to the diagonal values of inverse(X'X),
// where X is referenced via the matrix R, from a previous call to dqrsl
// with (in practice) bx.  The net result is that Diags is the diagonal
// values of inverse(bx'bx).  We assume that R is created from a full rank X.
//
// TODO This could be simplified

static void CalcDiags(
    double Diags[],     // out: nCols x 1
    const double R[],   // in: nCases x nCols, QR from prev call to dqrsl
    const int nCases,   // in
    const int nCols)    // in
{
    #define R_(i,j)     R [(i) + (j) * nCases]
    #define R1_(i,j)    R1[(i) + (j) * nCols]
    #define B_(i,j)     B [(i) + (j) * nCols]

    double *R1 = (double *)malloc1(nCols * nCols * sizeof(double),  // nCols rows of R
                            "R1\t\t\tnCols %d nCols %d sizeof(double) %d",
                            nCols, nCols, sizeof(double));

    double *B =  (double *)calloc1(nCols * nCols, sizeof(double),   // rhs of R1 * x = B
                            "B\t\t\tnCols %d nCols %d sizeof(double) %d",
                            nCols, nCols, sizeof(double));
    int i, j;
    for (i = 0; i < nCols; i++) {   // copy nCols rows of R into R1
        for (j =  0; j < nCols; j++)
            R1_(i,j) = R_(i,j);
        B_(i,i) = 1;                // set diag of B to 1
    }
    int job = 1;            // 1 means solve R1 * x = B where R1 is upper triangular
    int info = 0;
    for (i = 0; i < nCols; i++) {
        dtrsl_(             // LINPACK function
            R1,             // in: t, matrix of the system, untouched
            (int *)&nCols,  // in: ldt (typecast discards const)
            (int *)&nCols,  // in: n
            &B_(0,i),       // io: b, on return has solution x
            &job,           // in:
            &info);         // io:

        ASSERT(info == 0);
    }
    // B is now inverse(R1).  Calculate B x B.

    for (i = 0; i < nCols; i++)
        for (j =  0; j < nCols; j++) {
            double Sum = 0;
            for (int k = max(i,j); k < nCols; k++)
                Sum += B_(i,k) * B_(j,k);
            B_(i,j) = B_(j,i) = Sum;
        }
    for (i = 0; i < nCols; i++)
         Diags[i] = B_(i,i);
    free1(B);
    free1(R1);
}

//-----------------------------------------------------------------------------
// Regress y on the used columns of x, in the standard way (using QR).
// UsedCols[i] is true for each each used col i in x; unused cols are ignored.
//
// The returned Betas argument is computed from, and is indexed on,
// the compacted x vector, not on the original x.
//
// The returned iPivots should only be used if *pnRank != nUsedCols.
// The entries of iPivots refer to columns in the full x (and are 0 based).
// Entries in iPivots at *pnRank and above specify linearly dependent columns in x.
//
// To maximize compatibility we call the same routines as the R function lm.

static void Regress(
    double       Betas[],       // out: nUsedCols * nResp, can be NULL
    double       Residuals[],   // out: nCases * nResp, can be NULL
    double       *pRss,         // out: RSS, summed over all nResp, can be NULL
    double       Diags[],       // out: diags of inv(transpose(x) * x), can be NULL
    int          *pnRank,       // out: nbr of indep cols in x
    int          iPivots[],     // out: nCols, can be NULL
    const double x[],           // in: nCases x nCols, must include intercept
    const double y[],           // in: nCases x nResp
    const double Weights[],     // in: nCases x 1, can be NULL
    const int    nCases,        // in: number of rows in x and in y
    const int    nResp,         // in: number of cols in y
    int          nCols,         // in: number of columns in x, some may not be used
    const bool   UsedCols[])    // in: specifies used columns in x
{
    double *xUsed;
    int nUsedCols = CopyUsedCols(&xUsed, x, nCases, nCols, UsedCols);

    bool MustFreeBetas = false;
    if (Betas == NULL) {
        Betas = (double *)malloc1(nUsedCols * nResp * sizeof(double),
                            "Betas\t\t\tnUsedCols %d nResp %d sizeof(double) %d",
                            nUsedCols, nResp, sizeof(double));
        MustFreeBetas = true;
    }
    bool MustFreeResiduals = false;
    if (Residuals == NULL) {
        Residuals = (double *)malloc1(nCases * nResp * sizeof(double),
                                "Residuals\t\tnCases %d nResp %d sizeof(double) %d",
                                nCases, nResp, sizeof(double));
        MustFreeResiduals = true;
    }
    bool MustFreePivots = false;
    if (iPivots == NULL) {
        iPivots = (int *)malloc1(nUsedCols * sizeof(int),
                            "iPivots\t\tnUsedCols %d sizeof(int) %d",
                            nUsedCols, sizeof(int));
        MustFreePivots = true;
    }
    int iCol;
    for (iCol = 0; iCol < nUsedCols; iCol++)
        iPivots[iCol] = iCol+1;

    // apply weights to x and y if Weights is not NULL

    double *wx = xUsed;
    double *wy = (double *)y;       // cast discards "const" else compiler warning
    double *Weightss = NULL;        // sqrt of Weights
    if (Weights) {
        // wx is xUsed but with each element multiplied by the sqrt of
        // the corresponding element of Weights.  Ditto for wy.

        int iCase;
        Weightss = (double *)malloc1(nCases * sizeof(double),
                                "Weightss\t\t\tnCases %d sizeof(double) %d",
                                nCases, sizeof(double));

        for (iCase = 0; iCase < nCases; iCase++)
            Weightss[iCase] = sqrt(Weights[iCase]);

        wx = (double *)malloc1(nCases * nUsedCols * sizeof(double),
                        "wx\t\t\tnCases %d nUsedCols %d sizeof(double) %d",
                        nCases, nUsedCols, sizeof(double));

        wy = (double *)malloc1(nCases * nResp * sizeof(double),
                        "wy\t\t\tnCases %d nResp %d sizeof(double) %d",
                        nCases, nResp, sizeof(double));

        for (iCase = 0; iCase < nCases; iCase++)
            for (int iCol = 0; iCol < nUsedCols; iCol++)
                wx[iCase + iCol * nCases] =
                    Weightss[iCase] * xUsed[iCase + iCol * nCases];

        for (iCase = 0; iCase < nCases; iCase++)
            for (int iResp = 0; iResp < nResp; iResp++)
                wy[iCase + iResp * nCases] =
                    Weightss[iCase] * y[iCase + iResp * nCases];
    }
    // compute Betas and yHat (use Residuals as a temporary buffer to store yHat)

    double *qraux = (double *)malloc1(nUsedCols * sizeof(double),
                                "qraux\t\t\tnUsedCols %d sizeof(double) %d",
                                nUsedCols, sizeof(double));

    double *work = (double *)malloc1(nCases * nUsedCols * sizeof(double),
                                "work\t\t\tnCases %d nUsedCols %d sizeof(double) %d",
                                nCases, nUsedCols, sizeof(double));

    dqrdc2_(                // R function, QR decomp based on LINPACK dqrdc
        wx,                 // io:  x, on return upper tri of x is R of QR
        (int *)&nCases,     // in:  ldx (typecast discards const)
        (int *)&nCases,     // in:  n
        &nUsedCols,         // in:  p
        (double*)&QR_TOL,   // in:  tol
        pnRank,             // out: k, num of indep cols of x
        qraux,              // out: qraux
        iPivots,            // out: jpvt
        work);              // work

    double Rss = 0;
    int job = 101;          // specify c=1 e=1 to compute qty, b, yHat
    int info;
    for (int iResp = 0; iResp < nResp; iResp++) {
        dqrsl_(                                 // LINPACK function
            wx,                                 // in:  x, generated by dqrdc2
            (int *)&nCases,                     // in:  ldx (typecast discards const)
            (int *)&nCases,                     // in:  n
            pnRank,                             // in:  k
            qraux,                              // in:  qraux
            (double *)(wy + iResp * nCases),    // in:  y
            work,                               // out: qy, unused here
            work,                               // out: qty, unused here
            (double *)(&Betas_(0,iResp)),       // out: b
            work,                               // out: rsd, unused here
            (double *)(&Residuals_(0,iResp)),   // out: xb = yHat = ls approx of x*b
            &job,                               // in:  job
            &info);                             // in:  info

        ASSERT(info == 0);

        // compute Residuals and Rss (sum over all responses)

        if (Weightss)
            for (int iCase = 0; iCase < nCases; iCase++) {
                Residuals_(iCase,iResp) /= Weightss[iCase];
                Residuals_(iCase,iResp) = (y_(iCase,iResp) - Residuals_(iCase,iResp));
                Rss += sq(Residuals_(iCase, iResp));
            }
        else
            for (int iCase = 0; iCase < nCases; iCase++) {
                Residuals_(iCase,iResp) = (y_(iCase,iResp) - Residuals_(iCase,iResp));
                Rss += sq(Residuals_(iCase, iResp));
            }
    }
    if (pRss)
        *pRss = Rss;

    if (*pnRank != nUsedCols) {
        // adjust iPivots for missing cols in UsedCols and for 1 offset

        int *PivotOffset = (int *)malloc1(nCols * sizeof(int),
                                    "PivotOffset\t\t\tnCols %d sizeof(int) %d",
                                    nCols, sizeof(int));
        int nOffset = 0, iOld = 0;
        for (iCol = 0; iCol < nCols; iCol++) {
            if (!UsedCols[iCol])
                nOffset++;
            else {
                PivotOffset[iOld] = nOffset;
                if (++iOld > nUsedCols)
                    break;
            }
        }
        for (iCol = 0; iCol < nUsedCols; iCol++)
            iPivots[iCol] = iPivots[iCol] - 1 + PivotOffset[iPivots[iCol] - 1];
        free1(PivotOffset);
    }
    if (Diags)
        CalcDiags(Diags, wx, nCases, nUsedCols);
    if (MustFreePivots)
        free1(iPivots);
    if (MustFreeResiduals)
        free1(Residuals);
    if (MustFreeBetas)
        free1(Betas);
    if (Weightss) {
        free1(Weightss);
        free1(wx);
        free1(wy);
    }
    free1(xUsed);
    free1(qraux);
    free1(work);
}

//-----------------------------------------------------------------------------
// This routine is for testing Regress from R, to compare results to R's lm().

#if USING_R
void RegressR(
    double       Betas[],       // out: (nUsedCols+1) * nResp, +1 is for intercept
    double       Residuals[],   // out: nCases * nResp
    double       Rss[],         // out: RSS, summed over all nResp
    double       Diags[],       // out: diags of inv(transpose(x) * x)
    int          *pnRank,       // out: nbr of indep cols in x
    int          iPivots[],     // out: nCols, can be R_NilValue
    const double x[],           // in: nCases x nCols
    const double y[],           // in: nCases x nResp
    const double Weights[],     // in: nCases x 1, can be R_NilValue
    const int    *pnCases,      // in: number of rows in x and in y
    const int    *pnResp,       // in: number of cols in y
    int          *pnCols,       // in: number of columns in x, some may not be used
    const bool   UsedCols[])    // in: specifies used columns in x
{
    if ((void *)Weights == (void *)R_NilValue)
        Weights = NULL;

    Regress(Betas, Residuals, Rss, Diags, pnRank, iPivots,
        x, y, Weights, *pnCases, *pnResp, *pnCols, UsedCols);
}
#endif

//-----------------------------------------------------------------------------
// Regress y on bx to get Residuals and Betas.  If bx isn't of full rank,
// remove dependent cols, update UsedCols, and regress again on the bx with
// removed cols.

static void RegressAndFix(
    double Betas[],         // out: nMaxTerms x nResp, can be NULL
    double Residuals[],     // out: nCases x nResp, can be NULL
    double Diags[],         // out: if !NULL set to diags of inv(transpose(bx) * bx)
    bool   UsedCols[],      // io:  will remove cols if necessary, nMaxTerms x 1
    const  double bx[],     // in:  nCases x nMaxTerms
    const double y[],       // in:  nCases x nResp
    const double Weights[], // in: nCases x 1, can be NULL
    const int nCases,       // in
    const int nResp,        // in: number of cols in y
    const int nTerms)       // in: number of cols in bx, some may not be used
{
    int nRank;
    int *iPivots = (int *)malloc1(nTerms * sizeof(int),
                            "iPivots\t\tnTerms %d sizeof(int) %d",
                            nTerms, sizeof(int));
    Regress(Betas, Residuals, NULL, Diags, &nRank, iPivots,
        bx, y, Weights, nCases, nResp, nTerms, UsedCols);
    int nUsedCols = GetNbrUsedCols(UsedCols, nTerms);
    int nDeficient = nUsedCols - nRank;
    if (nDeficient) {           // rank deficient?
        // Remove linearly dependent columns.
        // The lin dep columns are at index nRank and higher in iPivots.

        for (int iCol = nRank; iCol < nUsedCols; iCol++)
            UsedCols[iPivots[iCol]] = false;

        Regress(Betas, Residuals, NULL, Diags, &nRank, NULL,
            bx, y, Weights, nCases, nResp, nTerms, UsedCols);
        nUsedCols = nUsedCols - nDeficient;
        if (nRank != nUsedCols)
            warning("Could not fix rank deficient bx: nUsedCols %d nRank %d",
                nUsedCols,  nRank);
        else if (TraceGlobal >= 1)
            printf("Fixed rank deficient bx by removing %d term%s, %d term%s remain%s\n",
                nDeficient, ((nDeficient==1)? "": "s"),
                nUsedCols,  ((nUsedCols==1)? "": "s"), ((nUsedCols==1)? "s": ""));
    }
    free1(iPivots);
}

//-----------------------------------------------------------------------------
static INLINE double Mean(const double x[], int n)
{
    double mean = 0;
    for (int i = 0; i < n; i++)
        mean += x[i] / n;
    return mean;
}

//-----------------------------------------------------------------------------
// get mean centered sum of squares

static INLINE double SumOfSquares(const double x[], const double mean, int n)
{
    double ss = 0;
    for (int i = 0; i < n; i++)
        ss += sq(x[i] - mean);
    return ss;
}

//-----------------------------------------------------------------------------
static INLINE double GetGcv(const int nTerms, // nbr basis terms including intercept
                const int nCases, double Rss, const double Penalty)
{
    double cost;
    if (Penalty == -1)  // special case: terms and knots are free
        cost = 0;
    else {
        const double nKnots = ((double)nTerms-1) / 2;
        cost = (nTerms + Penalty * nKnots) / nCases;
    }
    // test against cost ensures that GCVs are non-decreasing as nbr of terms increases
    return cost >= 1? POS_INF : Rss / (nCases * sq(1 - cost));
}

//-----------------------------------------------------------------------------
// We only consider knots that are nMinSpan distance apart, to increase resistance
// to runs of correlated noise.  This function determines that distance.
// It implements eqn 43 FriedmanMars (see refs), but with an extension for nMinSpan.
// If bx==NULL then instead of counting valid entries in bx we use nCases,
// and ignore the term index iTerm.
//
// nMinSpan: if =0, use internally calculated min span
//           if >0, use instead of internally calculated min span
//           if <0, use old (incorrect) method of calculating minspan
//                  this was the method used prior to earth 2.4-0

static INLINE int GetMinSpan(int nCases, int nPreds, const double *bx,
                             const int iTerm)
{
    if (nMinSpanGlobal > 0)                     // user specified a fixed span?
        return nMinSpanGlobal;

    int nUsed = 0;                              // Nm in Friedman's notation
    if (bx == NULL)
        nUsed = nCases;
    else for (int iCase = 0; iCase < nCases; iCase++)
        if (bx_(iCase,iTerm) > 0)
            nUsed++;

    static const double temp1 = 2.9702;         // -log(-log(0.95)
    static const double temp2 = 1.7329;         // 2.5 * log(2)
    const double n = (nMinSpanGlobal < 0) ? nCases: nPreds; // CHANGED earth 2.4
    return (int)((temp1 + log(n * nUsed)) / temp2);
}

//-----------------------------------------------------------------------------
// We don't consider knots that are too close to the ends.
// This function determines how close to an end we can get.
// It implements eqn 45 FriedmanMars (see refs), re-expressed
// for efficient computation

static INLINE int GetEndSpan(const int nCases, const int nPreds)
{
    static const double log_2 = 0.69315;            // log(2)
    static const double temp1 = 7.32193;            // 3 + log(20)/log(2);
    const double n = (nMinSpanGlobal < 0) ? nCases: nPreds; // CHANGED earth 2.4
    return (int)(temp1 + log(n) / log_2);
}

//-----------------------------------------------------------------------------
// Return true if model term type is not already in model
// i.e. if the hockey stick functions in a pre-existing term do not use the same
// predictors (ignoring knot values).
//
// In practice this nearly always returns true.

static bool GetNewFormFlag(const int iPred, const int iTerm,
                        const int Dirs[], const bool UsedCols[],
                        const int nTerms, const int nPreds, const int nMaxTerms)
{
    bool IsNewForm = true;
    for (int iTerm1 = 1; iTerm1 < nTerms; iTerm1++) // start at 1 to skip intercept
        if (UsedCols[iTerm1]) {
            IsNewForm = false;
            if (Dirs_(iTerm1,iPred) == 0)
                return true;
            for (int iPred1 = 0; iPred1 < nPreds; iPred1++)
                if (iPred1 != iPred && Dirs_(iTerm1,iPred1) != Dirs_(iTerm,iPred1))
                    return true;
        }
    return IsNewForm;
}

//-----------------------------------------------------------------------------
static double GetCut(int iCase, const int iPred, const int nCases,
                        const double x[], const int xOrder[])
{
    if (iCase < 0 || iCase >= nCases)
        error("GetCut iCase %d: iCase < 0 || iCase >= nCases", iCase);
    const int ix = xOrder_(iCase,iPred);
    ASSERT(ix >= 0 && ix < nCases);
    return x_(ix,iPred);
}

//-----------------------------------------------------------------------------
// The BetaCache is used when searching for a new term pair, via FindTerm.
// Most of the calculation for the orthogonal regression betas is repeated
// with the same data, and thus we can save time by caching betas.
// (The "Betas" are the regression coefficients.)
//
// iParent    is the term that forms the base for the new term
// iPred      is the predictor for the new term
// iOrthCol   is the column index in the bxOrth matrix

static double *BetaCacheGlobal; // [iOrthCol,iParent,iPred]
                                // dim nPreds x nMaxTerms x nMaxTerms

static void InitBetaCache(const bool UseBetaCache,
                          const int nMaxTerms, const int nPreds)
{
    int nCache =  nMaxTerms * nMaxTerms * nPreds;
    if (!UseBetaCache) {
        BetaCacheGlobal = NULL;
    // 3e9 below is somewhat arbitrary but seems about right (in 2011)
    } else if (nCache * sizeof(double) > 3e9) {
            printf(
"\nNote: earth's beta cache would require %s, so forcing Use.beta.cache=FALSE.\n"
"      Invoke earth with Use.beta.cache=FALSE to make this message go away.\n\n",
                sFormatMemSize(nCache * sizeof(double), false));
            fflush(stdout);
            BetaCacheGlobal = NULL;
    } else {
       if (TraceGlobal >= 5)    // print cache size
            printf("BetaCache %s\n",
                sFormatMemSize(nCache * sizeof(double), false));

        BetaCacheGlobal = (double *)malloc1(nCache * sizeof(double),
            "BetaCacheGlobal\tnMaxTerms %d nMaxTerms %d nPreds %d sizeof(double) %d",
            nMaxTerms, nMaxTerms, nPreds, sizeof(double));

        for (int i = 0; i < nCache; i++)    // mark all entries as uninited
            BetaCacheGlobal[i] = POS_INF;
    }
}

static void FreeBetaCache(void)
{
    if (BetaCacheGlobal)
        free1(BetaCacheGlobal);
}

//-----------------------------------------------------------------------------
// Init a new bxOrthCol to the residuals from regressing y on the used columns
// of the orthogonal matrix bxOrth.  The length (i.e. sum of sqaures divided
// by nCases) of each column of bxOrth must be 1 with mean 0 (except the
// first column which is the intercept).
//
// In practice this function is called with the params shown in {braces}
// and is called only by InitBxOrthCol.
//
// This function must be fast.
//
// In calculation of Beta, we used to have
//     xty += pbxOrth[iCase] * y[iCase];
// and now we have
//    xty += pbxOrth[iCase] * bxOrthCol[iCase];
// i.e. we use the "modified" instead of the "classic" Gram Schmidt.
// This is supposedly less susceptible to round off errors, although I haven't
// seen it have any effect on any of the data sets we have tested.

static INLINE void OrthogResiduals(
    double bxOrthCol[],     // out: nCases x 1      { bxOrth[,nTerms] }
    const double y[],       // in:  nCases x nResp  { bx[,nTerms], xbx }
    const double bxOrth[],  // in:  nTerms x nPreds { bxOrth }
    const int nCases,       // in
    const int nTerms,       // in: nTerms in model, i.e. number of used cols in bxOrth
    const bool UsedTerms[], // in: UsedTerms[i] is true if col is used, unused cols ignored
                            //     Following parameters are only for the beta cache
    const int iParent,      // in: if >= 0, use BetaCacheGlobal {FindTerm iTerm, addTermP -1}
    const int iPred,        // in: predictor index i.e. col index in input matrix x
    const int nMaxTerms)    // in:
{
    double *pCache;
    if (iParent >= 0 && BetaCacheGlobal)
        pCache = BetaCacheGlobal + iParent*nMaxTerms + iPred*sq(nMaxTerms);
    else
        pCache = NULL;

    memcpy(bxOrthCol, y, nCases * sizeof(double));

    for (int iTerm = 0; iTerm < nTerms; iTerm++)
        if (UsedTerms[iTerm]) {
            const double *pbxOrth = &bxOrth_(0, iTerm);
            double Beta;
            if (pCache && pCache[iTerm] != POS_INF)
                Beta = pCache[iTerm];
            else {
                double xty = 0;
                for (int iCase = 0; iCase < nCases; iCase++)
                    xty += pbxOrth[iCase] * bxOrthCol[iCase]; // see header comment
                Beta = xty;  // no need to divide by xtx, it is 1
                ASSERT(FINITE(Beta));
                if (pCache)
                    pCache[iTerm] = Beta;
            }
#if USE_BLAS
            const double NegBeta = -Beta;
            daxpy_(&nCases, &NegBeta, pbxOrth, &ONE, bxOrthCol, &ONE);
#else
            for (int iCase = 0; iCase < nCases; iCase++)
                bxOrthCol[iCase] -= Beta * pbxOrth[iCase];
#endif
        }
}

//-----------------------------------------------------------------------------
// Init the rightmost column of bxOrth i.e. the column indexed by nTerms.
// The new col is the normalized residuals from regressing y on the
// lower (i.e. already existing) cols of bxOrth.
// Also updates bxOrthCenteredT and bxOrthMean.
//
// In practice this function is called only with the params shown in {braces}

static INLINE void InitBxOrthCol(
    double bxOrth[],         // io: col nTerms is changed, other cols not touched
    double bxOrthCenteredT[],// io: kept in sync with bxOrth
    double bxOrthMean[],     // io: element at nTerms is updated
    bool   *pGoodCol,        // io: set to false if col sum-of-squares is under BX_TOL
    const double *y,         // in: { AddCandLinTerm xbx, addTermPair bx[,nTerms] }
    const int nTerms,        // in: column goes in at index nTerms, 0 is the intercept
    const bool WorkingSet[], // in
    const int nCases,        // in
    const int nMaxTerms,     // in
    const int iCacheTerm,    // in: if >= 0, use BetaCacheGlobal {FindTerm iTerm, AddTermP -1}
                             //     if < 0 then recalc Betas from scratch
    const int iPred,         // in: predictor index i.e. col index in input matrix x
    const double Weights[])  // in:
{
    int iCase;
    *pGoodCol = true;
    Weights = Weights; // prevent compiler warning: unused parameter 'Weights'

    if (nTerms == 0) {          // column 0, the intercept
        double len = 1 / sqrt((double) nCases);
        bxOrthMean[0] = len;
        for (iCase = 0; iCase < nCases; iCase++)
            bxOrth_(iCase,0) = len;
    } else if (nTerms == 1) {   // column 1, the first basis function
        double yMean = Mean(y, nCases);
        for (iCase = 0; iCase < nCases; iCase++)
            bxOrth_(iCase,1) = y[iCase] - yMean;
    } else
        OrthogResiduals(&bxOrth_(0,nTerms), // resids go in rightmost col of bxOrth at nTerms
            y, bxOrth, nCases, nTerms, WorkingSet, iCacheTerm, iPred, nMaxTerms);

    if (nTerms > 0) {
        // normalize the column to length 1 and init bxOrthMean[nTerms]

        double bxOrthSS = SumOfSquares(&bxOrth_(0,nTerms), 0, nCases);
        const double Tol = (iCacheTerm < 0? 0: BX_TOL);
        if (bxOrthSS > Tol) {
            bxOrthMean[nTerms] = Mean(&bxOrth_(0,nTerms), nCases);
            const double len = sqrt(bxOrthSS);
            for (iCase = 0; iCase < nCases; iCase++)
                bxOrth_(iCase,nTerms) /= len;
        } else {
            *pGoodCol = false;
            bxOrthMean[nTerms] = 0;
            memset(&bxOrth_(0,nTerms), 0, nCases * sizeof(double));
        }
    }
    for (iCase = 0; iCase < nCases; iCase++)        // keep bxOrthCenteredT in sync
        bxOrthCenteredT_(nTerms,iCase) = bxOrth_(iCase,nTerms) - bxOrthMean[nTerms];
}

//-----------------------------------------------------------------------------
// Add a new term pair to the arrays.
// Each term in the new term pair is a copy of an existing parent term but extended
// by multiplying it by a new hockey stick function at the selected knot.
// If the upper term in the term pair is invalid then we still add the upper
// term but mark it as false in FullSet.

static void AddTermPair(
    int    Dirs[],              // io
    double Cuts[],              // io
    double bx[],                // io: MARS basis matrix
    double bxOrth[],            // io
    double bxOrthCenteredT[],   // io
    double bxOrthMean[],        // io
    bool   FullSet[],           // io
    int    nFactorsInTerm[],    // io
    int    nUses[],             // io: nbr of times each predictor is used in the model
    const int nTerms,           // in: new term pair goes in at index nTerms and nTerms1
    const int iBestParent,      // in: parent term
    const int iBestCase,        // in
    const int iBestPred,        // in
    const int nPreds,           // in
    const int nCases,           // in
    const int nMaxTerms,        // in
    const bool IsNewForm,       // in
    const bool IsLinPred,       // in: pred was discovered by search to be linear
    const int LinPreds[],       // in: user specified preds which must enter linearly
    const double x[],           // in
    const int xOrder[],         // in
    const double Weights[])     // in:
{
    const double BestCut = GetCut(iBestCase, iBestPred, nCases, x, xOrder);
    ASSERT(IsLinPred || iBestCase != 0);
    const int nTerms1 = nTerms+1;

    // copy the parent term to the new term pair

    int iPred;
    bool PrintedParent = false;
    for (iPred = 0; iPred < nPreds; iPred++) {
        Dirs_(nTerms, iPred) =
        Dirs_(nTerms1,iPred) = Dirs_(iBestParent,iPred);

        Cuts_(nTerms, iPred) =
        Cuts_(nTerms1,iPred) = Cuts_(iBestParent,iPred);

        if (TraceGlobal >= 2 && !PrintedParent && Dirs_(iBestParent,iPred)) {
            // print parent term (this appends to prints by PrintForwardStep)
            printf("%-3d ", iBestParent+IOFFSET);
            PrintedParent = true;
        }
    }
    // incorporate the new hockey stick function

    nFactorsInTerm[nTerms]  =
    nFactorsInTerm[nTerms1] = nFactorsInTerm[iBestParent] + 1;

    int DirEntry = 1;
    if (LinPreds[iBestPred]) {
        ASSERT(IsLinPred);
        DirEntry = 2;
    }
    Dirs_(nTerms, iBestPred) = DirEntry;
    Dirs_(nTerms1,iBestPred) = -1; // will be ignored if adding only one hinge

    Cuts_(nTerms, iBestPred) =
    Cuts_(nTerms1,iBestPred) = BestCut;

    FullSet[nTerms] = true;
    if (!IsLinPred && IsNewForm)
        FullSet[nTerms1] = true;

    // If the term is not valid, then we don't wan't to use it as the base for
    // a new term later (in FindTerm).  Enforce this by setting
    // nFactorsInTerm to a value greater than any posssible nMaxDegree.

    if (!FullSet[nTerms1])
        nFactorsInTerm[nTerms1] = MAX_DEGREE + 1;

    // fill in new columns of bx, at nTerms and nTerms+1 (left and right hinges)

    int iCase;
    if (DirEntry == 2) {
        for (iCase = 0; iCase < nCases; iCase++)
            bx_(iCase,nTerms) = bx_(iCase,iBestParent) * x_(iCase,iBestPred);
    } else for (iCase = 0; iCase < nCases; iCase++) {
        if (x_(iCase,iBestPred) - BestCut > 0)
            bx_(iCase,nTerms) =
                bx_(iCase,iBestParent) * (x_(iCase,iBestPred) - BestCut);
        else
            bx_(iCase,nTerms1) =
                bx_(iCase,iBestParent) * (BestCut - x_(iCase,iBestPred));
    }
    nUses[iBestPred]++;

    // init the col in bxOrth at nTerms and init bxOrthMean[nTerms]

    bool GoodCol;
    InitBxOrthCol(bxOrth, bxOrthCenteredT, bxOrthMean, &GoodCol,
        &bx_(0,nTerms), nTerms, FullSet, nCases, nMaxTerms, -1, nPreds, Weights);
                            // -1 means don't use BetaCacheGlobal, calc Betas afresh

    // init the col in bxOrth at nTerms1 and init bxOrthMean[nTerms1]

    if (FullSet[nTerms1]) {
        InitBxOrthCol(bxOrth, bxOrthCenteredT, bxOrthMean, &GoodCol,
            &bx_(0,nTerms1), nTerms1, FullSet, nCases, nMaxTerms, -1, iPred, Weights);
    } else {
        memset(&bxOrth_(0,nTerms1), 0, nCases * sizeof(double));
        bxOrthMean[nTerms1] = 0;
        for (iCase = 0; iCase < nCases; iCase++)    // keep bxOrthCenteredT in sync
            bxOrthCenteredT_(nTerms1,iCase) = 0;
    }
}

//-----------------------------------------------------------------------------
// The caller has selected a candidate predictor iPred and a candidate iParent.
// This function now selects a knot.  If it finds a knot it will
// update *piBestCase and pRssDeltaForParentPredPair.
//
// The general idea: scan backwards through all (ordered) values (i.e. potential
// knots) for the given predictor iPred, calculating RssDelta.
// If RssDelta > *pRssDeltaForParentPredPair (and all else is ok), then
// select the knot (by updating *piBestCase and *pRssDeltaForParentPredPair).
//
// There are currently nTerms in the model. We want to add a term pair
// at index nTerms and nTerms+1.
//
// This function must be fast.
//
// A note on the iSpan variable. We used to have
//     if (iCase % nMinSpan == 0)
// now we have
//     if (iSpan-- == 1)
// which is measurably faster, at least on a Pentium D.
// When we init iSpan (before the loop) we have a bit of code to
// initialize to an offset that puts the knots as the same positions as
// previous versions of earth.

static INLINE void FindKnot(
    int    *piBestCase,         // out: possibly updated, index into the ORDERED x's
    double *pRssDeltaForParentPredPair, // io: updated if knot is better
    double CovCol[],            // scratch buffer, overwritten, nTerms x 1
    double ycboSum[],           // scratch buffer, overwritten, nMaxTerms x nResp
    double CovSx[],             // scratch buffer, overwritten, nTerms x 1
    double *ybxSum,             // scratch buffer, overwritten, nResp x 1
    const int nTerms,           // in
    const int iParent,          // in: parent term
    const int iPred,            // in: predictor index
    const int nCases,           // in
    const int nResp,            // in: number of cols in y
    const int nMaxTerms,        // in
    const double RssDeltaLin,   // in: change in RSS if predictor iPred enters linearly
    const double MaxAllowedRssDelta, // in: FindKnot rejects any changes in Rss greater than this
    const double bx[],          // in: MARS basis matrix
    const double bxOrth[],      // in
    const double bxOrthCenteredT[], // in
    const double bxOrthMean[],  // in
    const double x[],           // in: nCases x nPreds
    const double y[],           // in: nCases x nResp
    const double Weights[],     // in: nCases x 1, must not be NULL
    const int xOrder[],         // in
    const double yMean[],       // in: vector nResp x 1
    const int nMinSpan,
    const int nEndSpan,
    const double NewVarAdjust)  // in: 1 if not a new var, 1+NewVarPenalty if new var
{
    Weights = Weights; // prevent compiler warning: unused parameter 'Weights'
    ASSERT(MaxAllowedRssDelta > 0);
#if USE_BLAS
    double Dummy = bxOrth[0];   // prevent compiler warning: unused parameter
    Dummy = bxOrthMean[0];
#else
    double Dummy = bxOrthCenteredT[0];
    Dummy = nMaxTerms;
#endif
    const int nCases_nEndSpan = nCases - nEndSpan;
    ASSERT(nMinSpan > 0);
    int iSpan = (nCases - 1) % nMinSpan;
    if (iSpan == 0)
        iSpan = nMinSpan;

    int iResp;
    for (iResp = 0; iResp < nResp; iResp++)
        ycboSum_(nTerms, iResp) = 0;
    memset(CovCol, 0, (nTerms+1) * sizeof(double));
    memset(CovSx,  0, (nTerms+1) * sizeof(double));
    memset(ybxSum, 0, nResp * sizeof(double));
    double bxSum = 0, bxSqSum = 0, bxSqxSum = 0, bxxSum = 0, st = 0;

    for (int iCase = nCases - 2; iCase >= nEndSpan; iCase--) { // -2 allows for ix1
        // may Mars have mercy on the poor soul who enters here

        const int    ix0 = xOrder_(iCase,  iPred); // get the x's in descending order
        const double x0  = x_(ix0,iPred);
        const int    ix1 = xOrder_(iCase+1,iPred);
        const double x1  = x_(ix1,iPred);
        const double bx1 = bx_(ix1,iParent);
        const double xDelta = x1 - x0;
        const double bxSq = sq(bx1);

#if USE_BLAS
        daxpy_(&nTerms, &bx1, &bxOrthCenteredT_(0,ix1), &ONE, CovSx,  &ONE);
        daxpy_(&nTerms, &xDelta, CovSx, &ONE, CovCol, &ONE);
#else
        int it;
        for (it = 0; it < nTerms; it++) {
            CovSx[it]  += (bxOrth_(ix1,it) - bxOrthMean[it]) * bx1;
            CovCol[it] += xDelta * CovSx[it];
        }
#endif
        bxSum    += bx1;
        bxSqSum  += bxSq;
        bxxSum   += bx1 * x1;
        bxSqxSum += bxSq * x1;
        const double su = st;
        st = bxxSum - bxSum * x0;

        CovCol[nTerms] += xDelta * (2 * bxSqxSum - bxSqSum * (x0 + x1)) +
                          (sq(su) - sq(st)) / nCases;

        if (nResp == 1) {    // treat nResp==1 as a special case, for speed
            ybxSum[0] += (y_(ix1, 0) - yMean[0]) * bx1;
            ycboSum_(nTerms, 0) += xDelta * ybxSum[0];
        } else for (iResp = 0; iResp < nResp; iResp++) {
            ybxSum[iResp] += (y_(ix1, iResp) - yMean[iResp]) * bx1;
            ycboSum_(nTerms, iResp) += xDelta * ybxSum[iResp];
        }
        if (iSpan-- == 1) {
            iSpan = nMinSpan;
            if (CovCol[nTerms] > 0) {
                // calculate RssDelta and see if this knot beats the previous best

                double RssDelta = RssDeltaLin;
                for (iResp = 0; iResp < nResp; iResp++) {
#if USE_BLAS
                    const double temp1 =
                        ycboSum_(nTerms,iResp) -
                        ddot_(&nTerms, &ycboSum_(0,iResp), &ONE, CovCol, &ONE);

                    const double temp2 =
                        CovCol[nTerms] - ddot_(&nTerms, CovCol, &ONE, CovCol, &ONE);
#else
                    double temp1 = ycboSum_(nTerms,iResp);
                    double temp2 = CovCol[nTerms];
                    int it;
                    for (it = 0; it < nTerms; it++) {
                        temp1 -= ycboSum_(it,iResp) * CovCol[it];
                        temp2 -= sq(CovCol[it]);
                    }
#endif
                    if (temp2 / CovCol[nTerms] > BX_TOL)
                        RssDelta += sq(temp1) / temp2;
                }
                RssDelta /= NewVarAdjust;

                // TODO HastieTibs code had an extra test here, seems unnecessary
                // !(iCase > 0 && x_(ix0,iPred) == x_(xOrder_(iCase-1,iPred),iPred))

                if (RssDelta > *pRssDeltaForParentPredPair &&
                        RssDelta < MaxAllowedRssDelta      &&
                        iCase < nCases_nEndSpan            &&
                        bx1 > 0) {
                    *piBestCase = iCase;
                    *pRssDeltaForParentPredPair = RssDelta;
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// Add a candidate term at bx[,nTerms], with the parent term multiplied by
// the predictor iPred entering linearly.  Do this by setting the knot at
// the lowest value xMin of x, since max(0,x-xMin)==x-xMin for all x.  The
// change in RSS caused by adding this term forms the base RSS delta which
// we will try to beat in the search in FindKnot.
//
// This also initializes CovCol, bxOrth[,nTerms], and ycboSum[nTerms,]

static INLINE void AddCandidateLinearTerm(
    double *pRssDeltaLin,       // out: change to RSS caused by adding new term
    bool   *pIsNewForm,         // io:
    double xbx[],               // out: nCases x 1
    double CovCol[],            // out: nMaxTerms x 1
    double ycboSum[],           // io: nMaxTerms x nResp
    double bxOrth[],            // io
    double bxOrthCenteredT[],   // io
    double bxOrthMean[],        // io
    const int iPred,            // in
    const int iParent,          // in
    const double x[],           // in: nCases x nPreds
    const double y[],           // in: nCases x nResp
    const double Weights[],     // in: nCases x 1, must not be NULL
    const int nCases,           // in
    const int nResp,            // in: number of cols in y
    const int nTerms,           // in
    const int nMaxTerms,        // in
    const double yMean[],       // in: vector nResp x 1
    const double bx[],          // in: MARS basis matrix
    const bool FullSet[],       // in
    const double NewVarAdjust)  // in
{
    // set xbx to x[,iPred] * bx[,iParent]

    int iCase;
    for (iCase = 0; iCase < nCases; iCase++)
        xbx[iCase] = x_(iCase,iPred) * bx_(iCase,iParent);

    // init bxOrth[,nTerms] and bxOrthMean[nTerms] for the candidate term
    // TODO look into *pIsNewForm handling here, it's confusing

    InitBxOrthCol(bxOrth, bxOrthCenteredT, bxOrthMean, pIsNewForm,
        xbx, nTerms, FullSet, nCases, nMaxTerms, iParent, iPred, Weights);

    // init CovCol and ycboSum[nTerms], for use by FindKnot later

    memset(CovCol, 0, (nTerms-1) * sizeof(double));
    CovCol[nTerms] = 1;
    int iResp;
    for (iResp = 0; iResp < nResp; iResp++) {
        ycboSum_(nTerms, iResp) = 0;
        for (iCase = 0; iCase < nCases; iCase++)
            ycboSum_(nTerms, iResp) += (y_(iCase, iResp) - yMean[iResp]) *
                                       bxOrth_(iCase,nTerms);
    }
    // calculate change to RSS caused by adding candidate new term

    *pRssDeltaLin = 0;
    for (iResp = 0; iResp < nResp; iResp++) {
        double yboSum = 0;
        for (iCase = 0; iCase < nCases; iCase++)
            yboSum += y_(iCase,iResp) * bxOrth_(iCase,nTerms);
        *pRssDeltaLin += sq(yboSum) / NewVarAdjust;
    }
    if (TraceGlobal >= 7)
        printf("Case %4d Cut % 12.4g< RssDelta %-12.5g ",
            0+IOFFSET, GetCut(0, iPred, nCases, x, xOrder), *pRssDeltaLin);
}

//-----------------------------------------------------------------------------
// The caller has selected a candidate parent term iParent.
// This function now selects a predictor, and a knot for that predictor.
//
// TODO These functions have a ridiculous number of parameters, I know.
//
// TODO A note on the comparison against ALMOST_ZERO below:
// It's not a clean solution but seems to work ok.
// It was added after we saw different results on different
// machines for certain datasets e.g. (tested on earth 1.4.0)
// ldose  <- rep(0:5, 2) - 2
// ldose1 <- c(0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 0.3, 1.4, 2.5, 3.6, 4.7, 5.8)
// sex3 <- factor(rep(c("male", "female", "andro"), times=c(6,4,2)))
// fac3 <- factor(c("lev2", "lev2", "lev1", "lev1", "lev3", "lev3",
//                  "lev2", "lev2", "lev1", "lev1", "lev3", "lev3"))
// numdead <- c(1,4,9,13,18,20,0,2,6,10,12,16)
// numdead2 <- c(2,3,10,13,19,20,0,3,7,11,13,17)
// pair <- cbind(numdead, numdead2)
// df <- data.frame(sex3, ldose, ldose1, fac3)
// am <-  earth(df, pair, trace=6, pmethod="none", degree=2)

static INLINE void FindPred(
    int    *piBestCase,         // out: return -1 if no new term available
                                //      else return an index into the ORDERED x's
    int    *piBestPred,         // out
    int    *piBestParent,       // out: existing term on which we are basing the new term
    double *pBestRssDeltaForTerm,   // io: updated if new predictor is better
    double *pBestRssDeltaForParent, // io: used only by FAST_MARS
    bool   *pIsNewForm,         // out
    bool   *pIsLinPred,         // out: true if knot is at min x val so x enters linearly
    double MaxRssPerPred[],     // io: nPreds x 1, max RSS for each predictor over all parents
    double xbx[],               // io: nCases x 1
    double CovSx[],             // io
    double CovCol[],            // io
    double ycboSum[],           // io: nMaxTerms x nResp
    double bxOrth[],            // io
    double bxOrthCenteredT[],   // io
    double bxOrthMean[],        // io
    const int iBestPred,        // in: if -1 then search for best predictor, else use this predictor
    const int iParent,          // in
    const double x[],           // in: nCases x nPreds
    const double y[],           // in: nCases x nResp
    const double Weights[],     // in: nCases x 1
    const int nCases,           // in
    const int nResp,            // in: number of cols in y
    const int nPreds,           // in
    const int nTerms,           // in
    const int nMaxTerms,        // in
    const double yMean[],       // in: vector nResp x 1
    const double MaxAllowedRssDelta, // in: FindKnot rejects any changes in Rss greater than this
    const double bx[],          // in: MARS basis matrix
    const bool FullSet[],       // in
    const int xOrder[],         // in
    const int nUses[],          // in: nbr of times each predictor is used in the model
    const int Dirs[],           // in
    const double NewVarPenalty, // in: penalty for adding a new variable (default is 0)
    const int LinPreds[])       // in: nPreds x 1, 1 if predictor must enter linearly
{
#if USING_R
    const int nServiceR = 1000000 / nCases;
#endif
    double *ybxSum = (double *)malloc1(nResp * sizeof(double),  // working var for FindKnot
                        "ybxSum\t\tnResp %d sizeof(double) %d",
                        nResp, sizeof(double));
    bool UpdatedBestRssDelta = false;
    int iFirstPred = 0;
    int iLastPred = nPreds - 1;
    if (iBestPred >= 0) {
        // we already know the best predictor to use so don't iterate over all preds
        iFirstPred = iBestPred;
        iLastPred = iBestPred;
    }
    for (int iPred = iFirstPred; iPred <= iLastPred; iPred++) {
        if (Dirs_(iParent,iPred) != 0) {    // predictor is in parent term?
            if (TraceGlobal >= 7)
                printf("|Parent %-2d Pred %-2d"
                    "                                   "
                    "                skip (pred is in parent)\n",
                    iParent+IOFFSET, iPred+IOFFSET);
#if USING_R
        } else if (!IsAllowed(iPred, iParent, Dirs, nPreds, nMaxTerms)) {
            if (TraceGlobal >= 7)
                printf("|Parent %-2d Pred %-2d"
                    "                                   "
                    "                skip (not allowed by \"allowed\" func)\n",
                    iParent+IOFFSET, iPred+IOFFSET);
#endif
        } else {
#if USING_R
            static int iServiceR = 0;
            if (++iServiceR > nServiceR) {
                ServiceR();
                iServiceR = 0;
            }
#endif
            if (TraceGlobal >= 7)
                printf("|Parent %-2d Pred %-2d ", iParent+IOFFSET, iPred+IOFFSET);
            const double NewVarAdjust = 1 + (nUses[iPred] == 0? NewVarPenalty: 0);
            double RssDeltaLin = 0;    // change in RSS for iPred entering linearly
            bool IsNewForm = GetNewFormFlag(iPred, iParent, Dirs,
                                FullSet, nTerms, nPreds, nMaxTerms);
            if (IsNewForm) {
                // create a candidate term at bx[,nTerms],
                // with iParent and iPred entering linearly

                AddCandidateLinearTerm(&RssDeltaLin, &IsNewForm,
                    xbx, CovCol, ycboSum, bxOrth, bxOrthCenteredT, bxOrthMean,
                    iPred, iParent, x, y, Weights,
                    nCases, nResp, nTerms, nMaxTerms,
                    yMean, bx, FullSet, NewVarAdjust);

                if (fabs(RssDeltaLin - *pBestRssDeltaForTerm) < ALMOST_ZERO)
                    RssDeltaLin = *pBestRssDeltaForTerm;        // see header note
                if (RssDeltaLin > *pBestRssDeltaForParent)
                    *pBestRssDeltaForParent = RssDeltaLin;
                if (RssDeltaLin > *pBestRssDeltaForTerm) {
                    // The new term (with predictor entering linearly) beats other
                    // candidate terms so far.

                    if (TraceGlobal >= 7)
                        printf("BestRssDeltaForTermSoFar %g (lin pred) ",
                            RssDeltaLin - *pBestRssDeltaForTerm);

                    UpdatedBestRssDelta = true;
                    *pBestRssDeltaForTerm = RssDeltaLin;
                    *pIsLinPred    = true;
                    *piBestCase    = 0;         // knot is at the lowest value of x
                    *piBestPred    = iPred;
                    *piBestParent  = iParent;
                }
            }
            double RssDeltaForParentPredPair = RssDeltaLin;
            if (TraceGlobal >= 7)
                printf("\n");
            if (!LinPreds[iPred]) {
                const int nMinSpan = GetMinSpan(nCases, nPreds, bx, iParent);
                const int nEndSpan = GetEndSpan(nCases, nPreds);
                int iBestCase = -1;
                FindKnot(&iBestCase, &RssDeltaForParentPredPair,
                        CovCol, ycboSum, CovSx, ybxSum,
                        (IsNewForm? nTerms + 1: nTerms),
                        iParent, iPred, nCases, nResp, nMaxTerms,
                        RssDeltaLin, MaxAllowedRssDelta,
                        bx, bxOrth, bxOrthCenteredT, bxOrthMean,
                        x, y, Weights, xOrder, yMean,
                        nMinSpan, nEndSpan, NewVarAdjust);

                if (RssDeltaForParentPredPair > *pBestRssDeltaForParent)
                    *pBestRssDeltaForParent = RssDeltaForParentPredPair;
                if (RssDeltaForParentPredPair > *pBestRssDeltaForTerm) {
                    UpdatedBestRssDelta = true;
                    *pBestRssDeltaForTerm = RssDeltaForParentPredPair;
                    *pIsLinPred    = false;
                    *piBestCase    = iBestCase;
                    *piBestPred    = iPred;
                    *piBestParent  = iParent;
                    *pIsNewForm    = IsNewForm;
                    if (TraceGlobal >= 7)
                        printf("|                  "
                            "Case %4d Cut % 12.4g  "
                            "RssDelta %-12.5g BestRssDeltaForTermSoFar\n",
                            iBestCase+IOFFSET,
                            GetCut(iBestCase, iPred, nCases, x, xOrder),
                            *pBestRssDeltaForTerm);
                }
            }
            if (MaxRssPerPred && RssDeltaForParentPredPair > MaxRssPerPred[iPred])
                MaxRssPerPred[iPred] = RssDeltaForParentPredPair;
        } // else
    } // for iPred
    free1(ybxSum);
    if (UpdatedBestRssDelta && nUses[*piBestPred] == 0) {
        // de-adjust for NewVarPenalty (only makes a difference if NewVarPenalty != 0)
        const double NewVarAdjust = 1 + NewVarPenalty;
        *pBestRssDeltaForTerm *= NewVarAdjust;
    }
}

//-----------------------------------------------------------------------------
// Find a new term to add to the model, if possible, and return the
// selected case (i.e. knot), predictor, and parent term indices.
//
// The new term is a copy of an existing parent term but extended
// by multiplying the parent by a new hockey stick function at the selected knot.
//
// Actually, this usually finds a term _pair_, with left and right hockey sticks.
//
// There are currently nTerms in the model. We want to add a term at index nTerms.

static void FindTerm(
    int    *piBestCase,         // out: return -1 if no new term available
                                //      else return an index into the ORDERED x's
    int    *piBestPred,         // out:
    int    *piBestParent,       // out: existing term on which we are basing the new term
    double *pBestRssDeltaForTerm, // out: adding new term reduces RSS this much
                                  //      will be set to 0 if no possible new term
    bool   *pIsNewForm,         // out
    bool   *pIsLinPred,         // out: true if knot is at min x val so x enters linearly
    double MaxRssPerPred[],     // io: nPreds x 1, max RSS for each predictor over all parents
    double bxOrth[],            // io: column nTerms overwritten
    double bxOrthCenteredT[],   // io: kept in sync with bxOrth
    double bxOrthMean[],        // io: element nTerms overwritten
    const int iBestPred,        // in: if -1 then search for best predictor, else use this predictor
    const double x[],           // in: nCases x nPreds
    const double y[],           // in: nCases x nResp
    const double Weights[],     // in: nCases x 1
    const int nCases,           // in:
    const int nResp,            // in: number of cols in y
    const int nPreds,           // in:
    const int nTerms,           // in:
    const int nMaxDegree,       // in:
    const int nMaxTerms,        // in:
    const double yMean[],       // in: vector nResp x 1
    const double MaxAllowedRssDelta, // in: FindKnot rejects any changes in Rss greater than this
    const double bx[],          // in: MARS basis matrix
    const bool FullSet[],       // in:
    const int xOrder[],         // in:
    const int nFactorsInTerm[], // in:
    const int nUses[],          // in: nbr of times each predictor is used in the model
    const int Dirs[],           // in:
    const int nFastK,           // in: Fast MARS K
    const double NewVarPenalty, // in: penalty for adding a new variable (default is 0)
    const int LinPreds[])       // in: nPreds x 1, 1 if predictor must enter linearly
{
#if !FAST_MARS
    int Dummy = nFastK;             // prevent compiler warning: unused parameter
    Dummy = 0;
#endif
    if (TraceGlobal >= 7)
        printf("\n|Searching for new term %-3d                    "
               "RssDelta 0\n",
               nTerms+IOFFSET);

    *piBestCase = -1;
    *pBestRssDeltaForTerm = 0;
    *pIsLinPred = false;
    *pIsNewForm = false;
    int iCase;

    xbx = (double *)malloc1(nCases * sizeof(double),
                "xbx\t\t\tnCases %d sizeof(double) %d",
                nCases, sizeof(double));
    CovSx  = (double *)malloc1(nMaxTerms * sizeof(double),
                "CovSx\t\t\tnMaxTerms %d sizeof(double) %d",
                nMaxTerms, sizeof(double));
    CovCol = (double *)calloc1(nMaxTerms, sizeof(double),
                "CovCol\t\tnMaxTerms %d sizeof(double) %d",
                nMaxTerms, sizeof(double));
    ycboSum  = (double *)calloc1(nMaxTerms * nResp, sizeof(double),
                "ycbpSum\t\tnMaxTerms %d nResp %d sizeof(double) %d",
                nMaxTerms, nResp, sizeof(double));

    for (int iResp = 0; iResp < nResp; iResp++)
        for (int iTerm = 0; iTerm < nTerms; iTerm++)
            for (iCase = 0; iCase < nCases; iCase++)
                ycboSum_(iTerm,iResp) +=
                    (y_(iCase,iResp) - yMean[iResp]) * bxOrth_(iCase,iTerm);

    int iParent;
#if FAST_MARS
    GetNextParent(true, nFastK); // init queue iterator
    while ((iParent = GetNextParent(false, nFastK)) > -1) {
#else
    for (iParent = 0; iParent < nTerms; iParent++) {
#endif
        // Assume a bad RssDelta for iParent.  This pushes parent terms that
        // can't be used to the bottom of the FastMARS queue.  (A parent can't be
        // used if nFactorsInTerm is too big or all predictors are in the parent).

        double BestRssDeltaForParent = -1;    // used only by FAST_MARS

        if (nFactorsInTerm[iParent] >= nMaxDegree) {
            if (TraceGlobal >= 7)
                printf("|Parent %-2d"
                    "                                                      "
                    "     skip (nFactorsInTerm %d)\n",
                    iParent+IOFFSET, nFactorsInTerm[iParent]);
        } else {
            FindPred(piBestCase, piBestPred, piBestParent, pBestRssDeltaForTerm,
                &BestRssDeltaForParent, pIsNewForm, pIsLinPred, MaxRssPerPred,
                xbx, CovSx, CovCol, ycboSum, bxOrth, bxOrthCenteredT, bxOrthMean,
                iBestPred, iParent, x, y, Weights,
                nCases, nResp, nPreds, nTerms, nMaxTerms, yMean, MaxAllowedRssDelta,
                bx, FullSet, xOrder, nUses, Dirs, NewVarPenalty,
                LinPreds);
        }
#if FAST_MARS
        UpdateRssDeltaInQ(iParent, nTerms, BestRssDeltaForParent);
#endif
    } // iParent
    if (TraceGlobal >= 7)
        printf("\n");
    free1(ycboSum);
    free1(CovCol);
    free1(CovSx);
    free1(xbx);
}

//-----------------------------------------------------------------------------
static void PrintForwardProlog(const int nCases, const int nPreds,
        const char *sPredNames[])   // in: predictor names, can be NULL
{
    if (TraceGlobal == 1)
        printf("Forward pass term 1");
    else if(TraceGlobal == 1.5)
        printf("Forward pass term 1\n");
    else if (TraceGlobal >= 2) {
        const char *sMinSpan = (nMinSpanGlobal < 0)? " (old minspan calculation)": "";
        printf("Forward pass: minspan %d endspan %d%s\n\n",
            GetMinSpan(nCases, nPreds, NULL, 0),
            GetEndSpan(nCases, nPreds), sMinSpan);

        printf("         GRSq    RSq     DeltaRSq Pred ");
        if (sPredNames)
            printf("    PredName  ");
        printf("       Cut  Terms   ParentTerm\n");

        printf("1      0.0000 0.0000                               %s%d\n",
            (sPredNames? "              ":""), IOFFSET);
    }
}

//-----------------------------------------------------------------------------
static void PrintForwardStep(
        const int nTerms,
        const int nUsedTerms,
        const int iBestCase,
        const int iBestPred,
        const double RSq,
        const double RSqDelta,
        const double Gcv,
        const double GcvNull,
        const int nCases,
        const int xOrder[],
        const double x[],
        const bool IsLinPred,
        const bool IsNewForm,
        const char *sPredNames[])   // in: predictor names, can be NULL
{
    if (TraceGlobal == 6)
        printf("\n\n");
    if (TraceGlobal == 1) {
        printf(", ");
        if (nTerms % 30 == 29)
            printf("\n     ");
        printf("%d", nTerms+1);
    } else if (TraceGlobal == 1.5)
        printf("Forward pass term %d\n", nTerms+1);
    else if (TraceGlobal >= 2) {
        printf("%-4d%9.4f %6.4f %12.4g  ",
            nTerms+IOFFSET, 1-Gcv/GcvNull, RSq, RSqDelta);
        if (iBestPred < 0)
            printf("  -                                ");
        else {
            printf("%3d", iBestPred+IOFFSET);
            if (sPredNames) {
                if (sPredNames[iBestPred] && sPredNames[iBestPred][0])
                    printf(" %12.12s ", sPredNames[iBestPred]);
                else
                    printf(" %12.12s ", " ");
            }
            if (iBestCase == -1)
                printf("       none  ");
            else
                printf("% 11.5g%c ",
                    GetCut(iBestCase, iBestPred, nCases, x, xOrder),
                        (IsLinPred? '<': ' '));
            if (!IsLinPred && IsNewForm)  // two new used terms?
                printf("%-3d %-3d ", nUsedTerms-2+IOFFSET, nUsedTerms-1+IOFFSET);
            else
                printf("%-3d     ", nUsedTerms-1+IOFFSET);
            // AddTermPair will print the parents shortly, if any
        }
    }
    if (TraceGlobal != 0)
        fflush(stdout);
}

//-----------------------------------------------------------------------------
static void PrintForwardEpilog(
            const int nTerms, const int nMaxTerms,
            const double Thresh,
            const double RSq, const double RSqDelta,
            const double Gcv, const double GcvNull,
            const int iBestCase,
            const bool FullSet[])
{
    if (TraceGlobal >= 1) {
        double GRSq = 1-Gcv/GcvNull;

        // print reason why we stopped adding terms
        // NOTE: this code must match the loop termination conditions in ForwardPass

        // treat very low nMaxTerms as a special case
        // because RSDelta etc. not yet completely initialized

        if (nMaxTerms < 3)
            printf("\nReached max number of terms %d", nMaxTerms);

        else if (Thresh != 0 && GRSq < MIN_GRSQ) {
            if(GRSq < -1000)
                printf("\nReached GRSq = -Inf at %d terms\n", nTerms);
            else
                printf("\nReached min GRSq (GRSq %g < %g) at %d terms\n",
                    GRSq, MIN_GRSQ, nTerms);
        }
        else if (Thresh != 0 && RSqDelta < Thresh)
            printf("\nReached delta RSq threshold (DeltaRSq %g < %g) at %d terms\n",
                RSqDelta, Thresh, nTerms);

        else if (RSq > 1-Thresh)
            printf("\nReached max RSq (RSq %g > %g) at %d terms\n",
                RSq, 1-Thresh, nTerms);

        else if (iBestCase < 0)
            printf("\nNo new term increases RSq (reached numerical limits) at %d terms\n",
                nTerms);

        else {
            printf("\nReached max number of terms %d", nMaxTerms);
            if (nTerms < nMaxTerms)
                printf(" (no room for another term pair)");
            printf("\n");
        }
        printf("After forward pass GRSq %.4g RSq %.4g\n", GRSq, RSq);
    }
    if (TraceGlobal >= 2) {
        printf("Forward pass complete: %d terms", nTerms);
        int nUsed = GetNbrUsedCols(FullSet, nMaxTerms);
        if (nUsed != nTerms)
            printf(" (%d terms used)", nUsed);
        printf("\n");
    }
    if (TraceGlobal >= 3)
        printf("\n");
}

//-----------------------------------------------------------------------------
static void CheckVec(const double x[], int nCases, int nCols, const char sVecName[])
{
    int iCol, iCase;

    for (iCol = 0; iCol < nCols; iCol++)
        for (iCase = 0; iCase < nCases; iCase++) {
#if USING_R
             if (ISNA(x[iCase + iCol * nCases])) {
                 if (nCols > 1)
                     error("%s[%d,%d] is NA",
                         sVecName, iCase+IOFFSET, iCol+IOFFSET);
                 else
                     error("%s[%d] is NA", sVecName, iCase+IOFFSET);
             }
#endif
             if (ISNAN(x[iCase + iCol * nCases])) {
                 if (nCols > 1)
                     error("%s[%d,%d] is NaN",
                         sVecName, iCase+IOFFSET, iCol+IOFFSET);
                 else
                     error("%s[%d] is NaN", sVecName, iCase+IOFFSET);
             }
             if (!FINITE(x[iCase + iCol * nCases])) {
                 if (nCols > 1)
                     error("%s[%d,%d] is not finite",
                         sVecName, iCase+IOFFSET, iCol+IOFFSET);
                 else
                     error("%s[%d] is not finite", sVecName, iCase+IOFFSET);
             }
    }
}

//-----------------------------------------------------------------------------
static void CheckRssNull(double RssNull, const double y[], int iResp, int nCases)
{
    if (RssNull / nCases < 1e-8) {
        if (iResp)
            error("variance of y[,%d] is zero (values are all equal to %g)",
                iResp+IOFFSET, &y_(0,iResp));
        else
            error("variance of y is zero (values are all equal to %g)",
                &y_(0,iResp));
    }
}

//-----------------------------------------------------------------------------
static double *pInitWeights(const double WeightsArg[], int nCases)
{
    Weights = (double *)malloc1(nCases * sizeof(double),
                    "Weights\t\tnCases %d sizeof(double) %d",
                    nCases, sizeof(double));
    if (!WeightsArg)
        for (int iCase = 0; iCase < nCases; iCase++)
            Weights[iCase] = 1;
    else for (int iCase = 0; iCase < nCases; iCase++) {
        Weights[iCase] = WeightsArg[iCase];
#if USING_R
        if (ISNA(Weights[iCase]))
            error("Weights[%d] is NA");
#endif
        if (ISNAN(Weights[iCase]))
            error("Weights[%d] is NaN");
        if (!FINITE(Weights[iCase]))
            error("Weights[%d] is not finite");
        if (Weights[iCase] < ALMOST_ZERO)
            error("Weights[%d] is less than or equal to zero");
    }
    return Weights;
}

//-----------------------------------------------------------------------------
// Forward pass
//
// After initializing the intercept term, the main for loop adds terms in pairs.
// In the for loop, nTerms is the index of the potential new term; nTerms+1
// the index of its partner.
// The upper term in the term pair may not be useable.  If so we still
// increment nTerms by 2 but don't set the flag in FullSet.
//
// TODO feature: add option to prescale x and y

static void ForwardPass(
    int    *pnTerms,            // out: highest used term number in full model
    bool   FullSet[],           // out: 1 * nMaxTerms, indices of lin indep cols of bx
    double bx[],                // out: MARS basis matrix, nCases * nMaxTerms
    int    Dirs[],              // out: nMaxTerms * nPreds, -1,0,1,2 for iTerm, iPred
    double Cuts[],              // out: nMaxTerms * nPreds, cut for iTerm, iPred
    int    nFactorsInTerm[],    // out: number of hockey stick funcs in each MARS term
    int    nUses[],             // out: nbr of times each predictor is used in the model
    const double x[],           // in: nCases x nPreds
    const double y[],           // in: nCases x nResp
    const double WeightsArg[],  // in: nCases x 1, can be NULL, currently ignored
    const int nCases,           // in: number of rows in x and elements in y
    const int nResp,            // in: number of cols in y
    const int nPreds,           // in:
    const int nMaxDegree,       // in:
    const int nMaxTerms,        // in:
    const double Penalty,       // in: GCV penalty per knot
    double Thresh,              // in: forward step threshold
    int nFastK,                 // in: Fast MARS K
    const double FastBeta,      // in: Fast MARS ageing coef
    const double NewVarPenalty, // in: penalty for adding a new variable (default is 0)
    const int  LinPreds[],      // in: nPreds x 1, 1 if predictor must enter linearly
    const bool UseBetaCache,    // in: true to use the beta cache, for speed
    const char *sPredNames[])   // in: predictor names, can be NULL
{
    if (TraceGlobal >= 5)
        printf("earth.c %s\n", VERSION);

    // The limits below are somewhat arbitrary and generous.
    // They are intended to catch gross errors on the part of the
    // caller, and to prevent crashes because of 0 sizes etc.
    // We use error rather than ASSERT because these are user settable params
    // and we want to be informative from the user's perspective.
    // The errors are reported using the variable names in the R code.

    // prevent possible minspan range problems, also prevent crash when nCases==0
    if (nCases < 8)
        error("need at least 8 rows in x, you have %d", nCases);
#if 0 // removed for earth 2.0-6
    if (nCases < nPreds)    // (this check may not actually be necessary)
        warning("Need as many rows as columns in x (nrow %d ncol %d)",
                 nCases, nPreds);
#endif
    if (nCases > 1e8)
        error("too many rows %d in input matrix, max is 1e8", nCases);
    if (nResp < 1)
        error("nResp %d < 1", nResp);
    if (nResp > 1e6)
        error("nResp %d > 1e6", nResp);
    if (nPreds < 1)
        error("nPreds %d < 1", nPreds);
    if (nPreds > 1e5)
        error("nPreds %d > 1e5", nPreds);
    if (nMaxDegree <= 0)
        error("degree %d <= 0", nMaxDegree);
    if (nMaxDegree > MAX_DEGREE)
        error("degree %d > %d", nMaxDegree, MAX_DEGREE);
    if (nMaxTerms < 3)      // prevent internal misbehaviour
        error("nk %d < 3", nMaxTerms);
    if (nMaxTerms > 10000)
        error("nk %d > 10000", nMaxTerms);
    if (nFastK <= 0)
        nFastK = 10000+1;   // bigger than any nMaxTerms
    if (nFastK < 3)         // avoid possible queue boundary conditions
        nFastK = 3;
    if (Penalty < 0 && Penalty != -1)
        error("penalty %g < 0, the only legal value less than 0 is -1 "
            "(meaning terms and knots are free)", Penalty);
    if (Penalty > 1000)
        error("penalty %g > 1000", Penalty);
    if (Thresh < 0)
        error("thresh %g < 0", Thresh);
    if (Thresh >= 1)
        error("thresh %g >= 1", Thresh);
    if (nMinSpanGlobal > nCases/2)
        error("minspan %d > nrow(x)/2 %d", nMinSpanGlobal, nCases/2);
    if (FastBeta < 0)
        error("fast.beta %g < 0", FastBeta);
    if (FastBeta > 1000)
        error("fast.beta %g > 1000", FastBeta);
    if (TraceGlobal < 0)
        warning("trace %g < 0", TraceGlobal);
    if (TraceGlobal > 10)
        warning("trace %g > 10", TraceGlobal);
    if (NewVarPenalty < 0)
        warning("newvar.penalty %g < 0", NewVarPenalty);
    if (NewVarPenalty > 10)
        warning("newvar.penalty %g > 10", NewVarPenalty);
    if (UseBetaCache != 0 && UseBetaCache != 1)
        warning("Use.Beta.Cache is neither TRUE nor FALSE");

    CheckVec(x, nCases, nPreds, "x");
    CheckVec(y, nCases, nResp,  "y");

    bxOrth          = (double *)malloc1(nCases * nMaxTerms * sizeof(double),
                        "bxOrth\t\tnCases %d nMaxTerms %d  sizeof(double) %d",
                        nCases, nMaxTerms, sizeof(double));

    bxOrthCenteredT = (double *)malloc1(nMaxTerms * nCases * sizeof(double),
                        "bxOrthCenteredT\tnMaxTerms %d nCases %d  sizeof(double) %d",
                        nMaxTerms, nCases, sizeof(double));

    bxOrthMean      = (double *)malloc1(nMaxTerms * nResp * sizeof(double),
                        "bxOrthMean\t\tnMaxTerms %d nResp %d  sizeof(double) %d",
                        nMaxTerms, nResp, sizeof(double));

    yMean           = (double *)malloc1(nResp * sizeof(double),
                        "yMean\t\t\tnResp %d sizeof(double) %d",
                        nResp, sizeof(double));

    memset(FullSet,        0, nMaxTerms * sizeof(bool));
    memset(Dirs,           0, nMaxTerms * nPreds * sizeof(int));
    memset(Cuts,           0, nMaxTerms * nPreds * sizeof(double));
    memset(nFactorsInTerm, 0, nMaxTerms * sizeof(int));
    memset(nUses,          0, nPreds * sizeof(int));
    memset(bx,             0, nCases * nMaxTerms * sizeof(double));
    Weights = pInitWeights(WeightsArg, nCases);
    xOrder = OrderArray(x, nCases, nPreds);
    InitBetaCache(UseBetaCache, nMaxTerms, nPreds);
    FullSet[0] = true;  // intercept
    bool GoodCol;
    InitBxOrthCol(bxOrth, bxOrthCenteredT, bxOrthMean, &GoodCol,   // intercept col 0
        &bx_(0,0), 0 /*nTerms*/, FullSet, nCases, nMaxTerms, -1, -1, Weights);
    ASSERT(GoodCol);

    for (int iCase = 0; iCase < nCases; iCase++)
        bx_(iCase,0) = 1;
    double RssNull = 0;
    for (int iResp = 0; iResp < nResp; iResp++) {
        yMean[iResp] = Mean(&y_(0,iResp), nCases);
        RssNull += SumOfSquares(&y_(0,iResp), yMean[iResp], nCases);
        CheckRssNull(RssNull, y, iResp, nCases);
    }
    double Rss = RssNull, RssDelta = RssNull, RSq = 0, RSqDelta = 0;
    int nUsedTerms = 1;     // number of used basis terms including intercept, for GCV calc
    double Gcv = 0, GcvNull = GetGcv(nUsedTerms, nCases, RssNull, Penalty);
    PrintForwardProlog(nCases, nPreds, sPredNames);
#if FAST_MARS
    InitQ(nMaxTerms);
    AddTermToQ(0, 1, RssNull, true, nMaxTerms, FastBeta); // intercept term into Q
#endif
    int nTerms = -1, iBestCase = -1;
    for (nTerms = 1;                                    // start after intercept
            nTerms < nMaxTerms-1 && RSq < 1-Thresh;     // -1 allows for upper term in pair
            nTerms += 2) {                              // add terms in pairs
        int iBestPred = -1, iBestParent = -1;
        bool IsNewForm, IsLinPred;
#if USING_R
        ServiceR();
#endif
        if (Rss <= 0)
            error("assertion failed: Rss <= 0 (y is all const?)");
        ASSERT(RssDelta > 0);
        const double MaxAllowedRssDelta = min(1.01 * Rss, 2 * RssDelta);

        FindTerm(&iBestCase, &iBestPred, &iBestParent,
            &RssDelta, &IsNewForm, &IsLinPred, NULL /* MaxRssPerPred */,
            bxOrth, bxOrthCenteredT, bxOrthMean, -1, x, y, Weights,
            nCases, nResp, nPreds, nTerms, nMaxDegree, nMaxTerms,
            yMean, MaxAllowedRssDelta,
            bx, FullSet, xOrder, nFactorsInTerm, nUses, Dirs,
            nFastK, NewVarPenalty, LinPreds);

        nUsedTerms++;
        if (!IsLinPred && IsNewForm)    // add paired term too?
            nUsedTerms++;
        Rss -= RssDelta;
        if (Rss < ALMOST_ZERO)          // RSS can go slightly neg due to rounding
            Rss = 0;                    // or can have very small values
        Gcv = GetGcv(nUsedTerms, nCases, Rss, Penalty);
        const double OldRSq = RSq;
        RSq = 1-Rss/RssNull;
        RSqDelta = RSq - OldRSq;
        if (RSqDelta < ALMOST_ZERO) // for consistent results with different
            RSqDelta = 0;           // float hardware else print nbrs like -2e-18

        PrintForwardStep(nTerms, nUsedTerms, iBestCase, iBestPred, RSq, RSqDelta,
            Gcv, GcvNull, nCases, xOrder, x, IsLinPred, IsNewForm, sPredNames);

        if (iBestCase < 0 ||
                (Thresh != 0 && ((1-Gcv/GcvNull) < MIN_GRSQ || RSqDelta < Thresh))) {
            if (TraceGlobal >= 2)
                printf("reject term\n");
            break;                      // NOTE break
        }
        AddTermPair(Dirs, Cuts, bx, bxOrth, bxOrthCenteredT, bxOrthMean,
            FullSet, nFactorsInTerm, nUses,
            nTerms, iBestParent, iBestCase, iBestPred, nPreds, nCases,
            nMaxTerms, IsNewForm, IsLinPred, LinPreds, x, xOrder, Weights);

#if FAST_MARS
        if (!IsLinPred && IsNewForm) {  // good upper term?
            AddTermToQ(nTerms,   nTerms, POS_INF, false, nMaxTerms, FastBeta);
            AddTermToQ(nTerms+1, nTerms, POS_INF, true,  nMaxTerms, FastBeta);
        } else
            AddTermToQ(nTerms,   nTerms, POS_INF, true,  nMaxTerms, FastBeta);
        if (TraceGlobal == 6)
            PrintSortedQ(nFastK);
#endif
        if (TraceGlobal >= 2)
            printf("\n");
    }
    PrintForwardEpilog(nTerms, nMaxTerms, Thresh, RSq, RSqDelta,
                       Gcv, GcvNull, iBestCase, FullSet);
    *pnTerms = nTerms;
    FreeBetaCache();
#if FAST_MARS
    FreeQ();
#endif
    free1(xOrder);
    free1(Weights);
    Weights = NULL;
    free1(yMean);
    free1(bxOrthMean);
    free1(bxOrthCenteredT);
    free1(bxOrth);
}

//-----------------------------------------------------------------------------
// This is an interface from R to the C routine ForwardPass

#if USING_R
void ForwardPassR(              // for use by R
    int    FullSet[],           // out: nMaxTerms x 1, bool vec of lin indep cols of bx
    double bx[],                // out: MARS basis matrix, nCases x nMaxTerms
    double Dirs[],              // out: nMaxTerms x nPreds, elements are -1,0,1,2
    double Cuts[],              // out: nMaxTerms x nPreds, cut for iTerm,iPred
    const double x[],           // in: nCases x nPreds
    const double y[],           // in: nCases x nResp
    const double WeightsArg[],  // in: nCases x 1, can be R_NilValue, currently ignored
    const int *pnCases,         // in: number of rows in x and elements in y
    const int *pnResp,          // in: number of cols in y
    const int *pnPreds,         // in: number of cols in x
    const int *pnMaxDegree,     // in:
    const int *pnMaxTerms,      // in:
    const double *pPenalty,     // in:
    double *pThresh,            // in: forward step threshold
    const int *pnMinSpan,       // in:
    const int *pnFastK,         // in: Fast MARS K
    const double *pFastBeta,    // in: Fast MARS ageing coef
    const double *pNewVarPenalty, // in: penalty for adding a new variable (default is 0)
    const int  LinPreds[],        // in: nPreds x 1, 1 if predictor must enter linearly
    const SEXP Allowed,           // in: constraints function
    const int *pnAllowedFuncArgs, // in: number of arguments to Allowed function, 3 or 4
    const SEXP Env,               // in: environment for Allowed function
    const int *pnUseBetaCache,    // in: 1 to use the beta cache, for speed
    const double *pTrace,         // in: 0 none 1 overview 2 forward 3 pruning 4 more pruning
    const char *sPredNames[])     // in: predictor names in trace printfs, can be R_NilValue
{
    TraceGlobal = *pTrace;
    nMinSpanGlobal = *pnMinSpan;

    const int nCases = *pnCases;
    const int nResp = *pnResp;
    const int nPreds = *pnPreds;
    const int nMaxTerms = *pnMaxTerms;

    // nUses is the number of time each predictor is used in the model
    nUses = (int *)malloc1(*pnPreds * sizeof(int),
                    "nUses\t\t\t*pnPreds %d sizeof(int)",
                    *pnPreds, sizeof(int));

    // nFactorsInTerm is number of hockey stick functions in basis term
    nFactorsInTerm = (int *)malloc1(nMaxTerms * sizeof(int),
                        "nFactorsInTerm\tnMaxTerms %d sizeof(int) %d",
                        nMaxTerms, sizeof(int));

    iDirs = (int *)calloc1(nMaxTerms * nPreds, sizeof(int),
                        "iDirs\t\t\tnMaxTerms %d nPreds %d sizeof(int) %d",
                        nMaxTerms, nPreds, sizeof(int));

    // convert int to bool (may be redundant, depending on compiler)
    BoolFullSet = (int *)malloc1(nMaxTerms * sizeof(bool),
                        "BoolFullSet\t\tnMaxTerms %d sizeof(bool) %d",
                        nMaxTerms, sizeof(bool));

    int iTerm;
    for (iTerm = 0; iTerm < nMaxTerms; iTerm++)
        BoolFullSet[iTerm] = FullSet[iTerm];

    // convert R NULL to C NULL
    if ((void *)sPredNames == (void *)R_NilValue)
        sPredNames = NULL;
    if ((void *)WeightsArg == (void *)R_NilValue)
        WeightsArg = NULL;

    InitAllowedFunc(Allowed, *pnAllowedFuncArgs, Env, sPredNames, nPreds);

    int nTerms;
    ForwardPass(&nTerms, BoolFullSet, bx, iDirs, Cuts, nFactorsInTerm, nUses,
            x, y, WeightsArg, nCases, nResp, nPreds, *pnMaxDegree, nMaxTerms,
            *pPenalty, *pThresh, *pnFastK, *pFastBeta, *pNewVarPenalty,
            LinPreds, (bool)(*pnUseBetaCache), sPredNames);

    FreeAllowedFunc();

    // remove linearly independent columns if necessary -- this updates BoolFullSet

    RegressAndFix(NULL, NULL, NULL, BoolFullSet,
        bx, y, WeightsArg, nCases, nResp, nMaxTerms);

    for (iTerm = 0; iTerm < nMaxTerms; iTerm++)     // convert int to double
        for (int iPred = 0; iPred < nPreds; iPred++)
            Dirs[iTerm + iPred * nMaxTerms] =
                iDirs[iTerm + iPred * nMaxTerms];

    for (iTerm = 0; iTerm < nMaxTerms; iTerm++)     // convert bool to int
        FullSet[iTerm] = BoolFullSet[iTerm];

    free1(BoolFullSet);
    free1(iDirs);
    free1(nFactorsInTerm);
    free1(nUses);
}
#endif // USING_R

//-----------------------------------------------------------------------------
// Step backwards through the terms, at each step deleting the term that
// causes the least RSS increase.  The subset of terms and RSS of each subset are
// saved in PruneTerms and RssVec (which are indexed on subset size).
//
// The crux of the method used here is that the change in RSS (for nResp=1)
// caused by removing predictor iPred is DeltaRss = sq(Betas[iPred]) / Diags[iPred]
// where Diags is the diagonal elements of the inverse of X'X.
// See for example Miller (see refs in file header) section 3.4 p44.
//
// For multiple responses we sum the above DeltaRss over all responses.
//
// This method is fast and simple but accuracy can be poor if inv(X'X) is
// ill conditioned.  The Miller code in the R package "leaps" uses a more
// stable method, but does not support multiple responses.
//
// The "Xtx" in the name refers to the X'X matrix.

static int EvalSubsetsUsingXtx(
    bool   PruneTerms[],    // out: nMaxTerms x nMaxTerms
    double RssVec[],        // out: nMaxTerms x 1, RSS of each subset
    const int    nCases,    // in
    const int    nResp,     // in: number of cols in y
    const int    nMaxTerms, // in: number of MARS terms in full model
    const double bx[],      // in: nCases x nMaxTerms, all cols must be indep
    const double y[],       // in: nCases * nResp
    const double WeightsArg[]) // in: nCases x 1, can be NULL
{
    double *Betas = (double *)malloc1(nMaxTerms * nResp * sizeof(double),
                        "Betas\t\t\tnMaxTerms %d nResp %d sizeof(double) %d",
                        nMaxTerms, nResp, sizeof(double));

    double *Diags = (double *)malloc1(nMaxTerms * sizeof(double),
                        "Diags\t\t\tnMaxTerms %d sizeof(double) %d",
                        nMaxTerms, sizeof(double));

    if (WeightsArg)
        Weights = pInitWeights(WeightsArg, nCases);

    WorkingSet = (bool *)malloc1(nMaxTerms * sizeof(bool),
                        "WorkingSet\t\tnMaxTerms %d sizeof(bool) %d",
                        nMaxTerms, sizeof(bool));

    for (int i = 0; i < nMaxTerms; i++)
        WorkingSet[i] = true;

    int error_code = 0;
    for (int nUsedCols = nMaxTerms; nUsedCols > 0; nUsedCols--) {
        int nRank;
        double Rss;
        Regress(Betas, NULL, &Rss, Diags, &nRank, NULL,
            bx, y, Weights, nCases, nResp, nMaxTerms, WorkingSet);

        if(nRank != nUsedCols)
        {
        	error_code = 1;
        	break;
        }
//            error("nRank %d != nUsedCols %d "
//                "(probably because of lin dep terms in bx)\n",
//                nRank, nUsedCols);

        RssVec[nUsedCols-1] = Rss;
        memcpy(PruneTerms + (nUsedCols-1) * nMaxTerms, WorkingSet,
            nMaxTerms * sizeof(bool));

        if (nUsedCols == 1)
            break;

        // set iDelete to the best term for deletion

        int iDelete = -1;   // term to be deleted
        int iTerm1 = 0;     // index taking into account false vals in WorkingSet
        double MinDeltaRss = POS_INF;
        for (int iTerm = 0; iTerm < nMaxTerms; iTerm++) {
            if (WorkingSet[iTerm]) {
                double DeltaRss = 0;
                for (int iResp = 0; iResp < nResp; iResp++)
                    DeltaRss += sq(Betas_(iTerm1, iResp)) / Diags[iTerm1];
                if (iTerm > 0 && DeltaRss < MinDeltaRss) {   // new minimum?
                    MinDeltaRss = DeltaRss;
                    iDelete = iTerm;
                }
                iTerm1++;
            }
        }
        ASSERT(iDelete > 0);
        WorkingSet[iDelete] = false;
    }
    if (WeightsArg)
        free1(Weights);
    free1(WorkingSet);
    free1(Diags);
    free1(Betas);

    return error_code;
}

//-----------------------------------------------------------------------------
// This is invoked from R if y has multiple columns i.e. a multiple response model.
// It is needed because the alternative (the leaps package) supports
// only one response.

#if USING_R
void EvalSubsetsUsingXtxR(      // for use by R
    double       PruneTerms[],  // out: specifies which cols in bx are in best set
    double       RssVec[],      // out: nTerms x 1
    const int    *pnCases,      // in
    const int    *pnResp,       // in: number of cols in y
    const int    *pnMaxTerms,   // in
    const double bx[],          // in: MARS basis matrix, all cols must be indep
    const double y[],           // in: nCases * nResp
    const double WeightsArg[])  // in: nCases x 1, can be R_NilValue
{
    const int nMaxTerms = *pnMaxTerms;
    bool *BoolPruneTerms = (int *)malloc1(nMaxTerms * nMaxTerms * sizeof(bool),
                                "BoolPruneTerms\tMaxTerms %d nMaxTerms %d sizeof(bool) %d",
                                nMaxTerms, nMaxTerms, sizeof(bool));

    if ((void *)WeightsArg == (void *)R_NilValue)
        WeightsArg = NULL;

    EvalSubsetsUsingXtx(BoolPruneTerms, RssVec, *pnCases, *pnResp,
                        nMaxTerms, bx, y, WeightsArg);

    // convert BoolPruneTerms to upper triangular matrix PruneTerms

    for (int iModel = 0; iModel < nMaxTerms; iModel++) {
        int iPrune = 0;
        for (int iTerm = 0; iTerm < nMaxTerms; iTerm++)
            if (BoolPruneTerms[iTerm + iModel * nMaxTerms])
                PruneTerms[iModel + iPrune++ * nMaxTerms] = iTerm + 1;
    }
    free1(BoolPruneTerms);
}
#endif

//-----------------------------------------------------------------------------
#if STANDALONE
static void BackwardPass(
    double *pBestGcv,       // out: GCV of the best model i.e. BestSet columns of bx
    bool   BestSet[],       // out: nMaxTerms x 1, indices of best set of cols of bx
    double Residuals[],     // out: nCases x nResp
    double Betas[],         // out: nMaxTerms x nResp
    const double bx[],      // in: nCases x nMaxTerms
    const double y[],       // in: nCases x nResp
    const double WeightsArg[], // in: nCases x 1, can be NILL
    const int nCases,       // in: number of rows in bx and elements in y
    const int nResp,        // in: number of cols in y
    const int nMaxTerms,    // in: number of cols in bx
    const double Penalty)   // in: GCV penalty per knot
{
    double *RssVec = (double *)malloc1(nMaxTerms * sizeof(double),
                        "RssVec\t\tnMaxTerms %d sizeof(double) %d",
                        nMaxTerms, sizeof(double));

    bool *PruneTerms = (bool *)malloc1(nMaxTerms * nMaxTerms * sizeof(bool),
                        "PruneTerms\t\tnMaxTerms %d nMaxTerms %d sizeof(bool) %d",
                        nMaxTerms, nMaxTerms, sizeof(bool));

    EvalSubsetsUsingXtx(PruneTerms, RssVec, nCases, nResp,
                        nMaxTerms, bx, y, WeightsArg);

    // now we have the RSS for each model, so find the iModel which has the best GCV

    if (TraceGlobal >= 3)
        printf("Backward pass:\nSubsetSize         GRSq          RSq\n");
    int iBestModel = -1;
    double GcvNull = GetGcv(1, nCases, RssVec[0], Penalty);
    double BestGcv = POS_INF;
    for (int iModel = 0; iModel < nMaxTerms; iModel++) {
        const double Gcv = GetGcv(iModel+1, nCases, RssVec[iModel], Penalty);
        if(Gcv < BestGcv) {
            iBestModel = iModel;
            BestGcv = Gcv;
        }
        if (TraceGlobal >= 3)
            printf("%10d %12.4f %12.4f\n", iModel+IOFFSET,
                1 - BestGcv/GcvNull, 1 - RssVec[iModel]/RssVec[0]);
    }
    if (TraceGlobal >= 3)
        printf("\nBackward pass complete: selected %d terms of %d, GRSq %g RSq %g\n\n",
            iBestModel+IOFFSET, nMaxTerms,
            1 - BestGcv/GcvNull, 1 - RssVec[iBestModel]/RssVec[0]);

    // set BestSet to the model which has the best GCV

    ASSERT(iBestModel >= 0);
    memcpy(BestSet, PruneTerms + iBestModel * nMaxTerms, nMaxTerms * sizeof(bool));
    free1(PruneTerms);
    free1(RssVec);
    *pBestGcv = BestGcv;

    // get final model Betas, Residuals, Rss

    RegressAndFix(Betas, Residuals, NULL, BestSet,
        bx, y, WeightsArg, nCases, nResp, nMaxTerms);

}
#endif // STANDALONE

//-----------------------------------------------------------------------------
#if STANDALONE
static int DiscardUnusedTerms(
    double bx[],             // io: nCases x nMaxTerms
    int    Dirs[],           // io: nMaxTerms x nPreds
    double Cuts[],           // io: nMaxTerms x nPreds
    bool   WhichSet[],       // io: tells us which terms to discard
    int    nFactorsInTerm[], // io
    const int nMaxTerms,
    const int nPreds,
    const int nCases)
{
    int nUsed = 0, iTerm;
    for (iTerm = 0; iTerm < nMaxTerms; iTerm++)
        if (WhichSet[iTerm]) {
            memcpy(bx + nUsed * nCases, bx + iTerm * nCases, nCases * sizeof(double));
            for (int iPred = 0; iPred < nPreds; iPred++) {
                Dirs_(nUsed, iPred) = Dirs_(iTerm, iPred);
                Cuts_(nUsed, iPred) = Cuts_(iTerm, iPred);
            }
            nFactorsInTerm[nUsed] = nFactorsInTerm[iTerm];
            nUsed++;
        }
    memset(WhichSet, 0, nMaxTerms * sizeof(bool));
    for (iTerm = 0; iTerm < nUsed; iTerm++)
        WhichSet[iTerm] = true;
    return nUsed;
}
#endif // STANDALONE

//-----------------------------------------------------------------------------
#if STANDALONE
void Earth(
    double *pBestGcv,       // out: GCV of the best model i.e. BestSet columns of bx
    int    *pnTerms,        // out: max term nbr in final model, after removing lin dep terms
    bool   BestSet[],       // out: nMaxTerms x 1, indices of best set of cols of bx
    double bx[],            // out: nCases x nMaxTerms
    int    Dirs[],          // out: nMaxTerms x nPreds, -1,0,1,2 for iTerm, iPred
    double Cuts[],          // out: nMaxTerms x nPreds, cut for iTerm, iPred
    double Residuals[],     // out: nCases x nResp
    double Betas[],         // out: nMaxTerms x nResp
    const double x[],       // in: nCases x nPreds
    const double y[],       // in: nCases x nResp
    const double WeightsArg[], // in: nCases x 1, can be NULL, currently ignored
    const int nCases,       // in: number of rows in x and elements in y
    const int nResp,        // in: number of cols in y
    const int nPreds,       // in: number of cols in x
    const int nMaxDegree,   // in: Friedman's mi
    const int nMaxTerms,    // in: includes the intercept term
    const double Penalty,   // in: GCV penalty per knot
    double Thresh,          // in: forward step threshold
    const int nMinSpan,     // in: set to non zero to override internal calculation
    const bool Prune,       // in: do backward pass
    const int nFastK,       // in: Fast MARS K
    const double FastBeta,  // in: Fast MARS ageing coef
    const double NewVarPenalty, // in: penalty for adding a new variable
    const int LinPreds[],       // in: nPreds x 1, 1 if predictor must enter linearly
    const bool UseBetaCache,    // in: 1 to use the beta cache, for speed
    const double Trace,         // in: 0 none 1 overview 2 forward 3 pruning 4 more pruning
    const char *sPredNames[])   // in: predictor names in trace printfs, can be NULL
{
#if _MSC_VER && _DEBUG
    InitMallocTracking();
#endif
    TraceGlobal = Trace;
    nMinSpanGlobal = nMinSpan;

    // nUses is the number of time each predictor is used in the model
    nUses = (int *)malloc1(nPreds * sizeof(int),
                        "nUses\t\t\tnPreds %d sizeof(int) %d",
                        nPreds, sizeof(int));

    // nFactorsInTerm is number of hockey stick functions in basis term
    nFactorsInTerm = (int *)malloc1(nMaxTerms * sizeof(int),
                            "nFactorsInTerm\tnMaxTerms %d sizeof(int) %d",
                            nMaxTerms, sizeof(int));

    int nTerms;
    ForwardPass(&nTerms, BestSet, bx, Dirs, Cuts, nFactorsInTerm, nUses,
        x, y, WeightsArg, nCases, nResp, nPreds, nMaxDegree, nMaxTerms,
        Penalty, Thresh, nFastK, FastBeta, NewVarPenalty,
        LinPreds, UseBetaCache, sPredNames);

    // ensure bx is full rank by updating BestSet, and get Residuals and Betas

    RegressAndFix(Betas, Residuals, NULL, BestSet,
        bx, y, WeightsArg, nCases, nResp, nMaxTerms);

    if (TraceGlobal >= 6)
        PrintSummary(nMaxTerms, nTerms, nPreds, nResp,
            BestSet, Dirs, Cuts, Betas, nFactorsInTerm);

    int nMaxTerms1 = DiscardUnusedTerms(bx, Dirs, Cuts, BestSet, nFactorsInTerm,
                        nMaxTerms, nPreds, nCases);
    if (Prune)
        BackwardPass(pBestGcv, BestSet, Residuals, Betas,
            bx, y, WeightsArg, nCases, nResp, nMaxTerms1, Penalty);

    if (TraceGlobal >= 6)
        PrintSummary(nMaxTerms, nMaxTerms1, nPreds, nResp,
            BestSet, Dirs, Cuts, Betas, nFactorsInTerm);

    *pnTerms = nMaxTerms1;
    free1(nFactorsInTerm);
    free1(nUses);
}
#endif // STANDALONE

//-----------------------------------------------------------------------------
// Return the max number of knots in any term.
// Lin dep factors are considered as having one knot (at the min value of the predictor)

#if STANDALONE
static int GetMaxKnotsPerTerm(
    const bool   UsedCols[],    // in
    const int    Dirs[],        // in
    const int    nPreds,        // in
    const int    nTerms,        // in
    const int    nMaxTerms)     // in
{
    int nKnotsMax = 0;
    for (int iTerm = 1; iTerm < nTerms; iTerm++)
        if (UsedCols[iTerm]) {
            int nKnots = 0; // number of knots in this term
            for (int iPred = 0; iPred < nPreds; iPred++)
                if (Dirs_(iTerm, iPred) != 0)
                    nKnots++;
            if (nKnots > nKnotsMax)
                nKnotsMax = nKnots;
        }
    return nKnotsMax;
}
#endif // STANDALONE

//-----------------------------------------------------------------------------
// print a string representing the earth expresssion, one term per line
// TODO spacing is not quite right and is overly complicated

#if STANDALONE
static void FormatOneResponse(
    const bool   UsedCols[],// in: nMaxTerms x 1, indices of best set of cols of bx
    const int    Dirs[],    // in: nMaxTerms x nPreds, -1,0,1,2 for iTerm, iPred
    const double Cuts[],    // in: nMaxTerms x nPreds, cut for iTerm, iPred
    const double Betas[],   // in: nMaxTerms x nResp
    const int    nPreds,
    const int    iResp,
    const int    nTerms,
    const int    nMaxTerms,
    const int    nDigits,   // number of significant digits to print
    const double MinBeta)   // terms with fabs(betas) less than this are not printed, 0 for all
{
    int iBestTerm = 0;
    int nKnotsMax = GetMaxKnotsPerTerm(UsedCols, Dirs, nPreds, nTerms, nMaxTerms);
    int nKnots = 0;
    char s[1000];
    ASSERT(nDigits >= 0);
    char sFormat[50];  sprintf(sFormat,  "%%-%d.%dg", nDigits+6, nDigits);
    char sFormat1[50]; sprintf(sFormat1, "%%%d.%dg",  nDigits+6, nDigits);
    int nPredWidth;
    if (nPreds > 100)
        nPredWidth = 3;
    else if (nPreds > 10)
        nPredWidth = 2;
    else
        nPredWidth = 1;
    char sPredFormat[20]; sprintf(sPredFormat, "%%%dd", nPredWidth);
    char sPad[500]; sprintf(sPad, "%*s", 28+nDigits+nPredWidth, " ");    // comment pad
    const int nUsedCols = nTerms;       // nUsedCols is needed for the Betas_ macro
    printf(sFormat, Betas_(0, iResp));  // intercept
    while (nKnots++ < nKnotsMax)
        printf(sPad);
    printf(" // 0\n");

    for (int iTerm = 1; iTerm < nTerms; iTerm++)
        if (UsedCols[iTerm]) {
            iBestTerm++;
            if (fabs(Betas_(iBestTerm, iResp)) >= MinBeta) {
                printf("%+-9.3g", Betas_(iBestTerm, iResp));
                nKnots = 0;
                for (int iPred = 0; iPred < nPreds; iPred++) {
                    switch(Dirs_(iTerm, iPred)) {
                        case  0:
                            break;
                        case -1:
                            sprintf(s, " * max(0, %s - %*sx[%s])",
                                sFormat, nDigits+2, " ", sPredFormat);
                            printf(s, Cuts_(iTerm, iPred), iPred);
                            nKnots++;
                            break;
                        case  1:
                            sprintf(s, " * max(0, x[%s]%*s-  %s)",
                                sPredFormat,  nDigits+2, " ", sFormat1);
                            printf(s, iPred, Cuts_(iTerm, iPred));
                            nKnots++;
                            break;
                        case  2:
                            sprintf(s, " * x[%s]%*s                    ",
                                sPredFormat,  nDigits+2, " ");
                            printf(s, iPred);
                            nKnots++;
                            break;
                        default:
                            ASSERT(false);
                            break;
                    }
                }
                while (nKnots++ < nKnotsMax)
                    printf(sPad);
                printf(" // %d\n", iBestTerm);
            }
        }
}

void FormatEarth(
    const bool   UsedCols[],// in: nMaxTerms x 1, indices of best set of cols of bx
    const int    Dirs[],    // in: nMaxTerms x nPreds, -1,0,1,2 for iTerm, iPred
    const double Cuts[],    // in: nMaxTerms x nPreds, cut for iTerm, iPred
    const double Betas[],   // in: nMaxTerms x nResp
    const int    nPreds,
    const int    nResp,     // in: number of cols in y
    const int    nTerms,
    const int    nMaxTerms,
    const int    nDigits,   // number of significant digits to print
    const double MinBeta)   // terms with fabs(betas) less than this are not printed, 0 for all
{
    for (int iResp = 0; iResp < nResp; iResp++) {
        if (nResp > 1)
            printf("Response %d:\n", iResp+IOFFSET);
        FormatOneResponse(UsedCols, Dirs, Cuts, Betas, nPreds, iResp,
            nTerms, nMaxTerms, nDigits, MinBeta);
    }
}
#endif // STANDALONE

//-----------------------------------------------------------------------------
// return the value predicted by an earth model, given  a vector of inputs x

#if STANDALONE
static double PredictOneResponse(
    const double x[],        // in: vector nPreds x 1 of input values
    const bool   UsedCols[], // in: nMaxTerms x 1, indices of best set of cols of bx
    const int    Dirs[],     // in: nMaxTerms x nPreds, -1,0,1,2 for iTerm, iPred
    const double Cuts[],     // in: nMaxTerms x nPreds, cut for iTerm, iPred
    const double Betas[],    // in: nMaxTerms x 1
    const int    nPreds,     // in: number of cols in x
    const int    nTerms,
    const int    nMaxTerms)
{
    double yHat = Betas[0];
    int iTerm1 = 0;
    for (int iTerm = 1; iTerm < nTerms; iTerm++)
        if (UsedCols[iTerm]) {
            iTerm1++;
            double Term = Betas[iTerm1];
            for (int iPred = 0; iPred < nPreds; iPred++)
                switch(Dirs_(iTerm, iPred)) {
                    case  0: break;
                    case -1: Term *= max(0, Cuts_(iTerm, iPred) - x[iPred]); break;
                    case  1: Term *= max(0, x[iPred] - Cuts_(iTerm, iPred)); break;
                    case  2: Term *= x[iPred]; break;
                    default: ASSERT("bad direction" == NULL); break;
                }
            yHat += Term;
        }
    return yHat;
}

void PredictEarth(
    double       y[],        // out: vector nResp
    const double x[],        // in: vector nPreds x 1 of input values
    const bool   UsedCols[], // in: nMaxTerms x 1, indices of best set of cols of bx
    const int    Dirs[],     // in: nMaxTerms x nPreds, -1,0,1,2 for iTerm, iPred
    const double Cuts[],     // in: nMaxTerms x nPreds, cut for iTerm, iPred
    const double Betas[],    // in: nMaxTerms x nResp
    const int    nPreds,     // in: number of cols in x
    const int    nResp,      // in: number of cols in y
    const int    nTerms,
    const int    nMaxTerms)
{
    for (int iResp = 0; iResp < nResp; iResp++)
        y[iResp] = PredictOneResponse(x, UsedCols, Dirs, Cuts,
                       Betas + iResp * nTerms, nPreds, nTerms, nMaxTerms);
}
#endif // STANDALONE

//-----------------------------------------------------------------------------
// Example main routine
// See earth/src/tests/test.earthc.c for another example

#if STANDALONE
extern "C"{

	void error(const char *args, ...)       // params like printf
	{
		char s[1000];
		va_list p;
		va_start(p, args);
		vsprintf(s, args, p);
		va_end(p);
		printf("\nError: %s\n", s);
		exit(-1);
	}

	void xerbla_(char *srname, int *info)   // needed by BLAS and LAPACK routines
	{
		char buf[7];
		strncpy(buf, srname, 6);
		buf[6] = 0;
		error("BLAS/LAPACK routine %6s gave error code %d", buf, -(*info));
	}

}
#endif


/*
 * Extern interface for ctypes
 */

extern "C" void EarthForwardPass(
	int    *pnTerms,            // out: highest used term number in full model
	bool   FullSet[],           // out: 1 * nMaxTerms, indices of lin indep cols of bx
	double bx[],                // out: MARS basis matrix, nCases * nMaxTerms
	int    Dirs[],              // out: nMaxTerms * nPreds, -1,0,1,2 for iTerm, iPred
	double Cuts[],              // out: nMaxTerms * nPreds, cut for iTerm, iPred
	int    nFactorsInTerm[],    // out: number of hockey stick funcs in each MARS term
	int    nUses[],             // out: nbr of times each predictor is used in the model
	const double x[],           // in: nCases x nPreds
	const double y[],           // in: nCases x nResp
	const double WeightsArg[],  // in: nCases x 1, can be NULL, currently ignored
	const int nCases,           // in: number of rows in x and elements in y
	const int nResp,            // in: number of cols in y
	const int nPreds,           // in:
	const int nMaxDegree,       // in:
	const int nMaxTerms,        // in:
	const double Penalty,       // in: GCV penalty per knot
	double Thresh,              // in: forward step threshold
	int nFastK,                 // in: Fast MARS K
	const double FastBeta,      // in: Fast MARS ageing coef
	const double NewVarPenalty, // in: penalty for adding a new variable (default is 0)
	const int  LinPreds[],      // in: nPreds x 1, 1 if predictor must enter linearly
	const bool UseBetaCache,    // in: true to use the beta cache, for speed
	const char *sPredNames[])   // in: predictor names, can be NULL
{
	ForwardPass(pnTerms, FullSet, bx, Dirs, Cuts, nFactorsInTerm, nUses,
			x, y, WeightsArg, nCases, nResp, nPreds, nMaxDegree,
			nMaxTerms, Penalty, Thresh, nFastK, FastBeta, NewVarPenalty,
			LinPreds, UseBetaCache, sPredNames);
}

extern "C" int EarthEvalSubsetsUsingXtx(
	bool   PruneTerms[],    // out: nMaxTerms x nMaxTerms
	double RssVec[],        // out: nMaxTerms x 1, RSS of each subset
	const int    nCases,    // in
	const int    nResp,     // in: number of cols in y
	const int    nMaxTerms, // in: number of MARS terms in full model
	const double bx[],      // in: nCases x nMaxTerms, all cols must be indep
	const double y[],       // in: nCases * nResp
	const double WeightsArg[]) // in: nCases x 1, can be NULL
{
	return EvalSubsetsUsingXtx(PruneTerms, RssVec, nCases, nResp, nMaxTerms, bx, y, WeightsArg);
}

/*
 * ORANGE INTERFACE
 */

TEarthLearner::TEarthLearner()
{
	max_terms = 21;
	max_degree = 1;
	penalty = (max_degree > 1)? 3.0: 2.0;
	threshold = 0.001;
	prune = true;
	trace = 0.0;
	min_span = 0;
	fast_k = 20;
	fast_beta = 0.0;
    new_var_penalty = 0.0;
	use_beta_cache = true;
}

PClassifier TEarthLearner::operator() (PExampleGenerator examples, const int & weight_id)
{
	TDomain& domain = examples->domain.getReference();
	int num_preds = domain.attributes->size();
	int num_cases = examples->numberOfExamples();
	int num_responses = 1;
	if (num_cases < 0){
		raiseError("Cannot learn from an example generator of unknown size.");
	}

	// TODO: Check for classVar, assert all attributes are continuous

//	num_preds = 1;
//	num_cases = 100;

	double best_gcv;
	int num_terms;

	double *x = (double *) calloc(num_preds * num_cases, sizeof(double));
	double *y = (double *) calloc(num_cases * num_responses, sizeof(double));
	double *bx = (double *) calloc(num_cases * max_terms, sizeof(double));
	bool *best_set = (bool *) calloc(max_terms, sizeof(bool));
	int *dirs = (int *) calloc(max_terms * num_preds, sizeof(int));
	double *cuts = (double *) calloc(max_terms * num_preds, sizeof(double));
	double *residuals = (double *) calloc(num_cases * num_responses, sizeof(double));
	double *betas = (double *) calloc(max_terms * num_responses, sizeof(double));
	int * lin_preds = (int *) calloc(num_preds, sizeof(int));
	double *weights = NULL;

	// Redefine x indexing
	#undef x_
	#define x_(i, j) x[i + j * num_cases]

	TExampleGenerator::iterator ex_iter = examples->begin();
	for (int i=0; i<num_cases; i++, ++ex_iter)
	{
		TExample &example = *ex_iter;
		for (int j=0; j<num_preds; j++)
		{
			double tempx;
			TValue &value = example[j];
			if (value.isSpecial())
				tempx = 0.0;
			else
				if (value.varType == TValue::INTVAR)
					tempx = (double) value.intV;
				else
					tempx = (double) value.floatV;
			x_(i, j) = tempx;
		}
		double tempy;
		TValue &class_value = example.getClass();
		if (class_value.varType == TValue::INTVAR)
			tempy = (double) class_value.intV;
		else
			tempy = (double) class_value.floatV;
		y[i] = tempy;
	}
//	for (int i = 0; i < num_cases; i++) {
//	        const double x0 = (double)i / num_cases;
//	        x[i] = x0;
//	        y[i] = sin(4 * x0);     // target function, change this to whatever you want
//	    }


	const char **preds_names = NULL; // Used for trace only.
//	preds_names = (char **) malloc(num_preds * sizeof(char *));
//	for (int i=0; i<num_preds; i++){
//		preds_names[i] = NULL;
//	}

	Earth(&best_gcv, &num_terms, best_set, bx, dirs, cuts, residuals, betas,
			x, y, weights, num_cases, num_responses, num_preds, max_degree,
			max_terms, penalty, threshold, min_span, prune,
			fast_k, fast_beta, new_var_penalty, lin_preds, use_beta_cache, trace, preds_names);

	PEarthClassifier classifier = mlnew TEarthClassifier(examples->domain, best_set, dirs, cuts, betas, num_preds, num_responses, num_terms, max_terms);
//	std::string str = classifier->format_earth();

	// Free memory
	free((void *)x);
	free((void *)y);
	free((void *)bx);
	free((void *)residuals);
//	free((void *)weights);
	free((void *)lin_preds);

	return classifier;
}

TEarthClassifier::TEarthClassifier(PDomain _domain, bool * best_set, int * dirs, double * cuts, double *betas, int _num_preds, int _num_responses, int _num_terms, int _max_terms)
{
	domain = _domain;
	classVar = domain->classVar;
	_best_set = best_set;
	_dirs = dirs;
	_cuts = cuts;
	_betas = betas;
	num_preds = _num_preds;
	num_responses = _num_responses;
	num_terms = _num_terms;
	max_terms = _max_terms;
	computesProbabilities = false;
	init_members();
}

TEarthClassifier::TEarthClassifier()
{
	domain = NULL;
	classVar = NULL;
	_best_set = NULL;
	_dirs = NULL;
	_cuts = NULL;
	_betas = NULL;
	num_preds = 0;
	num_responses = 0;
	num_terms = 0;
	max_terms = 0;
	computesProbabilities = false;
}

TEarthClassifier::TEarthClassifier(const TEarthClassifier & other)
{
	raiseError("Not implemented");
}

TEarthClassifier::~TEarthClassifier()
{
	if (_best_set)
		free(_best_set);
	if (_dirs)
		free(_dirs);
	if (_cuts)
		free(_cuts);
	if (_betas)
		free(_betas);
}

TValue TEarthClassifier::operator()(const TExample& example)
{
	double *x = to_xvector(example);
	double y = 0.0;
	PredictEarth(&y, x, _best_set, _dirs, _cuts, _betas, num_preds, num_responses, num_terms, max_terms);
	free(x);
	if (classVar->varType == TValue::INTVAR)
		return TValue((int) std::max<float>(0.0, floor(y + 0.5)));
	else
		return TValue((float) y);
}

std::string TEarthClassifier::format_earth(){
	FormatEarth(_best_set, _dirs, _cuts, _betas, num_preds, 1, num_terms, max_terms, 3, 0.0);
	// TODO: FormatEarth to a string.
	return "";
}

double* TEarthClassifier::to_xvector(const TExample& example)
{
//	TAttributeList &attributes = example.domain->attributes.getReference();
	double *x = (double *) calloc(num_preds, sizeof(double));
	for (int i=0; i<num_preds; i++){
		const TValue &val = example[i];
		if (val.isSpecial())
			x[i] = 0.0;
		else
			if (val.varType == TValue::INTVAR)
				x[i] = (double) val.intV;
			else
				x[i] = (double) val.floatV;
	}
	return x;
}

PBoolList TEarthClassifier::get_best_set()
{
	PBoolList list = mlnew TBoolList();
	for (bool * p=_best_set; p < _best_set + max_terms; p++)
		 list->push_back(*p);
	return list;
}

PFloatListList TEarthClassifier::get_dirs()
{
	PFloatListList list = mlnew TFloatListList();
	for (int i=0; i<max_terms; i++)
	{
		TFloatList * inner_list = mlnew TFloatList();
		for(int j=0; j<num_preds; j++)
			inner_list->push_back(_dirs[i + j*max_terms]);
		list->push_back(inner_list);
	}
	return list;
}

PFloatListList TEarthClassifier::get_cuts()
{
	PFloatListList list = mlnew TFloatListList();
	for (int i=0; i<max_terms; i++)
	{
		TFloatList * inner_list = mlnew TFloatList();
		for (int j=0; j<num_preds; j++)
			inner_list->push_back(_cuts[i + j*max_terms]);
		list->push_back(inner_list);
	}
	return list;
}

PFloatList TEarthClassifier::get_betas()
{
	PFloatList list = mlnew TFloatList();
	for (double * p=_betas; p < _betas + max_terms; p++)
		list->push_back((float)*p);
	return list;
}

void TEarthClassifier::init_members()
{
	best_set = get_best_set();
	dirs = get_dirs();
	cuts = get_cuts();
	betas = get_betas();

}

void TEarthClassifier::save_model(TCharBuffer& buffer)
{
	buffer.writeInt(max_terms);
	buffer.writeInt(num_terms);
	buffer.writeInt(num_preds);
	buffer.writeInt(num_responses);
	buffer.writeBuf((void *) _best_set, sizeof(bool) * max_terms);
	buffer.writeBuf((void *) _dirs, sizeof(int) * max_terms * num_preds);
	buffer.writeBuf((void *) _cuts, sizeof(double) * max_terms * num_preds);
	buffer.writeBuf((void *) _betas, sizeof(double) * max_terms * num_responses);
}

void TEarthClassifier::load_model(TCharBuffer& buffer)
{
	if (max_terms)
		raiseError("Cannot overwrite a model");

	max_terms = buffer.readInt();
	num_terms = buffer.readInt();
	num_preds = buffer.readInt();
	num_responses = buffer.readInt();

	_best_set = (bool *) calloc(max_terms, sizeof(bool));
	_dirs = (int *) calloc(max_terms * num_preds, sizeof(int));
	_cuts = (double *) calloc(max_terms * num_preds, sizeof(double));
	_betas = (double *) calloc(max_terms * num_responses, sizeof(double));

	buffer.readBuf((void *) _best_set, sizeof(bool) * max_terms);
	buffer.readBuf((void *) _dirs, sizeof(int) * max_terms * num_preds);
	buffer.readBuf((void *) _cuts, sizeof(double) * max_terms * num_preds);
	buffer.readBuf((void *) _betas, sizeof(double) * max_terms * num_responses);
	init_members();
}



