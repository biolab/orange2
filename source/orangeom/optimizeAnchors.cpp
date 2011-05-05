/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
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


#include "orange_api.hpp"
#include "examplegen.hpp"
#include "symmatrix.hpp"
#include "pnn.hpp"
#include "../orange/px/externs.px"

inline double sqr(const double &x)
{ return x*x; }

#define xfree(x) { if (x) free(x); }

typedef struct {double x, y; } TPoint;


/* loadRadvizData converts the Python lists and matrices into C++ structures.
                  It can choose a subset of attributes.
                  For discrete classes it also sorts the examples by the class value.

   INPUT:
     scaledData:    scaledData from the graph (continuous attribute values - no missing!)
     pyclasses:     class for each example in 'scaledData', or SymMatrix
     anchors:       anchor positions (Python list of lists with 2 or 3 elements)
     pyattrIndices: indices of attributes to be used
     contClass      0=discrete, 1=continuous, 2=no class, MDS

     scaledData and pyclasses must be of same length, and anchors and pyattrIndices also.
     scaledData can have more attributes than the length of anchors; pyattrIndices says which are chosen

   OUTPUT:
     nExamples      number of examples (length of 'scaledData'
     nAttrs         number of attributes (length of 'anchors')
     X              a list of attribute values (one-dimensional - flattened two-dim)
                    - contains only the chosen attributes
                    - for discrete classes, the list is sorted by class values
     classes        for discrete classes, this is an (int *) with indices of class groups in X;
                       therefore, the length of classes equals the number of classes+1
                    for continuous, it contains the class values
                    for MDS, it will return a pointer to symmatrix
                    IMPORTANT: 'classes' should be freed by the caller for discrete and continuous,
                               but mustn't be freed for MDS
     anc            anchor coordinates
     ll             anchor labels (stored for when the anchor list needs to be reconstructed)
     minClass       the minimal value encountered (for continuous only)
     maxClass       for continuous classes it is the maximal value, for discrete it is the number of classes-1
*/

bool loadRadvizData(PyObject *scaledData, PyObject *pyclasses, PyObject *anchors, PyObject *pyattrIndices, int &contClass,
                    int &nAttrs, int &nExamples,
                    double *&X, int *&classes, TPoint *&anc, PyObject **&ll,
                    double &minClass, double &maxClass)
{
  if (!PyList_Check(scaledData) || !PyList_Check(anchors))
    PYERROR(PyExc_TypeError, "scaled data and anchors should be given as lists", false);

  if (contClass < 2) {
    if (!PyList_Check(pyclasses))
      PYERROR(PyExc_TypeError, "classes should be given as a list", false);

    if (PyList_Size(scaledData) != PyList_Size(pyclasses))
      PYERROR(PyExc_TypeError, "'scaledData' and 'classes' have different lengths", false);
  }

  else {
    if (!PyOrSymMatrix_Check(pyclasses))
      PYERROR(PyExc_TypeError, "distance matrix should be given as a SymMatrix", false);

    TSymMatrix *&distances = (TSymMatrix *&)classes;
    distances = PyOrange_AsSymMatrix(pyclasses).getUnwrappedPtr();
    if (distances->dim != PyList_Size(scaledData))
      PYERROR(PyExc_TypeError, "the number of examples mismatches the distance matrix size", false);
  }

  if (PyList_Size(anchors) != PyList_Size(pyattrIndices))
    PYERROR(PyExc_TypeError, "'anchors' and 'attrIndices' have different lengths", false);

  nAttrs = PyList_Size(anchors);
  nExamples = PyList_Size(scaledData);

  X = (double *)malloc(nExamples * nAttrs * sizeof(double));
  anc = (TPoint *)malloc(nAttrs * sizeof(TPoint));
  ll = (PyObject **)malloc(nAttrs * sizeof(PyObject *));

  if (contClass < 2)
    classes = (int *)malloc(nExamples * (contClass ? sizeof(double) : sizeof(int)));

  // indices of the chosen attributes
  int *aii, *attrIndices = (int *)malloc(nAttrs * sizeof(int)), *aie = attrIndices + nAttrs;
  TPoint *anci;
  PyObject **lli;
  double *Xi;
  int i;
   
  for(anci = anc, aii = attrIndices, lli = ll, i = 0; i < nAttrs; i++, anci++, aii++, lli++) {
    *lli = NULL;
    PyArg_ParseTuple(PyList_GetItem(anchors, i), "dd|O", &anci->x, &anci->y, lli);
    *aii = PyInt_AsLong(PyList_GetItem(pyattrIndices, i));
  }


  if (contClass == 0) {
    // read the classes
    int *classesi, *classese;
    int maxCls = 0;
    for(classesi = classes, i = 0; i < nExamples; classesi++, i++) {
      *classesi = PyInt_AsLong(PyList_GetItem(pyclasses, i));
      if (*classesi > maxCls)
        maxCls = *classesi;
    }

    // prepare the indices for counting sort algorithm
    //   (we need maxCls+3: beginning of each of maxCls+1 classes, end of the last class, sentinel)
    int *rcls = (int *)malloc((maxCls+3) * sizeof(int));
    memset(rcls, 0, (maxCls+3) * sizeof(int));
    for(classesi = classes, classese = classes+nExamples; classesi != classese; rcls[1 + *classesi++]++);
    for(int *rclsi = rcls+1, *rclse = rcls+maxCls+2; rclsi != rclse; *rclsi += rclsi[-1], rclsi++);

    // read the examples and sort them by classes at the same time (put them at the correct indices)
    for(classesi = classes, i = 0; i < nExamples; i++, classesi++) {
      PyObject *ex = PyList_GetItem(scaledData, i);
      Xi = X + nAttrs * rcls[*classesi]++;
      for(aii = attrIndices; aii < aie; aii++)
        *Xi++ = PyFloat_AsDouble(PyList_GetItem(ex, *aii));
    }

    // shift the class indices to the right, insert 0 at the beginning
    memmove(rcls+1, rcls, (maxCls+1) * sizeof(int));
    *rcls = 0;

    free(classes);
    classes = rcls;
  }


  else if (contClass == 1) {
    // read the classes
    double *dclassesi;
    for(dclassesi = (double *)classes, i = 0; i < nExamples; *dclassesi++ = PyFloat_AsDouble(PyList_GetItem(pyclasses, i++)));

    // read the attribute values
    for(Xi = X, i = 0; i < nExamples; i++) {
      PyObject *ex = PyList_GetItem(scaledData, i);
      for(aii = attrIndices; aii < aie; aii++)
        *Xi++ = PyFloat_AsDouble(PyList_GetItem(ex, *aii));
    }
  }

  // nothing to do if contClass == 2 - we have already set the classes to symmatrix above

  // we don't need and don't return this
  free(attrIndices);

  return true;
}


/* Given coordinates of anchors and the symmetry type, it rotates the anchors
   so that the first lies at phi=0 and the second (if symmetry==2) is on the
   upper half-plane */

void symmetricTransformation(TPoint *anc, TPoint *ance, bool mirrorSymmetry)
{
   const double phi = atan2(anc[0].y, anc[0].x);
   const double phi2 = atan2(anc[1].y, anc[1].x);

   const int sign = mirrorSymmetry && ((phi2<phi) || (phi2-phi > 3.1419265)) ? -1 : 1;
   const double dphi = /*3.1419265/2.0*/ - phi;
   const double cs = cos(dphi), sn = sin(dphi);

   for(TPoint *anci = anc; anci != ance; anci++) {
     anci->y = sign * (anci->x * sn + anci->y * cs);
     anci->x =        (anci->x * cs - anci->y * sn);
   }
}


/* Computes forces for continuous class

   INPUT:
     pts           projections of examples
     classes       example classes
     nExamples     number of examples (the length of above arrays and of Fr)
     law           0=Linear, 1=Square, 2=Gaussian
     sigma2        sigma**2 for Gaussian law

   OUTPUT:
     F            forces acting on each example (memory should be allocated by the caller!)
*/

void computeForcesContinuous(TPoint *pts, const TPoint *ptse, const double *classes, 
                             const int &law, const double &sigma2,
                             TPoint *F)
{
  TPoint *Fi, *Fi2, *ptsi, *ptsi2;
  const double *classesi, *classesi2;

  for(ptsi = pts, Fi = F, classesi = classes; ptsi != ptse; ptsi++, Fi++, classesi++) {
    Fi->x = Fi-> y = 0.0;
    for(ptsi2 = pts, Fi2 = F, classesi2 = classes; ptsi2 != ptsi; ptsi2++, Fi2++, classesi2++) {
      const double dx = ptsi->x - ptsi2->x;
      const double dy = ptsi->y - ptsi2->y;
      double r2 = sqr(dx) + sqr(dy);
      if (r2 < 1e-20)
        continue;

      double fct = sqr(*classesi-*classesi2);
      switch (law) {
        case TPNN::InverseLinear:
          fct /= r2;
          break;
        case TPNN::InverseSquare:
          fct /=  (r2 * sqrt(r2));
          break;
        case TPNN::InverseExponential:
          fct /=  (exp(r2/sigma2) - 1);
      }

      const double druvx = dx * fct;
      Fi->x  += druvx;
      Fi2->x -= druvx;

      const double druvy = dy * fct;
      Fi->y  += druvy;
      Fi2->y -= druvy;
    }
  }
}


/* Computes forces for discrete class

   INPUT:
     pts               projections of examples
     classes           example classes
     nExamples         number of examples (the length of above arrays and of Fr)

     law               0=Linear, 1=Square, 2=Gaussian
     sigma2            sigma**2 for Gaussian law

     attractG          the factor to multiply the attractive forces with
     repelG            the factor to multiply the repulsive forces with
     dynamicBalancing  if true, the forces are balanced (prior to multiplying with the above factors
                          so that the total sum of the attractive equals the sum of repulsive)

     Comments:
       Fa is used to return the forces, but should be allocated by the caller
       Fr is used as a temporary, but should be allocated by the caller if both types of forces are used
       attractG, repelG, dynamicBalancing are used only if both types of forces are used


   OUTPUT:
     Fa                forces acting on each example (memory should be allocated by the caller!)
*/

void computeForcesDiscrete(TPoint *pts, const TPoint *ptse, const int *classes,
                           int law, const double &sigma2, const double &attractG, const double &repelG, const bool dynamicBalancing,
                           TPoint *Fa, TPoint *Fr
                          )
{
  TPoint *Fai, *Fri, *Fri2, *Fai2, *Fe, *ptsi, *ptsie, *ptsi2;

  if (attractG == 0.0)
    Fr = Fa; // if we have only repulsive forces, we can compute them directly into Fa
  else if (repelG)
    for(Fri = Fr, Fe = Fr + (ptse-pts); Fri != Fe; Fri++)
      Fri->x = Fri->y = 0.0;


  for(const int *classesi = classes; classesi[1]; classesi++) {

    /**** Attractive forces ****/

    if (attractG != 0.0) {

      if ((law == TPNN::InverseLinear) || (law == TPNN::InverseExponential) || (law == TPNN::Linear)) {
        double sumx = 0, sumy = 0;
        const double n = classesi[1] - *classesi;
        for(ptsi = pts + *classesi, ptsie = pts + classesi[1]; ptsi != ptsie; ptsi++) {
          sumx += ptsi->x;
          sumy += ptsi->y;
        }

        for(ptsi = pts + *classesi, ptsie = pts + classesi[1], Fai = Fa + *classesi; ptsi != ptsie; ptsi++, Fai++) {
          Fai->x = sumx - n * ptsi->x;
          Fai->y = sumy - n * ptsi->y;
        }
      }

      else {
        for(ptsi = pts + *classesi, ptsie = pts + classesi[1], Fai = Fa + *classesi; ptsi != ptsie; ptsi++, Fai++) {
          Fai->x = Fai-> y = 0.0;
          for(ptsi2 = pts + *classesi,                         Fai2 = Fa + *classesi; ptsi2 != ptsi; ptsi2++, Fai2++) {
            const double dx = ptsi->x - ptsi2->x;
            const double dy = ptsi->y - ptsi2->y;

            double fct;
            if (law == TPNN::InverseSquare)
              fct = - sqrt(sqr(dx) + sqr(dy));
            else {
              const double r2 = sqr(dx) + sqr(dy);
              fct = - sqrt(r2) * exp(-r2/sigma2);
            }

            const double druvx = dx * fct;
            Fai->x  += druvx;
            Fai2->x -= druvx;

            const double druvy = dy * fct;
            Fai->y  += druvy;
            Fai2->y -= druvy;
          }
        }
      }
    }


    /**** Repulsive forces ****/

    if ((repelG != 0.0) && classesi[2]) {

      for(ptsi = pts + *classesi, ptsie = pts + classesi[1], Fri = Fr + *classesi; ptsi != ptsie; ptsi++, Fri++) {
        for(ptsi2 = ptsie, Fri2 = Fr + classesi[1]; ptsi2 != ptse; ptsi2++, Fri2++) {
          const double dx = ptsi->x - ptsi2->x;
          const double dy = ptsi->y - ptsi2->y;
          double r2 = sqr(dx) + sqr(dy);
          if (r2 < 1e-20)
            continue;

          double fct;
          switch (law) {
            case TPNN::InverseLinear:
              fct = 1 / r2;
              break;
            case TPNN::Linear:
              fct = 1;
              break;
            case TPNN::InverseSquare:
              fct = 1 / (r2 * sqrt(r2));
              break;
            case TPNN::InverseExponential:
              fct = 1 / (exp(r2/sigma2) - 1);
              break;
            case TPNN::KNN:
              fct = sqrt(r2) * exp(-r2/sigma2);
              break;
          }

          const double druvx = dx * fct;
          Fri->x  += druvx;
          Fri2->x -= druvx;

          const double druvy = dy * fct;
          Fri->y  += druvy;
          Fri2->y -= druvy;
        }
      }
    }
  }
  
  // if both types of forces are used, balance them and mix them into Fa in the right proportions
  if ((repelG != 0.0) && (attractG != 0.0)) {
    double repelGk;

    if (dynamicBalancing) {
      double FrTot = 0;
      for(Fri = Fr, Fe = Fr + (ptse-pts); Fri != Fe; Fri++)
        FrTot += sqr(Fri->x) + sqr(Fri->y);

      double FaTot = 0;
      for(Fai = Fa, Fe = Fa + (ptse-pts); Fai != Fe; Fai++)
        FaTot += sqr(Fai->x) + sqr(Fai->y);

      repelGk = FrTot > 0.001 ? repelG * fabs(FaTot / FrTot) : repelG;
    }
    else
      repelGk = repelG;

    for(Fai = Fa, Fri = Fr, Fe = Fr + (ptse-pts); Fri != Fe; Fai++, Fri++) {
      Fai->x = attractG * Fai->x  +  repelGk * Fri->x;
      Fai->y = attractG * Fai->y  +  repelGk * Fri->y;
    }
  }
}






/* Computes forces for MDS-like FreeViz
   ALWAYS OPTIMIZES THE CLASSIC MDS STRESS REGARDLESS OF LAW

   INPUT:
     pts           projections of examples
     distances     a distance matrix
     nExamples     number of examples (the length of above arrays and of Fr)
     law           0=Linear, 1=Square, 2=Gaussian
     sigma2        sigma**2 for Gaussian law

   OUTPUT:
     F            forces acting on each example (memory should be allocated by the caller!)
*/

void computeForcesMDS(TPoint *pts, const TPoint *ptse, TSymMatrix *distances, 
                      const int &law, const double &sigma2,
                      TPoint *F)
{
  TPoint *Fi, *Fi2, *ptsi, *ptsi2;
  float *edistances = distances->elements;

  for(ptsi = pts, Fi = F; ptsi != ptse; ptsi++, Fi++, edistances++ /* skip diagonal */) {
    Fi->x = Fi-> y = 0.0;
    for(ptsi2 = pts, Fi2 = F; ptsi2 != ptsi; ptsi2++, edistances++, Fi2++) {

      const double dx = ptsi->x - ptsi2->x;
      const double dy = ptsi->y - ptsi2->y;
      double r2 = sqr(dx) + sqr(dy);
      if (r2 < 1e-20)
        continue;

      double fct;
      const double r = sqrt(r2);
      fct = (*edistances - r) / (*edistances > 1e-3 ? sqr(*edistances) : 1e-3);

      const double druvx = dx * fct;
      Fi->x  += druvx;
      Fi2->x -= druvx;

      const double druvy = dy * fct;
      Fi->y  += druvy;
      Fi2->y -= druvy;
    }
  }
}


     
PyObject *optimizeAnchors(PyObject *, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "(scaledData, classes, anchors[, attractG=1.0, repelG=-1.0, law=InverseLinear, steps=1, normalizeExamples=1]) -> new-anchors")
{
  PyTRY
    // get the arguments from Python
    PyObject *scaledData, *pyclasses, *anchors, *pyattrIndices;
    double attractG = 1.0,              repelG = -1.0,          sigma2 = 1.0;
    int    law = TPNN::InverseLinear,   steps = 1,              normalizeExamples = 0,
           contClass = 0,               dynamicBalancing = 0,   mirrorSymmetry = 0;

    static char *kwlist[] = {"scaledData", "classes", "anchors", "attrIndices", "attractG", "repelG", "law", "sigma2", "dynamicBalancing", "steps", "normalizeExamples", "contClass", "mirrorSymmetry", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "OOOO|ddidiiiii:optimizeAnchors", kwlist, &scaledData, &pyclasses, &anchors, &pyattrIndices, &attractG, &repelG, &law, &sigma2, &dynamicBalancing, &steps, &normalizeExamples, &contClass, &mirrorSymmetry))
      return NULL;


    double *Xi, *X;            // values of the chosen attributes
    int *classes;              // classes (for continuous) or indices for groups of classes (for discrete);
    int nAttrs, nExamples;     // number of (chosen) attributes and of examples
    TPoint *anci, *anc, *ance; // anchor coordinates
    PyObject **lli, **ll;      // anchor labels
    double minClass, maxClass; // minimal and maximal class values (for cont), #classes+1 (for disc)

    // convert the examples, classes and anchors from Python lists
    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, contClass, nAttrs, nExamples, X, classes, anc, ll, minClass, maxClass))
      return PYNULL;
    ance = anc + nAttrs;

    int i;
    TPoint *danci, *danc = (TPoint *)malloc(nAttrs * sizeof(TPoint)), *dance = danc + nAttrs;   // anchors' moves
    TPoint *ptsi, *pts = (TPoint *)malloc(nExamples * sizeof(TPoint)), *ptse = pts + nExamples; // projections of examples
    TPoint *Fai, *Fa = (TPoint *)malloc(nExamples * sizeof(TPoint)), *Fae = Fa + nExamples;     // forces on examples

    TPoint *Fr = !contClass && (attractG != 0.0) && (repelG != 0.0) 
                     ? (TPoint *)malloc(nExamples * sizeof(TPoint)) : NULL;    // temporary array for computeForces

    double *radi, *rad = NULL, *sumi, *sum = NULL, *sume;
    if (normalizeExamples) {
      rad = (double *)malloc(nAttrs * sizeof(double));                         // radii of anchors
      sum = (double *)malloc(nExamples * sizeof(double));                      // sums of attr values for each example
      sume = sum + nExamples;
    }


    while (steps--) {
      // compute the projection
      if (normalizeExamples) {
        for(anci = anc, radi = rad; anci != ance; anci++, radi++)
          *radi = sqrt(sqr(anci->x) + sqr(anci->y));

        for(sumi = sum, Xi = X, ptsi = pts; ptsi != ptse; sumi++, ptsi++) {
          ptsi->x = ptsi->y = *sumi = 0.0;
          for(anci = anc, radi = rad; anci != ance; anci++, Xi++, radi++) {
            ptsi->x += *Xi * anci->x;
            ptsi->y += *Xi * anci->y;
            *sumi += *Xi * *radi;
          }
          if (fabs(*sumi) > 1e-6) {
            ptsi->x /= *sumi;
            ptsi->y /= *sumi;
          }
          else
            *sumi = 1.0; // we also use *sumi later
        }
      }
      else {
        for(Xi = X, ptsi = pts; ptsi != ptse; ptsi++) {
          ptsi->x = ptsi->y = 0.0;
          for(anci = anc; anci != ance; anci++, Xi++) {
            ptsi->x += *Xi * anci->x;
            ptsi->y += *Xi * anci->y;
          }
        }
      }


      switch (contClass) {
        case 0:
          computeForcesDiscrete(pts, ptse, classes, law, sigma2, attractG, repelG, dynamicBalancing != 0, Fa, Fr);
          break;
        case 1:
          computeForcesContinuous(pts, ptse, (double *)classes, law, sigma2, Fa);
          break;
        case 2:
          computeForcesMDS(pts, ptse, (TSymMatrix *)classes, law, sigma2, Fa);
          break;
      };

      // Normalize forces if needed (why?! instead of dividing each *Xi?)
      if (normalizeExamples)
        for(Fai = Fa, sumi = sum; Fai != Fae; Fai++, sumi++) {
          Fai->x /= *sumi;
          Fai->y /= *sumi;
        }


      // Transmit forces on particles to the anchors
      for(danci = danc; danci != dance; danci++)
        danci->x = danci->y = 0.0;

      for(Fai = Fa, Xi = X; Fai != Fae; Fai++) {            // loop over examples
        for(danci = danc; danci != dance; danci++, Xi++) {  // loop over anchors
          danci->x += Fai->x * *Xi;
          danci->y += Fai->y * *Xi;
        }
      }


      // Scale the changes (the largest is anchor move is 0.1*radius)
      double scaling = 1e10;
      for(anci = anc, danci = danc; danci != dance; anci++, danci++) {
        double maxdr = 0.01 * (sqr(anci->x) + sqr(anci->y));
        double dr = sqr(danci->x) + sqr(danci->y);
        if (scaling * dr > maxdr)
          scaling = maxdr / dr;
      }

      scaling = sqrt(scaling);
      for(danci = danc; danci != dance; danci++) {
        danci->x *= scaling;
        danci->y *= scaling;
      }


      // Move anchors
      for(anci = anc, danci = danc; danci != dance; danci++, anci++) {
        anci->x +=  danci->x;
        anci->y +=  danci->y;
      }

  
      // Center anchors (so that the average is in the middle)
      double aax = 0.0, aay = 0.0;
      for(anci = anc; anci != ance; anci++) {
        aax += anci->x;
        aay += anci->y;
      }
      aax /= nAttrs ? nAttrs : 1;
      aay /= nAttrs ? nAttrs : 1;

      for(anci = anc; anci != ance; anci++) {
        anci->x -= aax;
        anci->y -= aay;
      }


      // Scale (so that the largest radius is 1)
      double maxr = 0.0;
      for(anci = anc; anci != ance; anci++) {
        const double r = sqr(anci->x) + sqr(anci->y);
        if (r > maxr)
          maxr = r;
      }

      if (maxr > 0.001) {
        maxr = sqrt(maxr);
        for(anci = anc; anci != ance; anci++) {
          anci->x /= maxr;
          anci->y /= maxr;
        }
      }
    }

    symmetricTransformation(anc, ance, mirrorSymmetry != 0);

    anchors = PyList_New(nAttrs);
    for(i = 0, anci = anc, lli = ll;i < nAttrs; lli++, i++, anci++)
      PyList_SetItem(anchors, i, *lli ? Py_BuildValue("ddO", anci->x, anci->y, *lli) : Py_BuildValue("dd", anci->x, anci->y));
      
    free(X);
    free(anc);
    free(ll);
    if (contClass < 2)
      free(classes);

    free(danc);
    free(pts);
    free(Fa);
    xfree(Fr);
    xfree(sum);
    xfree(rad);

    return anchors;
      
  PyCATCH;
}



PyObject *optimizeAnchorsRadial(PyObject *, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "(scaledData, classes, anchors[, attractG=1.0, repelG=-1.0, law=InverseLinear, steps=1, normalizeExamples=1]) -> new-anchors")
{
  PyTRY
    // get the arguments from Python
    PyObject *scaledData, *pyclasses, *anchors, *pyattrIndices;
    double attractG = 1.0,              repelG = -1.0,          sigma2 = 1.0;
    int law = TPNN::InverseLinear,      steps = 1,              normalizeExamples = 0,
        contClass = 0,                  dynamicBalancing = 0,   mirrorSymmetry = 0;

    static char *kwlist[] = {"scaledData", "classes", "anchors", "attrIndices", "attractG", "repelG", "law", "sigma2", "dynamicBalancing", "steps", "normalizeExamples", "contClass", "mirrorSymmetry", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "OOOO|ddidiiiii:optimizeAnchors", kwlist, &scaledData, &pyclasses, &anchors, &pyattrIndices, &attractG, &repelG, &law, &sigma2, &dynamicBalancing, &steps, &normalizeExamples, &contClass, &mirrorSymmetry))
      return NULL;

    double *Xi, *X;            // values of the chosen attributes
    int *classes;              // classes (for continuous) or indices for groups of classes (for discrete);
    TPoint *anci, *anc, *ance; // anchor coordinates
    int nAttrs, nExamples;     // number of (chosen) attributes and of examples
    PyObject **lli, **ll;      // anchor labels
    double minClass, maxClass; // minimal and maximal class values (for cont), #classes+1 (for disc)

    // convert the examples, classes and anchors from Python lists
    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, contClass, nAttrs, nExamples, X, classes, anc, ll, minClass, maxClass))
      return PYNULL;
    ance = anc + nAttrs;

    int i;
    double *dphii, *dphi = (double *)malloc(nAttrs * sizeof(double)), *dphie = dphi + nAttrs;    // changes of angles of anchors
    TPoint *ptsi, *pts = (TPoint *)malloc(nExamples * sizeof(TPoint)), *ptse = pts + nExamples;  // projections of examples
    TPoint *Fai, *Fa = (TPoint *)malloc(nExamples * sizeof(TPoint)), *Fae = Fa + nExamples;      // forces on examples

    TPoint *Fr = !contClass && (attractG != 0.0) && (repelG != 0.0) 
                     ? (TPoint *)malloc(nExamples * sizeof(TPoint)) : NULL;    // temporary array for computeForces

    double *sumi, *sum = NULL, *sume;
    if (normalizeExamples) {
      sum = (double *)malloc(nExamples * sizeof(double));                      // sums of attr values for each example
      sume = sum + nExamples;
    }

    while (steps--) {
      // compute the projection
      if (normalizeExamples) {
        for(sumi = sum, Xi = X, ptsi = pts; ptsi != ptse; sumi++, ptsi++) {
          ptsi->x = ptsi->y = *sumi = 0.0;
          for(anci = anc; anci != ance; anci++, Xi++) {
            ptsi->x += *Xi * anci->x;
            ptsi->y += *Xi * anci->y;
            *sumi += *Xi;
          }
          if (*sumi != 0.0) {
            ptsi->x /= *sumi;
            ptsi->y /= *sumi;
          }
          else
            *sumi = 1.0; // we also use *sumi later
        }
      }
      else {
        for(Xi = X, ptsi = pts; ptsi != ptse; ptsi++) {
          ptsi->x = ptsi->y = 0.0;
          for(anci = anc; anci != ance; anci++, Xi++) {
            ptsi->x += *Xi * anci->x;
            ptsi->y += *Xi * anci->y;
          }
        }
      }

      switch (contClass) {
        case 0:
          computeForcesDiscrete(pts, ptse, classes, law, sigma2, attractG, repelG, dynamicBalancing != 0, Fa, Fr);
          break;
        case 1:
          computeForcesContinuous(pts, ptse, (double *)classes, law, sigma2, Fa);
          break;
        case 2:
          computeForcesMDS(pts, ptse, (TSymMatrix *)classes, law, sigma2, Fa);
          break;
      };

      // Normalize forces if needed (why?! instead of dividing each *Xi?)
      if (normalizeExamples)
        for(Fai = Fa, sumi = sum; Fai != Fae; Fai++, sumi++) {
          Fai->x /= *sumi;
          Fai->y /= *sumi;
        }


      // Transmit the forces to the anchors
      for(dphii = dphi; dphii != dphie; *dphii++ = 0.0);

      for(Fai = Fa, Xi = X; Fai != Fae; Fai++)                                 // loop over examples
        for(dphii = dphi, anci = anc; dphii != dphie; dphii++, Xi++, anci++)   // loop over anchors
          *dphii += *Xi * (Fai->y * anci->x - Fai->x * anci->y);


      // Scale the changes - normalize the jumps
      double scaling = 1e10;
      for(dphii = dphi; dphii != dphie ; dphii++) {
        if (fabs(*dphii * scaling) > 0.01)
          scaling = fabs(0.01 / *dphii);
      }

      // Move anchors
      for(anci = anc, dphii = dphi; dphii != dphie; dphii++, anci++) {
        double tphi = atan2(anci->y, anci->x) + *dphii * scaling;
        anci->x = cos(tphi);
        anci->y = sin(tphi);
      }
    }

    symmetricTransformation(anc, ance, mirrorSymmetry != 0);

    anchors = PyList_New(nAttrs);
    for(i = 0, anci = anc, lli = ll; anci != ance; lli++, i++, anci++) 
      PyList_SetItem(anchors, i, *lli ? Py_BuildValue("ddO", anci->x, anci->y, *lli) : Py_BuildValue("dd", anci->x, anci->y));
      
    free(X);
    free(anc);
    free(ll);
    if (contClass < 2)
      free(classes);

    free(dphi);
    free(pts);
    free(Fa);
    xfree(Fr);
    xfree(sum);

    return anchors;
      
  PyCATCH;
}




PyObject *optimizeAnchorsR(PyObject *, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS | METH_KEYWORDS, "(scaledData, classes, anchors[, attractG=1.0, repelG=-1.0, law=InverseLinear, steps=1, normalizeExamples=1]) -> new-anchors")
{
  PyTRY
    // get the arguments from Python
    PyObject *scaledData, *pyclasses, *anchors, *pyattrIndices;
    double attractG = 1.0,              repelG = -1.0,          sigma2 = 1.0;
    int    law = TPNN::InverseLinear,   steps = 1,              normalizeExamples = 0,
           contClass = 0,               dynamicBalancing = 0,   mirrorSymmetry = 0;

    static char *kwlist[] = {"scaledData", "classes", "anchors", "attrIndices", "attractG", "repelG", "law", "sigma2", "dynamicBalancing", "steps", "normalizeExamples", "contClass", "mirrorSymmetry", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "OOOO|ddidiiiii:optimizeAnchors", kwlist,
                                         &scaledData, &pyclasses, &anchors, &pyattrIndices,
                                         &attractG, &repelG, &law, &sigma2, &dynamicBalancing,
                                         &steps, &normalizeExamples, &contClass, &mirrorSymmetry))
      return NULL;


    double *Xi, *X;            // values of the chosen attributes
    int *classes;              // classes (for continuous) or indices for groups of classes (for discrete);
    int nAttrs, nExamples;     // number of (chosen) attributes and of examples
    TPoint *anci, *anc, *ance; // anchor coordinates
    PyObject **lli, **ll;      // anchor labels
    double minClass, maxClass; // minimal and maximal class values (for cont), #classes+1 (for disc)

    // convert the examples, classes and anchors from Python lists
    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, contClass, nAttrs, nExamples, X, classes, anc, ll, minClass, maxClass))
      return PYNULL;
    ance = anc + nAttrs;

    int i;
    double *dri, *dr = (double *)malloc(nAttrs * sizeof(double)), *dre = dr + nAttrs;           // anchors' moves
    TPoint *ptsi, *pts = (TPoint *)malloc(nExamples * sizeof(TPoint)), *ptse = pts + nExamples; // projections of examples
    double *radi, *rad = (double *)malloc(nAttrs * sizeof(double));                       // radii of anchors
    TPoint *Fai, *Fa = (TPoint *)malloc(nExamples * sizeof(TPoint)), *Fae = Fa + nExamples;     // forces on examples

    TPoint *Fr = !contClass && (attractG != 0.0) && (repelG != 0.0) 
                     ? (TPoint *)malloc(nExamples * sizeof(TPoint)) : NULL;    // temporary array for computeForces

    
    double *sumi, *sum = NULL, *sume;
    if (normalizeExamples) {
      sum = (double *)malloc(nExamples * sizeof(double));                      // sums of attr values for each example
      sume = sum + nExamples;
    }

    while (steps--) {
      // compute the projection
      for(anci = anc, radi = rad; anci != ance; anci++, radi++)
        *radi = sqrt(sqr(anci->x) + sqr(anci->y));

      if (normalizeExamples) {
        for(sumi = sum, Xi = X, ptsi = pts; ptsi != ptse; sumi++, ptsi++) {
          ptsi->x = ptsi->y = *sumi = 0.0;
          for(anci = anc, radi = rad; anci != ance; anci++, Xi++, radi++) {
            ptsi->x += *Xi * anci->x;
            ptsi->y += *Xi * anci->y;
            *sumi += *Xi * *radi;
          }
          if (fabs(*sumi) > 1e-6) {
            ptsi->x /= *sumi;
            ptsi->y /= *sumi;
          }
          else
            *sumi = 1.0; // we also use *sumi later
        }
      }
      else {
        for(Xi = X, ptsi = pts; ptsi != ptse; ptsi++) {
          ptsi->x = ptsi->y = 0.0;
          for(anci = anc; anci != ance; anci++, Xi++) {
            ptsi->x += *Xi * anci->x;
            ptsi->y += *Xi * anci->y;
          }
        }
      }

      switch (contClass) {
        case 0:
          computeForcesDiscrete(pts, ptse, classes, law, sigma2, attractG, repelG, dynamicBalancing != 0, Fa, Fr);
          break;
        case 1:
          computeForcesContinuous(pts, ptse, (double *)classes, law, sigma2, Fa);
          break;
        case 2:
          computeForcesMDS(pts, ptse, (TSymMatrix *)classes, law, sigma2, Fa);
          break;
      };

      // Normalize forces if needed (why?! instead of dividing each *Xi?)
      if (normalizeExamples)
        for(Fai = Fa, sumi = sum; Fai != Fae; Fai++, sumi++) {
          Fai->x /= *sumi;
          Fai->y /= *sumi;
        }


      // Transmit forces on particles to the anchors
      for(dri = dr; dri != dre; *dri++ = 0.0);

      for(Fai = Fa, Xi = X; Fai != Fae; Fai++)                                 // loop over examples
        for(dri = dr, anci = anc; dri != dre; dri++, anci++, Xi++)             // loop over anchors
          *dri += *Xi * (Fai->x * anc->x + Fai->y * anci->y);

      double scaling = 1e10;
      for(anci = anc, dri = dr; dri != dre; anci++, dri++) {
        double maxdr = 0.1 * sqrt(sqr(anci->x) + sqr(anci->y));
        if ((maxdr > 1e-5) && (fabs(*dri) > 1e-5)) {
          if (scaling * *dri > maxdr)
            scaling = maxdr / *dri;
        }
      }

      // Move anchors
      double maxr = 0.0;
      for(anci = anc, dri = dr, radi = rad; dri != dre; dri++, anci++, radi++) {
        double newr = *radi + *dri * scaling;
        if (newr < 1e-4)
          newr = 1e-4;
        double rat = newr / *radi;
        if (rat > 10) {
          rat = 10;
          newr = *radi * 100;
        }
        anci->x *= rat;
        anci->y *= rat;
//        printf("%f\t%f\t%f\t%f\t%f\n", anci->x, anci->y, newr, sqrt(sqr(anci->x) + sqr(anci->y)), rat);
        if (newr > maxr)
          maxr = newr;
      }


      // Scale (so that the largest radius is 1)
      if (maxr > 0.001) {
        maxr = sqrt(maxr);
        for(anci = anc; anci != ance; anci++) {
          anci->x /= maxr;
          anci->y /= maxr;
        }
      }
    }
//    printf("\n");

    symmetricTransformation(anc, ance, mirrorSymmetry != 0);
    

    anchors = PyList_New(nAttrs);
    for(i = 0, anci = anc, lli = ll;i < nAttrs; lli++, i++, anci++) 
      PyList_SetItem(anchors, i, *lli ? Py_BuildValue("ddO", anci->x, anci->y, *lli) : Py_BuildValue("dd", anci->x, anci->y));

    free(X);
    free(anc);
    free(ll);
    if (contClass < 2)
      free(classes);

    free(pts);
    free(rad);
    free(Fa);
    xfree(Fr);
    xfree(sum);

    return anchors;
      
  PyCATCH;
}



#define nColors 6
#ifdef _MSC_VER
#pragma warning (disable: 4305 4309)
#endif

PyObject *potentialsBitmap(PyObject *, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(P2NN, rx, ry, offx, offy, cellsize, scaleFactor, grid) -> bitmap as string")
{
  PyTRY
    PyObject *cls;
    int rx, ry, cell;
    int offx, offy;
    double scaleFactor;
    int grid = 1;
    int circle = 0;
    if (!PyArg_ParseTuple(args, "Oiiiiid|ii:potentialsBitmap", &cls, &rx, &ry, &offx, &offy, &cell, &scaleFactor, &grid, &circle))
      return PYNULL;

    TP2NN *tp2nn = &dynamic_cast<TP2NN &>(PyOrange_AsOrange(cls).getReference());

    bool contClass = tp2nn->classVar->varType == TValue::FLOATVAR;

    const int nClasses = contClass ? 0 : tp2nn->classVar->noOfValues();
    const int nShades = contClass ? 0 : 255/nClasses;

    const int oneLine = (rx + 3) & 0xfffffffc;
    const int bitmapSize = oneLine * ry;
    char *bitmap = new char[bitmapSize];
    memset(bitmap, 255, bitmapSize);

    rx--; // precaution
    ry--;

    float *probs = new float[nClasses], *pe = probs + nClasses;
    const double minClass = tp2nn->minClass;
    const double divClass = tp2nn->maxClass == minClass ? 0.0 : 255.0 / (tp2nn->maxClass - minClass);

    const double rxbysf = rx*scaleFactor;
    const double rybysf = ry*scaleFactor;
    for(int y = 0; y < ry; y+=cell) {
      const double realy = (ry-y-offy)/rybysf;
      int dx = circle ? rx/2 * sqrt(1 - sqr((2*y)/float(ry) -1)) : rx/2;
      for(int x = rx/2-dx, xe = rx/2+dx; x < xe; x+=cell) {
        const double realx = (x-offx)/rxbysf;

        unsigned char color;

        if (contClass) {
          const int icolor = (tp2nn->averageClass(realx, -realy) - minClass) * divClass;
          if (icolor < 0)
            color = 0;
          else if (icolor > 255)
            color = 255;
          else
            color = icolor;
        }
        else {
          tp2nn->classDistribution(realx, realy, probs, nClasses);
          double sprobs = *probs;
          float *largest = probs;
          for(float *pi = probs+1; pi != pe; pi++) {
            sprobs += *pi;
            if (*pi > *largest)
              largest = pi;
          }
          color = floor(0.5 + nShades * (*largest/sprobs*nClasses - 1) / (nClasses - 1));
          if (color >= nShades)
            color = nShades - 1;
          else if (color < 0)
            color = 0;
          color += nShades * (largest - probs);
        }
        const int ys = y+cell < ry ? cell : ry-y;
        char *yy = bitmap + y*oneLine+x;

        if (grid)
          for(char *yye = yy + (ys-1)*oneLine; yy < yye; yy += oneLine)
            memset(yy, color, cell-1);
        else
          for(char *yye = yy + ys*oneLine; yy < yye; yy += oneLine)
            memset(yy, color, cell);
      }
    }

    return contClass ? Py_BuildValue("s#", bitmap, bitmapSize)
                     : Py_BuildValue("s#i", bitmap, bitmapSize, nShades);

  PyCATCH
}
