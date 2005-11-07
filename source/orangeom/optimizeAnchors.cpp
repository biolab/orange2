#include "orange_api.hpp"
#include "examplegen.hpp"
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
     pyclasses:     class for each example in 'scaledData'
     anchors:       anchor positions (Python list of lists with 2 or 3 elements)
     pyattrIndices: indices of attributes to be used

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
     anc            anchor coordinates
     ll             anchor labels (stored for when the anchor list needs to be reconstructed)
     minClass       the minimal value encountered (for continuous only)
     maxClass       for continuous classes it is the maximal value, for discrete it is the number of classes-1
*/

bool loadRadvizData(PyObject *scaledData, PyObject *pyclasses, PyObject *anchors, PyObject *pyattrIndices,
                    int &nAttrs, int &nExamples, int &contClass,
                    double *&X, int *&classes, TPoint *&anc, PyObject **&ll,
                    double &minClass, double &maxClass)
{
  if (!PyList_Check(scaledData) || !PyList_Check(pyclasses) || !PyList_Check(anchors))
    PYERROR(PyExc_TypeError, "scaled data, classes and anchors should be given a lists", false);

  if (PyList_Size(scaledData) != PyList_Size(pyclasses))
    PYERROR(PyExc_TypeError, "'scaledData' and 'classes' have different lengths", false);

  if (PyList_Size(anchors) != PyList_Size(pyattrIndices))
    PYERROR(PyExc_TypeError, "'anchors' and 'attrIndices' have different lengths", false);

  nAttrs = PyList_Size(anchors);
  nExamples = PyList_Size(scaledData);

  X = (double *)malloc(nExamples * nAttrs * sizeof(double));
  classes = (int *)malloc(nExamples * (contClass ? sizeof(double) : sizeof(int)));
  anc = (TPoint *)malloc(nAttrs * sizeof(TPoint));
  ll = (PyObject **)malloc(nAttrs * sizeof(PyObject *));

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

  if (contClass) {
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

  else {
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
   UNTESTED! DOESN'T USE THE LAW!

   INPUT:
     pts           projections of examples
     classes       example classes
     nExamples     number of examples (the length of above arrays and of Fr)
     law           0=Linear, 1=Square, 2=Gaussian

   OUTPUT:
     F            forces acting on each example (memory should be allocated by the caller!)
*/

void computeForcesContinuous(TPoint *pts, const TPoint *ptse, const double *classes, 
                             const int &law,
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

      double sr2 = sqrt(r2);
      double fx = dx /sr2;
      double fy = dy /sr2;
      if (r2 < 1e-10)
        r2 = 1e-10;

      const double TFr = -fabs(*classesi-*classesi2) / r2;
      const double FrX = TFr * fx;
      const double FrY = TFr * fy;
      Fi->x  += FrX;
      Fi2->x -= FrX;
      Fi->y  += FrY;
      Fi2->y -= FrY;
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

      if (law == TPNN::InverseSquare) {
        for(ptsi = pts + *classesi, ptsie = pts + classesi[1], Fai = Fa + *classesi; ptsi != ptsie; ptsi++, Fai++) {
          Fai->x = Fai-> y = 0.0;
          for(ptsi2 = pts + *classesi,                         Fai2 = Fa + *classesi; ptsi2 != ptsi; ptsi2++, Fai2++) {
            const double dx = ptsi->x - ptsi2->x;
            const double dy = ptsi->y - ptsi2->y;
            const double r = sqrt(sqr(dx) + sqr(dy));

            const double druvx = dx * r;
            Fai->x  += druvx;
            Fai2->x -= druvx;

            const double druvy = dy * r;
            Fai->y  += druvy;
            Fai2->y -= druvy;
          }
        }
      }

      else {
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
            case TPNN::InverseSquare:
              fct = 1 / (r2 * sqrt(r2));
              break;
            case TPNN::InverseExponential:
              fct = 1 / (exp(r2/sigma2) - 1);
          }

          const double druvx = - dx * fct;
          Fri->x  += druvx;
          Fri2->x -= druvx;

          const double druvy = - dy * fct;
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
    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, nAttrs, nExamples, contClass, X, classes, anc, ll, minClass, maxClass))
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


      // Compute the forces
      if (contClass)
        computeForcesContinuous(pts, ptse, (double *)classes, law, Fa);
      else
        computeForcesDiscrete(pts, ptse, classes, law, sigma2, attractG, repelG, dynamicBalancing != 0, Fa, Fr);


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
          danci->x -= Fai->x * *Xi;
          danci->y -= Fai->y * *Xi;
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
      aax /= nAttrs;
      aay /= nAttrs;

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
    free(classes);
    free(anc);
    free(ll);

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
    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, nAttrs, nExamples, contClass, X, classes, anc, ll, minClass, maxClass))
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



      // Compute the forces
      if (contClass)
        computeForcesContinuous(pts, ptse, (double *)classes, law, Fa);
      else
        computeForcesDiscrete(pts, ptse, classes, law, sigma2, attractG, repelG, dynamicBalancing!=0, Fa, Fr);


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
          *dphii -= *Xi * (Fai->y * anci->x - Fai->x * anci->y);


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
    free(classes);
    free(anc);
    free(ll);

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
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "OOOO|ddidiiiii:optimizeAnchors", kwlist, &scaledData, &pyclasses, &anchors, &pyattrIndices, &attractG, &repelG, &law, &sigma2, &dynamicBalancing, &steps, &normalizeExamples, &contClass, &mirrorSymmetry))
      return NULL;


    double *Xi, *X;            // values of the chosen attributes
    int *classes;              // classes (for continuous) or indices for groups of classes (for discrete);
    int nAttrs, nExamples;     // number of (chosen) attributes and of examples
    TPoint *anci, *anc, *ance; // anchor coordinates
    PyObject **lli, **ll;      // anchor labels
    double minClass, maxClass; // minimal and maximal class values (for cont), #classes+1 (for disc)

    // convert the examples, classes and anchors from Python lists
    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, nAttrs, nExamples, contClass, X, classes, anc, ll, minClass, maxClass))
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


      // Compute the forces
      if (contClass)
        computeForcesContinuous(pts, ptse, (double *)classes, law, Fa);
      else
        computeForcesDiscrete(pts, ptse, classes, law, sigma2, attractG, repelG, dynamicBalancing != 0, Fa, Fr);


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
          *dri -= *Xi * (Fai->x * anc->x + Fai->y * anci->y);

      double scaling = 1e10;
      for(anci = anc, dri = dr; dri != dre; anci++, dri++) {
        double maxdr = 0.1 * sqrt(sqr(anci->x) + sqr(anci->y));
        if ((maxdr > 1e-5) && (*dri > 1e-5)) {
          if (scaling * *dri > maxdr)
            scaling = maxdr / *dri;
        }
      }

      // Move anchors
      double maxr = 0.0;
      for(anci = anc, dri = dr, radi = rad; dri != dre; dri++, anci++, radi++) {
        const double newr = *radi + *dri * scaling;
        double rat = newr / *radi;
        anci->x *= rat;
        anci->y *= rat;
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

    symmetricTransformation(anc, ance, mirrorSymmetry != 0);
    

    anchors = PyList_New(nAttrs);
    for(i = 0, anci = anc, lli = ll;i < nAttrs; lli++, i++, anci++) 
      PyList_SetItem(anchors, i, *lli ? Py_BuildValue("ddO", anci->x, anci->y, *lli) : Py_BuildValue("dd", anci->x, anci->y));

    free(X);
    free(classes);
    free(anc);
    free(ll);

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

PyObject *potentialsBitmapCircle(PyObject *, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(P2NN, rx, ry, cellsize, scaleFactor) -> bitmap as string")
{
  PyTRY
    PyObject *cls;
    int rx, ry, cell;
    double scaleFactor;
    if (!PyArg_ParseTuple(args, "Oiiid:potentialsBitmap", &cls, &rx, &ry, &cell, &scaleFactor))
      return PYNULL;

    TP2NN *tp2nn = &dynamic_cast<TP2NN &>(PyOrange_AsOrange(cls).getReference());

    bool contClass = tp2nn->classVar->varType == TValue::FLOATVAR;

    const int nClasses = contClass ? 0 : tp2nn->classVar->noOfValues();
    const int nShades = contClass ? 0 : 255/nClasses;

    const int oneLine = (2*rx + 3) & 0xfffffffc;
    const int bitmapSize = oneLine * 2*ry;
    char *bitmap = new char[bitmapSize];
    memset(bitmap, 255, bitmapSize);
    char *bitmapmid = bitmap + oneLine*ry + rx;

    rx -= 1;
    ry -= 1;

    float *probs = new float[nClasses], *pe = probs + nClasses;
    const double minClass = tp2nn->minClass;
    const double divClass = tp2nn->maxClass == minClass ? 0.0 : 255.0 / (tp2nn->maxClass - minClass);

    const double rxbysf = rx*scaleFactor;
    for(int y = -ry+1; y < ry-1; y+=cell) {
      const double yry = double(y)/ry;
      const double realy = yry/scaleFactor;
      int xe = ceil(rx * sqrt(1.0 - yry*yry));
      xe += cell - xe % cell;
      for(int x = -xe; x < xe; x+=cell) {
        const double realx = x/rxbysf;

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
          tp2nn->classDistribution(realx, -realy, probs, nClasses);
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
/*
        const int ys = y+cell < ry ? cell : ry-y;
        for(char *yy = bitmapmid + y*oneLine + x, *yye = yy + ys*oneLine; yy < yye; yy += oneLine)
          memset(yy, color, cell);
*/
        const int ys = y+cell < ry ? cell : ry-y;
        char *yy = bitmapmid + y*oneLine+x;

/*        memset(yy, color, cell);
        yy += oneLine;
        for(char *yye = yy + (ys-1)*oneLine; yy < yye; yy += oneLine)
          *yy = yy[cell-1] = color;
        memset(yy, color, cell);
*/
        for(char *yye = yy + (ys-1)*oneLine; yy < yye; yy += oneLine)
          memset(yy, color, cell-1);
      }
    }

    return contClass ? Py_BuildValue("s#", bitmap, bitmapSize)
                     : Py_BuildValue("s#i", bitmap, bitmapSize, nShades);

  PyCATCH
}



PyObject *potentialsBitmapSquare(PyObject *, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(P2NN, rx, ry, cellsize, scaleFactor) -> bitmap as string")
{
  PyTRY
    PyObject *cls;
    int rx, ry, cell;
    double offx, offy;
    double scaleX, scaleY;
    if (!PyArg_ParseTuple(args, "Oiiddddi:potentialsBitmap", &cls, &rx, &ry, &offx, &offy, &scaleX, &scaleY, &cell))
      return PYNULL;

    TP2NN *tp2nn = &dynamic_cast<TP2NN &>(PyOrange_AsOrange(cls).getReference());

    bool contClass = tp2nn->classVar->varType == TValue::FLOATVAR;

    const int nClasses = contClass ? 0 : tp2nn->classVar->noOfValues();
    const int nShades = contClass ? 0 : 255/nClasses;

    const int oneLine = (rx + 3) & 0xfffffffc;
    const int bitmapSize = oneLine * ry;
    char *bitmap = new char[bitmapSize];
    memset(bitmap, 255, bitmapSize);

    rx -= 1;
    ry -= 1;

    float *probs = new float[nClasses], *pe = probs + nClasses;
    const double minClass = tp2nn->minClass;
    const double divClass = tp2nn->maxClass == minClass ? 0.0 : 255.0 / (tp2nn->maxClass - minClass);

    for(int y = 0; y < ry; y+=cell) {
      const double realy = (y-offy)*scaleY;
      for(int x = 0; x < rx; x+=cell) {
        const double realx = (x-offx)*scaleX;

        unsigned char color;

        if (contClass) {
          const int icolor = (tp2nn->averageClass(realx, realy) - minClass) * divClass;
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
        char *yy = bitmap + y*oneLine + x;

        for(char *yye = yy + (ys-1)*oneLine; yy < yye; yy += oneLine)
          memset(yy, color, cell-1);
      }
    }

    return contClass ? Py_BuildValue("s#", bitmap, bitmapSize)
                     : Py_BuildValue("s#i", bitmap, bitmapSize, nShades);

  PyCATCH
}





bool loadRadvizData(PyObject *scaledData, PyObject *anchors, PyObject *pyattrIndices,
                    int &nAttrs, int &nExamples,
                    double *&X, TPoint *&anc, PyObject **&ll)
{
  if (!PyList_Check(scaledData) || !PyList_Check(anchors))
    PYERROR(PyExc_TypeError, "scaled data and anchors should be given a lists", false);

  nAttrs = PyList_Size(anchors);
  nExamples = PyList_Size(scaledData);

  X = (double *)malloc(nExamples * nAttrs * sizeof(double));
  anc = (TPoint *)malloc(nAttrs * sizeof(TPoint));
  ll = (PyObject **)malloc(nAttrs * sizeof(PyObject *));

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

  for(Xi = X, i = 0; i < nExamples; i++) {
    PyObject *ex = PyList_GetItem(scaledData, i);
    for(aii = attrIndices; aii < aie; aii++)
      *Xi++ = PyFloat_AsDouble(PyList_GetItem(ex, *aii));
  }

  free(attrIndices);
  return true;
}


#include "symmatrix.hpp"

PyObject *MDSA(PyObject *, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "(various)")
{
  PyTRY
    PyObject *scaledData;
    PyObject *anchors;
    PyObject *pyattrIndices;
    PSymMatrix distances;
    int steps = 1;
    int normalizeExamples = 1;

    if (!PyArg_ParseTuple(args, "OOOO&|ii:MDSa", &scaledData, &anchors, &pyattrIndices, cc_SymMatrix, &distances, &steps, &normalizeExamples))
      return NULL;

    double *Xi, *X;
    TPoint *anci, *anc, *ance;
    PyObject **lli, **ll;
    int nAttrs, nExamples;

    if (!loadRadvizData(scaledData, anchors, pyattrIndices, nAttrs, nExamples, X, anc, ll))
      return PYNULL;

    ance = anc + nAttrs;

    int i;
    double *radi, *rad = (double *)malloc(nAttrs * sizeof(double)), *rade = rad + nAttrs;
    TPoint *danci, *danc = (TPoint *)malloc(nAttrs * sizeof(TPoint)), *dance = danc + nAttrs;
    TPoint *ptsi, *pts = (TPoint *)malloc(nExamples * sizeof(TPoint)), *ptse = pts + nExamples, *ptsi2;
    double *sumi, *sum = (double *)malloc(nExamples * sizeof(double)), *sume = sum + nExamples;
    TPoint *Fi, *F = (TPoint *)malloc(nExamples * sizeof(TPoint)), *Fe = F + nExamples, *Fi2;

    while (steps--) {
      if (normalizeExamples) {
        for(anci = anc, radi = rad; anci != ance; anci++, radi++)
          *radi = sqrt(sqr(anci->x) + sqr(anci->y));
      }

      for(sumi = sum, Xi = X, ptsi = pts; sumi != sume; sumi++, ptsi++) {
        ptsi->x = ptsi->y = *sumi = 0.0;
        for(anci = anc, radi = rad; anci != ance; anci++, Xi++, radi++) {
          ptsi->x += *Xi * anci->x;
          ptsi->y += *Xi * anci->y;
          if (normalizeExamples)
            *sumi += *Xi * *radi;
        }
        if (normalizeExamples)
          if (*sumi != 0.0) {
            ptsi->x /= *sumi;
            ptsi->y /= *sumi;
          }
          else
            *sumi = 1.0; // we also use *sumi later
      }

                     
      for(Fi = F; Fi != Fe; Fi++)
        Fi->x = Fi-> y = 0.0;

      for(danci = danc; danci != dance; danci++)
        danci->x = danci->y = 0.0;

      float *edistances = distances->elements;

      for(ptsi = pts, Fi = F; ptsi != ptse; ptsi++, Fi++, edistances++ /* skip diagonal */) {
        for(ptsi2 = pts, Fi2 = F; ptsi2 != ptsi; ptsi2++, edistances++, Fi2++) {
            const double dx = ptsi->x - ptsi2->x;
            const double dy = ptsi->y - ptsi2->y;
            double dist = sqrt(sqr(dx) + sqr(dy));
            if (dist < 1e-20)
              continue;

            double fx = dx / dist;
            double fy = dy / dist;
            if (dist < 1e-10)
              dist = 1e-10;

            double F = dist - *edistances;
//            const int sign = F > 0 ? 1 : -1;
//            F *= fabs(F);
//            printf("F%5.3f %f", F, *edistances);
            const double Fx = F * fx;
            const double Fy = F * fy;
            Fi->x += Fx;
            Fi->y += Fy;
            Fi2->x -= Fx;
            Fi2->y -= Fy;
        }
      }

      if (normalizeExamples) {
        for(Fi = F, sumi = sum, Xi = X; Fi != Fe; Fi++, sumi++) {
          Fi->x /= *sumi;
          Fi->y /= *sumi;
          for(danci = danc; danci != dance; danci++, Xi++) {
            danci->x -= Fi->x * *Xi;
            danci->y -= Fi->y * *Xi;
          }
        }
      }

      else {
        for(Fi = F, Xi = X; Fi != Fe; Fi++) {
          for(danci = danc; danci != dance; danci++, Xi++) {
            danci->x -= Fi->x * *Xi;
            danci->y -= Fi->y * *Xi;
          }
        }
      }

  // Scale the changes - normalize the jumps
      double scaling = 1e10;
      for(anci = anc, danci = danc; danci != dance; anci++, danci++) {
        double maxdr = 0.1 * sqrt(sqr(anci->x) + sqr(anci->y));
        double dr = sqrt(sqr(danci->x) + sqr(danci->y));
        if ((maxdr > 1e-5) && (dr > 1e-5)) {
          if (scaling * dr > maxdr)
            scaling = maxdr / dr;
        }
      }

      for(danci = danc; danci != dance; danci++) {
        danci->x *= scaling;
        danci->y *= scaling;
      }


  // Move anchors
      for(anci = anc, danci = danc; danci != dance; danci++, anci++) {
        anci->x += danci->x;
        anci->y += danci->y;
      }

 
  //Centering
      double aax = 0.0, aay = 0.0;
      for(anci = anc; anci != ance; anci++) {
        aax += anci->x;
        aay += anci->y;
      }

      aax /= nAttrs;
      aay /= nAttrs;

      for(anci = anc; anci != ance; anci++) {
        anci->x -= aax;
        anci->y -= aay;
      }

   // Scaling, rotating and mirroring

      // find the largest and the second largest not collocated with the largest
      double maxr = 0.0, maxr2 = 0.0;
      TPoint *anci_l = NULL, *anci_l2 = NULL;
/*      for(anci = anc; anci != ance; anci++) {
        const double r = sqr(anci->x) + sqr(anci->y);
        if (r > maxr) {
          maxr2 = maxr;
          anci_l2 = anci_l;
          maxr = r;
          anci_l = anci;
        }
        else if ((r > maxr2) && ((anci->x != anci_l->x) || (anci->y != anci_l->y))) {
          maxr2 = r;
          anci_l2 = anci;
        }
      }
*/
      for(anci = anc; anci != ance; anci++) {
        const double r = sqr(anci->x) + sqr(anci->y);
        if (r > maxr) {
          maxr = r;
          anci_l = anci;
        }
      }
      anci_l = anc;
      anci_l2 = anc+1;

      if (anci_l2) {
        maxr = maxr > 0.0 ? sqrt(maxr) : 1.0;

        double phi = atan2(anci_l->y, anci_l->x);
        double phi2 = atan2(anci_l2->y, anci_l2->x);

        // disabled to avoid the flips
        // int sign = (phi2>phi) && (phi2-phi < 3.1419265) ? 1 : -1;
        int sign = 1;

        double dphi = 3.1419265/2.0 - phi;
        double cs = cos(dphi)/maxr, sn = sin(dphi)/maxr;

        for(anci = anc; anci != ance; anci++) {
          const double tx = anci->x * cs - anci->y * sn;
          anci->y = anci->x * sn + anci->y * cs;
          anci->x = sign * tx;
        }
      }
    }

    anchors = PyList_New(nAttrs);
    for(i = 0, anci = anc, lli = ll;i < nAttrs; lli++, i++, anci++)
      PyList_SetItem(anchors, i, *ll ? Py_BuildValue("ddO", anci->x, anci->y, *lli) : Py_BuildValue("dd", anci->x, anci->y));
      

    for(anci = anc, radi = rad; anci != ance; anci++, radi++)
      *radi = sqrt(sqr(anci->x) + sqr(anci->y));

    for(sumi = sum, Xi = X; sumi != sume; sumi++) {
      *sumi = 0.0;
      for(radi = rad, i = nAttrs; i--; *sumi += *Xi++ * *radi++);
      if (*sumi == 0.0)
        *sumi = 1.0;
    }

    free(anc);
    free(danc);
    free(pts);
    free(ll);
    free(X);
    free(sum);
    free(F);
    free(rad);

    return Py_BuildValue("Od", anchors, 0);
      
  PyCATCH;
}

