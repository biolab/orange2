#include "orange_api.hpp"
#include "examplegen.hpp"
#include "pnn.hpp"
#include "../orange/px/externs.px"

//#include "rconversions.hpp"

inline double sqr(const double &x)
{ return x*x; }


typedef struct {double x, y; } TPoint;


float computeEnergyLow(const int &nExamples, const int &nAttrs, double *X, int *classes, TPoint *pts, TPoint *anc, double *sum, const double attractG, const double repelG, const int law)
{
  double *Xi;
  double *sumi, *sume = sum + nExamples;
  TPoint *anci, *ance = anc + nAttrs;
  TPoint *ptsi, *ptsj, *ptse = pts + nExamples;
  int *classesi, *classesj;

  for(sumi = sum, Xi = X, ptsi = pts; sumi != sume; sumi++, ptsi++) {
    ptsi->x = ptsi->y = 0.0;
    for(anci = anc; anci != ance; anci++, Xi++) {
      ptsi->x += *Xi * anci->x;
      ptsi->y += *Xi * anci->y;
    }
    ptsi->x /= *sumi;
    ptsi->y /= *sumi;
  }

  double E = 0.0;

  switch(law) {
    case TPNN::InverseLinear:
      for(ptsi = pts, classesi = classes; ptsi != ptse; ptsi++, classesi++)
        for(ptsj = pts, classesj = classes; ptsj != ptsi; ptsj++, classesj++) {
          const int duv = *classesi == *classesj ? attractG : repelG;
          if (duv) {
            double dist = sqr(ptsi->x - ptsj->x) + sqr(ptsi->y - ptsj->y);
            E += duv * log(dist < 1e-15 ? 1e-15 : dist);
          }
        }
      E /= 2.0; // this is needed since we omitted a sqrt inside log...
      break;

    case TPNN::InverseSquare:
      for(ptsi = pts, classesi = classes; ptsi != ptse; ptsi++, classesi++)
        for(ptsj = pts, classesj = classes; ptsj != ptsi; ptsj++, classesj++) {
          const int duv = *classesi == *classesj ? attractG : repelG;
          if (duv) {
            double dist = sqr(ptsi->x - ptsj->x) + sqr(ptsi->y - ptsj->y);
            E -= duv / (dist < 1e-15 ? 1e-15 : sqrt(dist));
          }
        }
      break;

    case TPNN::InverseExponential:
      for(ptsi = pts, classesi = classes; ptsi != ptse; ptsi++, classesi++)
        for(ptsj = pts, classesj = classes; ptsj != ptsi; ptsj++, classesj++) {
          const int duv = *classesi == *classesj ? attractG : repelG;
          if (duv) {
            double dist = sqr(ptsi->x - ptsj->x) + sqr(ptsi->y - ptsj->y);
            E -= duv * exp(dist < 1e-15 ? -1e-15 : -sqrt(dist));
          }
        }
      break;

  }
  return E;
}



bool loadRadvizData(PyObject *scaledData, PyObject *pyclasses, PyObject *anchors, PyObject *pyattrIndices,
                    int &nAttrs, int &nExamples,
                    double *&X, int *&classes, TPoint *&anc, PyObject **&ll)
{
  if (!PyList_Check(scaledData) || !PyList_Check(pyclasses) || !PyList_Check(anchors))
    PYERROR(PyExc_TypeError, "scaled data, classes and anchors should be given a lists", false);

  nAttrs = PyList_Size(anchors);
  nExamples = PyList_Size(scaledData);

  X = (double *)malloc(nExamples * nAttrs * sizeof(double));
  classes = (int *)malloc(nExamples * sizeof(int));
  anc = (TPoint *)malloc(nAttrs * sizeof(TPoint));
  ll = (PyObject **)malloc(nAttrs * sizeof(PyObject *));

  int *aii, *attrIndices = (int *)malloc(nAttrs * sizeof(int)), *aie = attrIndices + nAttrs;
  TPoint *anci;
  PyObject **lli;
  int *classesi;
  double *Xi;
  int i;
   
  for(anci = anc, aii = attrIndices, lli = ll, i = 0; i < nAttrs; i++, anci++, aii++, lli++) {
    *lli = NULL;
    PyArg_ParseTuple(PyList_GetItem(anchors, i), "dd|O", &anci->x, &anci->y, lli);
    *aii = PyInt_AsLong(PyList_GetItem(pyattrIndices, i));
  }

  for(classesi = classes, Xi = X, i = 0; i < nExamples; classesi++, i++) {
    *classesi = PyInt_AsLong(PyList_GetItem(pyclasses, i));

    PyObject *ex = PyList_GetItem(scaledData, i);
    for(aii = attrIndices; aii < aie; aii++)
      *Xi++ = PyFloat_AsDouble(PyList_GetItem(ex, *aii));
  }

  free(attrIndices);

  return true;
}


PyObject *computeEnergy(PyObject *, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, ".... ... ..")
{
  PyTRY
    PyObject *scaledData;
    PyObject *pyclasses;
    PyObject *anchors;
    PyObject *pyattrIndices;
    double attractG = 1.0, repelG = -1.0;
    int steps = 1;
    int law = 0;

    if (!PyArg_ParseTuple(args, "OOOO|ddi:optimizeAnchors", &scaledData, &pyclasses, &anchors, &pyattrIndices, &attractG, &repelG, &law))
      return NULL;

    double *X, *Xi;
    int *classes;
    TPoint *anci, *anc, *ance;
    PyObject **ll;
    int nAttrs, nExamples;

    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, nAttrs, nExamples, X, classes, anc, ll))
      return PYNULL;

    ance = anc + nAttrs;

    TPoint *pts = (TPoint *)malloc(nExamples * sizeof(TPoint));
    double *sumi, *sum = (double *)malloc(nExamples * sizeof(double)), *sume = sum + nExamples;
    double *radi, *rad = (double *)malloc(nAttrs * sizeof(double)), *rade = rad + nAttrs;

    for(anci = anc, radi = rad; anci != ance; anci++, radi++)
      *radi = sqrt(sqr(anci->x) + sqr(anci->y));

    for(sumi = sum, Xi = X; sumi != sume; sumi++) {
      *sumi = 0.0;
      for(radi = rad; radi != rade; *sumi += *Xi++ * *radi++);
      if (*sumi == 0.0)
        *sumi = 1.0;
    }

    double E = computeEnergyLow(nExamples, nAttrs, X, classes, pts, anc, sum, attractG, repelG, law);

    free(X);
    free(classes);
    free(anc);
    free(ll);
    free(sum);
    free(pts);
    free(rad);

    return PyFloat_FromDouble(E);
  PyCATCH
}


     
PyObject *optimizeAnchors(PyObject *, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "(scaledData, classes, anchors[, attractG=1.0, repelG=-1.0, law=InverseLinear, steps=1, normalizeExamples=1]) -> new-anchors")
{
  PyTRY
    PyObject *scaledData;
    PyObject *pyclasses;
    PyObject *anchors;
    PyObject *pyattrIndices;
    double attractG = 1.0, repelG = -1.0;
    int law = TPNN::InverseLinear;
    int steps = 1;
    int normalizeExamples = 1;

    if (!PyArg_ParseTuple(args, "OOOO|ddiii:optimizeAnchors", &scaledData, &pyclasses, &anchors, &pyattrIndices, &attractG, &repelG, &law, &steps, &normalizeExamples))
      return NULL;

    double *Xi, *X;
    int *classes;;
    TPoint *anci, *anc, *ance;
    PyObject **lli, **ll;
    int nAttrs, nExamples;

    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, nAttrs, nExamples, X, classes, anc, ll))
      return PYNULL;

    ance = anc + nAttrs;

    int i, u, v;
    double *radi, *rad = (double *)malloc(nAttrs * sizeof(double)), *rade = rad + nAttrs;
    TPoint *danci, *danc = (TPoint *)malloc(nAttrs * sizeof(TPoint)), *dance = danc + nAttrs;
    TPoint *ptsi, *pts = (TPoint *)malloc(nExamples * sizeof(TPoint)), *ptse = pts + nExamples;
    double *sumi, *sum = (double *)malloc(nExamples * sizeof(double)), *sume = sum + nExamples;
    TPoint *Ki, *K = (TPoint *)malloc(nExamples * sizeof(TPoint)), *Ke = K + nExamples;

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
        if (normalizeExamples) {
          if (*sumi == 0.0)
            *sumi = 1.0;
          ptsi->x /= *sumi;
          ptsi->y /= *sumi;
        }
      }


/* XXX   TO OPTIMIZE:
       - use pointers instead of indexing
       - sort the examples by classes so you don't have to compare them all the time
       - if attractive force are 0, you can gain some time by checking only the examples from different classes
*/
                     
      for(Ki = K; Ki != Ke; Ki++)
        Ki->x = Ki-> y = 0.0;

      for(danci = danc; danci != dance; danci++)
        danci->x = danci->y = 0.0;

      for(u = 0; u < nExamples; u++) {
        for(v = u+1; v < nExamples; v++) {
          const double duv = classes[u] == classes[v] ? attractG : repelG;
          if (duv == 0.0)
            continue;

          double ruv = sqr(pts[u].x - pts[v].x) + sqr(pts[u].y - pts[v].y);
          if (ruv < 1e-15)
            ruv = 1e-15;

          double druv;
          switch(law) {
            case TPNN::InverseLinear: druv = duv / sqrt(ruv); break;
            case TPNN::InverseSquare: druv = duv / ruv; break;
            case TPNN::InverseExponential: druv = duv / exp(-ruv); break;
          }

          const double druvx = druv * (pts[u].x - pts[v].x);
          K[u].x += druvx;
          K[v].x -= druvx;

          const double druvy = druv * (pts[u].y - pts[v].y);
          K[u].y += druvy;
          K[v].y -= druvy;
        }

        if (normalizeExamples) {
          K[u].x /= sum[u];
          K[u].y /= sum[u];
        }

        double *Xu = X + u * nAttrs;
        for(i = 0; i < nAttrs; i++, Xu++) {
          double ex = normalizeExamples ? *Xu/sum[u] : *Xu;
          danc[i].x -= K[u].x * ex;
          danc[i].y -= K[u].y * ex;
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
        int sign = (phi2>phi) && (phi2-phi < 3.1419265) ? 1 : -1;
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

    double E = computeEnergyLow(nExamples, nAttrs, X, classes, pts, anc, sum, attractG, repelG, law);

    free(anc);
    free(danc);
    free(pts);
    free(ll);
    free(classes);
    free(X);
    free(sum);
    free(K);
    free(rad);

    return Py_BuildValue("Od", anchors, E);
      
  PyCATCH;
}



PyObject *optimizeAnchorsRadial(PyObject *, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "(scaledData, classes, anchors[, attractG=1.0, repelG=-1.0, law=InverseLinear, steps=1, normalizeExamples=1]) -> new-anchors")
{
  PyTRY
    PyObject *scaledData;
    PyObject *pyclasses;
    PyObject *anchors;
    PyObject *pyattrIndices;
    double attractG = 1.0, repelG = -1.0;
    int law = TPNN::InverseLinear;
    int steps = 1;

    if (!PyArg_ParseTuple(args, "OOOO|ddii:optimizeAnchors", &scaledData, &pyclasses, &anchors, &pyattrIndices, &attractG, &repelG, &law, &steps))
      return NULL;

    double *Xi, *X;
    int *classes;;
    TPoint *anci, *anc, *ance;
    PyObject **lli, **ll;
    int nAttrs, nExamples;

    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, nAttrs, nExamples, X, classes, anc, ll))
      return PYNULL;

    ance = anc + nAttrs;

    int u, v, i;
    double *dphii, *dphi = (double *)malloc(nAttrs * sizeof(double)), *dphie = dphi + nAttrs;
    TPoint *ptsi, *pts = (TPoint *)malloc(nExamples * sizeof(TPoint)), *ptse = pts + nExamples;
    double *sumi, *sum = (double *)malloc(nExamples * sizeof(double)), *sume = sum + nExamples;
    TPoint *Ki, *K = (TPoint *)malloc(nExamples * sizeof(TPoint)), *Ke = K + nExamples;

    for(sumi = sum, Xi = X; sumi != sume; sumi++) {
      *sumi = 0.0;
      for(i = nAttrs; i--; *sumi += *Xi++);
      if (*sumi == 0.0)
        *sumi = 1.0;
    }

    while (steps--) {
      for(sumi = sum, Xi = X, ptsi = pts; sumi != sume; sumi++, ptsi++) {
        ptsi->x = ptsi->y = 0.0;
        for(anci = anc; anci != ance; anci++, Xi++) {
          ptsi->x += *Xi * anci->x;
          ptsi->y += *Xi * anci->y;
        }
        ptsi->x /= *sumi;
        ptsi->y /= *sumi;
      }


/* XXX   TO OPTIMIZE:
       - use pointers instead of indexing
       - sort the examples by classes so you don't have to compare them all the time
       - if attractive force are 0, you can gain some time by checking only the examples from different classes
*/
                     
      for(Ki = K; Ki != Ke; Ki++)
        Ki->x = Ki-> y = 0.0;

      for(dphii = dphi; dphii != dphie; dphii++)
        *dphii = 0.0;

      for(u = 0; u < nExamples; u++) {
        for(v = u+1; v < nExamples; v++) {
          const double duv = classes[u] == classes[v] ? attractG : repelG;
          if (duv == 0.0)
            continue;

          const double dx = pts[u].x - pts[v].x;
          const double dy = pts[u].y - pts[v].y;
          double ruv = sqr(dx) + sqr(dy);
          if (ruv < 1e-15)
            ruv = 1e-15;

          double druv;
          switch(law) {
            case TPNN::InverseLinear: druv = duv / sqrt(ruv); break;
            case TPNN::InverseSquare: druv = duv / ruv; break;
            case TPNN::InverseExponential: druv = duv / exp(-ruv); break;
          }

          const double Kefx = dx * druv;
          K[u].x += Kefx;
          K[v].x -= Kefx;

          const double Kefy = dy * druv;
          K[u].y += Kefy;
          K[v].y -= Kefy;
        }

        K[u].x /= sum[u];
        K[u].y /= sum[u];

        double *Xu = X + u * nAttrs;
        for(i = 0; i < nAttrs; i++, Xu++)
          dphi[i] -= *Xu/sum[u] * (K[u].y * anc[i].x - K[u].x * anc[i].y);
      }


  // Scale the changes - normalize the jumps
      double scaling = 1e10;
      for(dphii = dphi; dphii != dphie ; dphii++) {
        if (fabs(*dphii * scaling) > 0.1)
          scaling = fabs(0.01 / *dphii);
      }

  // Move anchors
      for(anci = anc, dphii = dphi; dphii != dphie; dphii++, anci++) {
        double tphi = atan2(anci->y, anci->x) + *dphii * scaling;
        anci->x = cos(tphi);
        anci->y = sin(tphi);
      }
    }

    anchors = PyList_New(nAttrs);
    for(i = 0, anci = anc, lli = ll;i < nAttrs; lli++, i++, anci++)
      PyList_SetItem(anchors, i, *ll ? Py_BuildValue("ddO", anci->x, anci->y, *lli) : Py_BuildValue("dd", anci->x, anci->y));
      
    double E = computeEnergyLow(nExamples, nAttrs, X, classes, pts, anc, sum, attractG, repelG, law);

    free(anc);
    free(dphi);
    free(pts);
    free(ll);
    free(classes);
    free(X);
    free(sum);
    free(K);

    return Py_BuildValue("Od", anchors, E);
      
  PyCATCH;
}


PyObject *optimizeAnchorsR(PyObject *, PyObject *args, PyObject *keywords) PYARGS(METH_VARARGS, "(scaledData, classes, anchors[, attractG=1.0, repelG=-1.0]) -> new-anchors")
{
  PyTRY
    PyObject *scaledData;
    PyObject *pyclasses;
    PyObject *anchors;
    PyObject *pyattrIndices;
    double attractG = 1.0, repelG = -1.0;
    int steps = 1;

    if (!PyArg_ParseTuple(args, "OOOO|ddi:optimizeAnchors", &scaledData, &pyclasses, &anchors, &pyattrIndices, &attractG, &repelG, &steps))
      return NULL;

    double *Xi, *X;
    int *classes;;
    TPoint *anci, *anc, *ance;
    PyObject **lli, **ll;
    int nAttrs, nExamples;

    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, nAttrs, nExamples, X, classes, anc, ll))
      return PYNULL;

    ance = anc + nAttrs;

    int i, u, v;
    double *radi, *rad = (double *)malloc(nAttrs * sizeof(double)), *rade = rad + nAttrs;
    double *dri, *dr = (double *)malloc(nAttrs * sizeof(double)), *dre = dr + nAttrs;
    TPoint *ptsi, *pts = (TPoint *)malloc(nExamples * sizeof(TPoint)), *ptse = pts + nExamples;
    double *sumi, *sum = (double *)malloc(nExamples * sizeof(double)), *sume = sum + nExamples;
    TPoint *Ki, *K = (TPoint *)malloc(nExamples * sizeof(TPoint)), *Ke = K + nExamples;

    while (steps--) {
      for(anci = anc, radi = rad; anci != ance; anci++, radi++)
        *radi = sqrt(sqr(anci->x) + sqr(anci->y));

      for(sumi = sum, Xi = X, ptsi = pts; sumi != sume; sumi++, ptsi++) {
        ptsi->x = ptsi->y = *sumi = 0.0;
        for(anci = anc, radi = rad; anci != ance; anci++, Xi++, radi++) {
          ptsi->x += *Xi * anci->x;
          ptsi->y += *Xi * anci->y;
          *sumi += *Xi * *radi;
        }
        if (*sumi == 0.0)
          *sumi = 1.0;
        ptsi->x /= *sumi;
        ptsi->y /= *sumi;
      }


/* XXX   TO OPTIMIZE:
       - use pointers instead of indexing
       - sort the examples by classes so you don't have to compare them all the time
       - if attractive force are 0, you can gain some time by checking only the examples from different classes
*/
                     
      for(Ki = K; Ki != Ke; Ki++)
        Ki->x = Ki-> y = 0.0;

      for(dri = dr; dri != dre; *(dri++) = 0.0);

      for(u = 0; u < nExamples; u++) {
        for(v = u+1; v < nExamples; v++) {
          const double duv = classes[u] == classes[v] ? attractG : repelG;
          if (duv == 0.0)
            continue;

          double ruv = sqr(pts[u].x - pts[v].x) + sqr(pts[u].y - pts[v].y);
          if (ruv < 1e-15)
            ruv = 1e-15;

          //const double druv = duv / exp(0.3 * log(ruv));
          //const double druv = duv / sqrt(ruv);
          const double druv = duv / ruv;

          const double druvx = druv * (pts[u].x - pts[v].x);
          K[u].x += druvx;
          K[v].x -= druvx;

          const double druvy = druv * (pts[u].y - pts[v].y);
          K[u].y += druvy;
          K[v].y -= druvy;
        }

        K[u].x /= sum[u];
        K[u].y /= sum[u];

        double *Xu = X + u * nAttrs;
        for(i = 0; i < nAttrs; i++, Xu++)
          dr[i] -= *Xu/sum[u] * (K[u].x * anc[i].x + K[u].y * anc[i].y);
      }


      double scaling = 1e10;
      for(anci = anc, dri = dr; dri != dre; anci++, dri++) {
        double maxdr = 0.1 * sqrt(sqr(anci->x) + sqr(anci->y));
        if ((maxdr > 1e-5) && (*dri > 1e-5)) {
          if (scaling * *dri > maxdr)
            scaling = maxdr / *dri;
        }
      }

      for(dri = dr; dri != dre; *dri++ *= scaling);

  // Move anchors
      for(anci = anc, dri = dr, radi = rad; dri != dre; dri++, anci++, radi++) {
        double rat = (*radi + *dri) / *radi;
        anci->x *= rat;
        anci->y *= rat;
      }

       //Centering and scaling
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


      double maxr = 0.0;
      for(anci = anc; anci != ance; anci++) {
        const double r = sqr(anci->x) + sqr(anci->y);
        if (r > maxr)
          maxr = r;
      }

      if (maxr > 0) {
        maxr = sqrt(maxr);
        for(anci = anc; anci != ance; anci++) {
          anci->x /= maxr;
          anci->y /= maxr;
        }
      }
     }

    anchors = PyList_New(nAttrs);
    for(i = 0, anci = anc, lli = ll;i < nAttrs; lli++, i++, anci++)
      PyList_SetItem(anchors, i, *ll ? Py_BuildValue("ddO", anci->x, anci->y, *lli) : Py_BuildValue("dd", anci->x, anci->y));
      
    double E = computeEnergyLow(nExamples, nAttrs, X, classes, pts, anc, sum, attractG, repelG, 0);

    free(anc);
    free(dr);
    free(pts);
    free(ll);
    free(classes);
    free(X);
    free(sum);
    free(K);

    return Py_BuildValue("Od", anchors, E);
      
  PyCATCH;
}



#define nColors 6
#ifdef _MSC_VER
#pragma warning (disable: 4305 4309)
#endif

PyObject *potentialsBitmap(PyObject *, PyObject *args, PyObject *) PYARGS(METH_VARARGS, "(P2NN, rx, ry, cellsize, scaleFactor) -> bitmap as string")
{
  PyTRY
    PyObject *cls;
    int rx, ry, cell;
    double scaleFactor;
    if (!PyArg_ParseTuple(args, "Oiiid:potentialsBitmap", &cls, &rx, &ry, &cell, &scaleFactor))
      return PYNULL;

    TP2NN *tp2nn = &dynamic_cast<TP2NN &>(PyOrange_AsOrange(cls).getReference());
    const int nClasses = tp2nn->classVar->noOfValues();
    const int nShades = 255/nClasses;

    const int oneLine = (2*rx + 3) & 0xfffffffc;
    const int bitmapSize = oneLine * 2*ry;
    char *bitmap = new char[bitmapSize];
    memset(bitmap, 255, bitmapSize);
    char *bitmapmid = bitmap + oneLine*ry + rx;

    float *probs = new float[nClasses], *pe = probs + nClasses;

    const double rxbysf = rx*scaleFactor;
    for(int y = -ry+1; y < ry-1; y+=cell) {
      const double yry = double(y)/ry;
      const double realy = yry/scaleFactor;
      int xe = ceil(rx * sqrt(1.0 - yry*yry));
      xe += cell - xe % cell;
      for(int x = -xe; x < xe; x+=cell) {
        const double realx = x/rxbysf;
        tp2nn->classDistribution(realx, -realy, probs, nClasses);
        double sprobs = *probs;
        float *largest = probs;
        for(float *pi = probs+1; pi != pe; pi++) {
          sprobs += *pi;
          if (*pi > *largest)
            largest = pi;
        }
        unsigned char color = floor(0.5 + nShades * (*largest/sprobs*nClasses - 1) / (nClasses - 1));
        if (color >= nShades)
          color = nShades - 1;
        else if (color < 0)
          color = 0;
        color += nShades * (largest - probs);
/*
        const int ys = y+cell < ry ? cell : ry-y;
        for(char *yy = bitmapmid + y*oneLine + x, *yye = yy + ys*oneLine; yy < yye; yy += oneLine)
          memset(yy, color, cell);
*/
        const int ys = y+cell < ry ? cell : ry-y;
        char *yy = bitmapmid + y*oneLine+x;
        memset(yy, color, cell);
        yy += oneLine;
        for(char *yye = yy + (ys-1)*oneLine; yy < yye; yy += oneLine)
          *yy = yy[cell-1] = color;
        memset(yy, color, cell);
      }
    }

    return Py_BuildValue("s#i", bitmap, bitmapSize, nShades);

  PyCATCH
}
