#include "orange_api.hpp"
#include "examplegen.hpp"
#include "pnn.hpp"
#include "../orange/px/externs.px"

//#include "rconversions.hpp"

inline double sqr(const double &x)
{ return x*x; }


typedef struct {double x, y; } TPoint;


float computeEnergyLow(const int &nExamples, const int &nAttrs, const int &contClass, double *X, int *classes, TPoint *pts, TPoint *anc, double *sum, const double attractG, const double repelG, const int law)
{
  double *Xi;
  double *sumi, *sume = sum + nExamples;
  TPoint *anci, *ance = anc + nAttrs;
  TPoint *ptsi, *ptsj, *ptse = pts + nExamples, *ptsie;

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

  if (contClass) {
    double *dclassesi, *dclassesj;
    double *&dclasses = (double *&)classes;

    switch(law) {
      case TPNN::InverseLinear:
        for(ptsi = pts, dclassesi = dclasses; ptsi != ptse; ptsi++, dclassesi++)
          for(ptsj = pts, dclassesj = dclasses; ptsj != ptsi; ptsj++, dclassesj++) {
            const double dist = sqr(ptsi->x - ptsj->x) + sqr(ptsi->y - ptsj->y);
            E -= fabs(*dclassesi - *dclassesj) * log(dist < 1e-15 ? 1e-15 : dist);
          }
        E /= 2.0; // this is needed since we omitted a sqrt inside log or inside .5x^2
        break;

      case TPNN::InverseSquare:
        for(ptsi = pts, dclassesi = dclasses; ptsi != ptse; ptsi++, dclassesi++)
          for(ptsj = pts, dclassesj = dclasses; ptsj != ptsi; ptsj++, dclassesj++) {
            double dist = sqr(ptsi->x - ptsj->x) + sqr(ptsi->y - ptsj->y);
            E += fabs(*dclassesi - *dclassesj) / (dist < 1e-15 ? 1e-15 : sqrt(dist));
          }
        break;

      case TPNN::InverseExponential:
        for(ptsi = pts, dclassesi = dclasses; ptsi != ptse; ptsi++, dclassesi++)
          for(ptsj = pts, dclassesj = dclasses; ptsj != ptsi; ptsj++, dclassesj++) {
            E += fabs(*dclassesi - *dclassesj) / exp(-sqrt(sqr(ptsi->x - ptsj->x) + sqr(ptsi->y - ptsj->y)));
          }
        break;

    }
  }

  else {
    for(int *classesi = classes; classesi[1]; classesi++) {
      if (attractG != 0.0) {
        for(ptsi = pts + *classesi, ptsie = pts + classesi[1]; ptsi != ptsie; ptsi++)
          for(ptsj = pts + *classesi; ptsj != ptsi; ptsj++) {
            const double dist = sqr(ptsi->x - ptsj->x) + sqr(ptsi->y - ptsj->y);
            if (dist > 1e-15)
              E += 150 * attractG * exp(1.5 * log(dist))/3.0;
          }
      }

      if ((repelG != 0.0) && classesi[2]) {
        for(ptsi = pts + *classesi, ptsie = pts + classesi[1]; ptsi != ptsie; ptsi++)
          for(ptsj = ptsie; ptsj != ptse; ptsj++) {
            double dist = sqr(ptsi->x - ptsj->x) + sqr(ptsi->y - ptsj->y);
            E -= repelG / (dist < 1e-15 ? 1e-15 : sqrt(dist));
          }

      }
    }
  }

  return E;
}



bool loadRadvizData(PyObject *scaledData, PyObject *pyclasses, PyObject *anchors, PyObject *pyattrIndices,
                    int &nAttrs, int &nExamples, int &contClass,
                    double *&X, int *&classes, TPoint *&anc, PyObject **&ll,
                    double &minClass, double &maxClass)
{
  if (!PyList_Check(scaledData) || !PyList_Check(pyclasses) || !PyList_Check(anchors))
    PYERROR(PyExc_TypeError, "scaled data, classes and anchors should be given a lists", false);

  nAttrs = PyList_Size(anchors);
  nExamples = PyList_Size(scaledData);

  X = (double *)malloc(nExamples * nAttrs * sizeof(double));
  classes = (int *)malloc(nExamples * (contClass ? sizeof(double) : sizeof(int)));
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

  if (contClass) {
    double *dclassesi;
    for(dclassesi = (double *)classes, i = 0; i < nExamples; *dclassesi++ = PyFloat_AsDouble(PyList_GetItem(pyclasses, i++)));

    for(Xi = X, i = 0; i < nExamples; i++) {
      PyObject *ex = PyList_GetItem(scaledData, i);
      for(aii = attrIndices; aii < aie; aii++)
        *Xi++ = PyFloat_AsDouble(PyList_GetItem(ex, *aii));
    }
  }

  else {
    int *classesi, *classese;
    int maxCls = 0;
    for(classesi = classes, i = 0; i < nExamples; classesi++) {
      *classesi = PyInt_AsLong(PyList_GetItem(pyclasses, i++));
      if (*classesi > maxCls)
        maxCls = *classesi;
    }

    // we need maxCls+3: beginning of each of maxCls+1 classes, end of the last class, sentinel
    int *rcls = (int *)malloc((maxCls+3) * sizeof(int));
    memset(rcls, 0, (maxCls+3) * sizeof(int));
    for(classesi = classes, classese = classes+nExamples; classesi != classese; rcls[1 + *classesi++]++);
    for(int *rclsi = rcls+1, *rclse = rcls+maxCls+2; rclsi != rclse; *rclsi += rclsi[-1], rclsi++);

    for(classesi = classes, i = 0; i < nExamples; i++, classesi++) {
      PyObject *ex = PyList_GetItem(scaledData, i);
      Xi = X + nAttrs * rcls[*classesi]++;
      for(aii = attrIndices; aii < aie; aii++)
        *Xi++ = PyFloat_AsDouble(PyList_GetItem(ex, *aii));
    }

    memmove(rcls+1, rcls, (maxCls+1) * sizeof(int));
    *rcls = 0;

    free(classes);
    classes = rcls;
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
    int contClass = 0;

    if (!PyArg_ParseTuple(args, "OOOO|ddii:computeEnergy", &scaledData, &pyclasses, &anchors, &pyattrIndices, &attractG, &repelG, &law, &contClass))
      return NULL;

    double *X, *Xi;
    int *classes;
    TPoint *anci, *anc, *ance;
    PyObject **ll;
    int nAttrs, nExamples;
    double minClass, maxClass;

    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, nAttrs, nExamples, contClass, X, classes, anc, ll, minClass, maxClass))
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

    double E = computeEnergyLow(nExamples, nAttrs, contClass, X, classes, pts, anc, sum, attractG, repelG, law);

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
    int contClass = 0;

    if (!PyArg_ParseTuple(args, "OOOO|ddiiii:optimizeAnchors", &scaledData, &pyclasses, &anchors, &pyattrIndices, &attractG, &repelG, &law, &steps, &normalizeExamples, &contClass))
      return NULL;

    double *Xi, *X;
    int *classes;
    TPoint *anci, *anc, *ance;
    PyObject **lli, **ll;
    int nAttrs, nExamples;

    double minClass, maxClass;
    double *&dclasses = (double *&)classes;

    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, nAttrs, nExamples, contClass, X, classes, anc, ll, minClass, maxClass))
      return PYNULL;

    ance = anc + nAttrs;

    int i;
    double *radi, *rad = (double *)malloc(nAttrs * sizeof(double)), *rade = rad + nAttrs;
    TPoint *danci, *danc = (TPoint *)malloc(nAttrs * sizeof(TPoint)), *dance = danc + nAttrs;
    TPoint *ptsi, *pts = (TPoint *)malloc(nExamples * sizeof(TPoint)), *ptse = pts + nExamples, *ptsi2, *ptsie;
    double *sumi, *sum = (double *)malloc(nExamples * sizeof(double)), *sume = sum + nExamples;
    TPoint *Fai, *Fa = (TPoint *)malloc(nExamples * sizeof(TPoint)), *Fae = Fa + nExamples, *Fai2;
    TPoint *Fri, *Fr = (TPoint *)malloc(nExamples * sizeof(TPoint)), *Fre = Fr + nExamples, *Fri2;
    double FaTot, FrTot;

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


      for(Fai = Fa; Fai != Fae; Fai++)
        Fai->x = Fai-> y = 0.0;

      for(Fri = Fr; Fri != Fre; Fri++)
        Fri->x = Fri-> y = 0.0;

      FaTot = FrTot = 0;

      for(danci = danc; danci != dance; danci++)
        danci->x = danci->y = 0.0;


      if (contClass) {
        double *classesi, *classesi2;
        for(ptsi = pts, Fri = Fr, classesi = (double *)classes; ptsi != ptse; ptsi++, Fri++, classesi++)
          for(ptsi2 = pts, Fri2 = Fr, classesi2 = (double *)classes; ptsi2 != ptsi; ptsi2++, Fri2++, classesi2++) {
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
            Fri->x  += FrX;
            Fri2->x -= FrX;
            Fri->y  += FrY;
            Fri2->y -= FrY;
        }
      }

      else {
        for(int *classesi = classes; classesi[1]; classesi++) {

          if (attractG != 0.0) {
            
            for(ptsi = pts + *classesi, ptsie = pts + classesi[1], Fai = Fa + *classesi; ptsi != ptsie; ptsi++, Fai++)
              for(ptsi2 = pts + *classesi,                         Fai2 = Fa + *classesi; ptsi2 != ptsi; ptsi2++, Fai2++) {
                const double dx = ptsi->x - ptsi2->x;
                const double dy = ptsi->y - ptsi2->y;
                const double r2 = sqr(dx) + sqr(dy);
                const double sr2 = sqrt(r2);
                if (r2 < 1e-15)
                  continue;
                const double TFa = 100 * attractG * r2;
                FaTot += TFa;
                const double druvx =  TFa * dx / sr2;
                const double druvy = TFa * dy / sr2;
                Fai->x  += druvx;
                Fai2->x -= druvx;
                Fai->y  += druvy;
                Fai2->y -= druvy;
              }
          }

          if ((repelG != 0.0) && classesi[2]) {
            for(ptsi = pts + *classesi, ptsie = pts + classesi[1], Fri = Fr + *classesi; ptsi != ptsie; ptsi++, Fri++)
              for(ptsi2 = ptsie, Fri2 = Fr + classesi[1]; ptsi2 != ptse; ptsi2++, Fri2++) {
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

                const double TFr = repelG / r2;
                FrTot += TFr;
                const double druvx = TFr * fx;
                Fri->x  += druvx;
                Fri2->x -= druvx;

                const double druvy = TFr * fy;
                Fri->y  += druvy;
                Fri2->y -= druvy;
              }
          }
        }
      }

      if (normalizeExamples) {
        for(Fai = Fa, Fri = Fr, sumi = sum, Xi = X; Fri != Fre; Fai++, Fri++, sumi++) {
          Fai->x /= *sumi;
          Fai->y /= *sumi;
          Fri->x /= *sumi;
          Fri->y /= *sumi;
          for(danci = danc; danci != dance; danci++, Xi++) {
            danci->x -= (Fai->x+Fri->x) * *Xi; // previously, Xi was here additionally divided by *sumi -- don't know why
            danci->y -= (Fai->y+Fri->y) * *Xi;
          }
        }
      }

      else {
        const double k =  /*FrTot > 0.001 ? fabs(FaTot / FrTot) : */1.0;
        for(Fai = Fa, Fri = Fr, Xi = X; Fri != Fre; Fri++, Fai++) {
          for(danci = danc; danci != dance; danci++, Xi++) {
            danci->x -= (Fai->x+k*Fri->x) * *Xi;
            danci->y -= (Fai->y+k*Fri->y) * *Xi;
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

    double E = computeEnergyLow(nExamples, nAttrs, contClass, X, classes, pts, anc, sum, attractG, repelG, law);

    free(anc);
    free(danc);
    free(pts);
    free(ll);
    free(classes);
    free(X);
    free(sum);
    free(Fa);
    free(Fr);
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
    int contClass = 0;

    if (!PyArg_ParseTuple(args, "OOOO|ddiii:optimizeAnchors", &scaledData, &pyclasses, &anchors, &pyattrIndices, &attractG, &repelG, &law, &steps, &contClass))
      return NULL;

    double *Xi, *X;
    int *classes;;
    TPoint *anci, *anc, *ance;
    PyObject **lli, **ll;
    int nAttrs, nExamples;

    double minClass, maxClass;

    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, nAttrs, nExamples, contClass, X, classes, anc, ll, minClass, maxClass))
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
            case TPNN::InverseExponential: druv = duv / exp(-sqrt(ruv)); break;
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
      
    double E = computeEnergyLow(nExamples, nAttrs, contClass, X, classes, pts, anc, sum, attractG, repelG, law);

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
    int contClass = 0;

    if (!PyArg_ParseTuple(args, "OOOO|ddii:optimizeAnchors", &scaledData, &pyclasses, &anchors, &pyattrIndices, &attractG, &repelG, &steps, &contClass))
      return NULL;

    double *Xi, *X;
    int *classes;;
    TPoint *anci, *anc, *ance;
    PyObject **lli, **ll;
    int nAttrs, nExamples;
    double minClass, maxClass;

    if (!loadRadvizData(scaledData, pyclasses, anchors, pyattrIndices, nAttrs, nExamples, contClass, X, classes, anc, ll, minClass, maxClass))
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
      
    double E = computeEnergyLow(nExamples, nAttrs, contClass, X, classes, pts, anc, sum, attractG, repelG, 0);

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
    TPoint *ptsi, *pts = (TPoint *)malloc(nExamples * sizeof(TPoint)), *ptse = pts + nExamples, *ptsi2, *ptsie;
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

