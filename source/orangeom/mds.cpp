#include "ppp/mds.ppp"
#include <math.h>
#include <iostream>
#include <stdlib.h>
using namespace std; 

void resize(PFloatListList &array, int dim1, int dim2){
	//cout<<"allocating"<<endl;
	array=mlnew TFloatListList();
	//cout << "resize"<<dim1<<endl;
    array->resize(dim1);
	//cout << array->size()<<endl;
	for(int i=0;i<dim1;i++){
		//cout<<"allocating"<<endl;
        array->at(i)=new TFloatList();
		array->at(i)->resize(dim2);
		//cout << array->at(i)->size()<<endl;
	}
}

TMDS::TMDS(PSymMatrix matrix, int dim=2){
    //resize(orig, size, size);
    distances=matrix;
    this->dim=dim;
    n=matrix->dim;
    projectedDistances=mlnew TSymMatrix(n);
    stress=mlnew TSymMatrix(n);
    resize(points, n, dim);
	freshD=false;
	avgStress=numeric_limits<float>::max();
}

void TMDS::SMACOFstep(){
    PSymMatrix R = mlnew TSymMatrix(n);
    getDistances();
    double sum=0, s, t;
	int i;
    for(i=0;i<n;i++){
        sum=0;
        for(int j=0;j<n;j++){
            if(j==i)
                continue;
            //if(projectedDistances->getitem(i,j)>1e-6)
                s=1.0/MMAX((double)projectedDistances->getitem(i,j),1e-6);
            //else
            //    s=0.0;
            t=distances->getitem(i,j)*s;
            R->getref(i,j)=-t;
            sum+=t;
        }
        R->getref(i,i)=sum;
    }
    //cout<<"SMCOFstep /2"<<endl;
    PFloatListList newPoints;
    resize(newPoints, n, dim);
    for(i=0;i<n;i++){
        for(int j=0;j<dim;j++){
            sum=0;
            for(int k=0;k<n;k++)
                sum+=R->getitem(i,k) * points->at(k)->at(j);
            newPoints->at(i)->at(j)=sum/n;
        }
	}
    points=newPoints;
	freshD=false;
	//cout <<"SMACOF out"<<endl;
}

void TMDS::getDistances(){
	float sum=0;
	if(freshD)
		return;
	for(int i=0; i<n; i++){
		for(int j=0; j<i; j++){
			sum=0.0;
			for(int k=0; k<dim; k++)
				sum+=pow(points->at(i)->at(k)-points->at(j)->at(k), 2);
			//cout<<"distance sq: "<<sum<<endl;
            projectedDistances->getref(i,j)=pow(sum,0.5f);
		}
        projectedDistances->getref(i,i)=0.0;
	}
	freshD=true;
}

float TMDS::getStress(PStressFunc fun){
	float s=0, total=0;
	if(!fun)
		return 0.0;
	//cout<<"getStress"<<endl;
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
            s=fun->operator()(projectedDistances->getitem(i,j), distances->getitem(i,j));
            stress->getref(i,j)=s;
			//cout<<"stress: "<<s<<endl;
			total+=(s<0)? -s: s;
		}
        stress->getref(i,i)=0;
	}
	avgStress=total/(n*n);
	//cout<<"avg. stress: "<<avgStress<<endl;
	return avgStress;
}

void TMDS::optimize(int numIter, PStressFunc fun, float eps){
	int iter=0;
	if(!fun)
		fun=mlnew TSgnRelStress();
	float oldStress=getStress(fun);
	float stress=oldStress*(1+eps*2);
	while(iter++<numIter){
		SMACOFstep();
        getDistances();
		stress=getStress(fun);
		if(fabs(oldStress-stress)<oldStress*eps)
			break;
		if(progressCallback)
			if(!progressCallback->call(float(iter)/float(numIter)))
				break;
		oldStress=stress;
	}
}



#include "externs.px"


/*************** PYTHON INTERFACE ***************/

#include "externs.px"
#include "orange_api.hpp"

C_NAMED(MDS, Orange, "(distanceMatrix [dim, points])->MDS")
BASED_ON(StressFunc, Orange)
C_CALL(KruskalStress, StressFunc, "(float, float[,float])->float")
C_CALL(SammonStress, StressFunc, "(float, float[,float])->float")
C_CALL(SgnSammonStress, StressFunc, "(float, float[,float])->float")
C_CALL(SgnRelStress, StressFunc, "(float, float[,float])->float")
C_CALL(StressFunc_Python, StressFunc,"")


PyObject* StressFunc_new(PyTypeObject *type, PyObject *args, PyObject *kwds) BASED_ON(Orange, "<abstract>")
{ if (type == (PyTypeObject *)&PyOrStressFunc_Type)
    return setCallbackFunction(WrapNewOrange(mlnew TStressFunc_Python(), type), args);
  else
    return WrapNewOrange(mlnew TStressFunc_Python(), type);
}

PyObject *StressFunc__reduce__(PyObject *self)
{
  return callbackReduce(self, PyOrStressFunc_Type);
}

extern ORANGEOM_API PyObject *orangeomModule;

PyObject *MDS_new(PyTypeObject *type, PyObject *args) BASED_ON(Orange, "(dissMatrix[, dim, points])")
{
    PyTRY
    int dim=2;
	float avgStress=-1.0;
    PSymMatrix matrix;
    PFloatListList points;
    if(!PyArg_ParseTuple(args, "O&|iO&f", cc_SymMatrix, &matrix, &dim, cc_FloatListList, &points, &avgStress))
        return NULL;

    PMDS mds=mlnew TMDS(matrix, dim);
    if(points && points->size()==matrix->dim)
        mds->points=points;
    else{
        PRandomGenerator rg=mlnew TRandomGenerator(0);
        for(int i=0;i<mds->n; i++)
            for(int j=0; j<mds->dim; j++)
                mds->points->at(i)->at(j)=rg->randfloat();
    }
	if(avgStress!=-1.0)
		mds->avgStress=avgStress;

    return WrapOrange(mds);
    PyCATCH
}

PyObject *MDS__reduce__(PyObject *self)
{
  PyTRY
    CAST_TO(TMDS, mds);

    return Py_BuildValue("O(NiNf)N", self->ob_type,
                                       WrapOrange(mds->distances),
                                       mds->dim,
                                       WrapOrange(mds->points),
                                       mds->avgStress,
                                       packOrangeDictionary(self));
  PyCATCH
}

/*
PyObject *MDS__reduce__(PyObject *self)
{
  PyTRY
    CAST_TO(TMDS, mds);

    return Py_BuildValue("O(ONiNif)N", getExportedFunction(orangeomModule, "__pickleLoaderMDS"),
                                       self->ob_type,
                                       WrapOrange(mds->distances),
                                       mds->dim,
                                       mds->points,
                                       mds->freshD ? 1 : 0,
                                       mds->avgStress,
                                       packOrangeDictionary(self));
  PyCATCH
}


PyObject* *__pickle&LoaderMDS(PyObject *, PyObject *args) PY__*&ARGS(METH_VARARGS, "(type, distances, dim, points, freshD, avgStress)")
{
  PyTRY
    PyTypeObject *type;
    PyObject *pydistances, *pydim, *pypoints;
    int freshD;
    float avgStress;
    if (!PyArg_ParseTuple(args, "OOOOif:__pickleLoaderMDS", &type, &pydistances, &pydim, &pypoints, &freshD, &avgStress))
      return NULL;

    // SET_ITEM steals references
    PyObject *newargs = PyTuple_New(3);
    PyTuple_SET_ITEM(newargs, 0, pydistances);
    PyTuple_SET_ITEM(newargs, 1, pydim);
    PyTuple_SET_ITEM(newargs, 2, pypoints);

    PyObject *pymds = MDS_new(type, newargs);
    Py_DECREF(newargs);
    if (!pymds) {
      Py_DECREF((PyObject *)type);
      return NULL;
    }

    PMDS mds = PyOrange_AsMDS(pymds);
    mds->freshD = freshD != 0;
    mds->avgStress = avgStress;

    return pymds;
  PyCATCH;
}
*/




PyObject *MDS_SMACOFstep(PyTypeObject  *self) PYARGS(METH_NOARGS, "()")
{
    PyTRY
    SELF_AS(TMDS).SMACOFstep();
    RETURN_NONE;
    PyCATCH
}

PyObject *MDS_getDistance(PyTypeObject *self) PYARGS(METH_NOARGS, "()")
{
    PyTRY
    SELF_AS(TMDS).getDistances();
    RETURN_NONE;
    PyCATCH
}

PyObject *MDS_getStress(PyTypeObject *self, PyObject *args) PYARGS(METH_VARARGS, "([stressFunc=SgnRelStress])")
{
    PyTRY
    PStressFunc sf;
    PyObject *callback=NULL;
    if(PyTuple_Size(args)==1){
        /*
        if(!PyArg_ParseTuple(args, "O&", cc_StressFunc, &sf))
            if(!(PyArg_ParseTuple(args, "O", &callback) &&
                (sf=PyOrange_AsStressFunc(mysetCallbackFunction(WrapNewOrange(mlnew TStressFunc_Python(),
                (PyTypeObject*)&PyOrStressFunc_Type), args)))))
                return NULL;
                */
        sf=PyOrange_AsStressFunc(StressFunc_new((PyTypeObject*)&PyOrStressFunc_Type, args, NULL));
        SELF_AS(TMDS).getStress(sf);
    }else
        SELF_AS(TMDS).getStress(mlnew TSgnRelStress());
    RETURN_NONE;
    PyCATCH
}

PyObject *MDS_optimize(PyObject* self, PyObject* args, PyObject* kwds) PYARGS(METH_VARARGS, "(numSteps[, stressFunc=orangemds.SgnRelStress, progressCallback=None])->None")
{
    PyTRY
    int iter;
    float eps=1e-3f;
    PProgressCallback callback;
    PStressFunc stress;
    PyObject *pyStress=NULL;
    if(!PyArg_ParseTuple(args, "i|O&f", &iter, cc_StressFunc, &stress, &eps)) {
        PyErr_Clear();
        if(PyArg_ParseTuple(args, "i|Of", &iter, &pyStress, &eps) && pyStress){
            PyObject *arg=Py_BuildValue("(O)", pyStress);
            stress=PyOrange_AsStressFunc(StressFunc_new((PyTypeObject*)&PyOrStressFunc_Type, arg, NULL));
        } else
            return NULL;
    }

    SELF_AS(TMDS).optimize(iter, stress, eps);
    RETURN_NONE;
    PyCATCH
}

PyObject *StressFunc_call(PyTypeObject *self, PyObject *args)
{
    PyTRY
    float cur, cor, w;
    if(!PyArg_ParseTuple(args, "ff|f:KruskalStress.__call__", &cur, &cor, &w))
        return NULL;
    if(PyTuple_Size(args)==2)
        return Py_BuildValue("f",SELF_AS(TStressFunc).operator ()(cur,cor));
    else
        return Py_BuildValue("f",SELF_AS(TStressFunc).operator ()(cur, cor, w));
    PyCATCH
}

#include "mds.px"
