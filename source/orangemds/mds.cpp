#include "ppp/mds.ppp"
#include "px/externs.px"
#include <math.h>
#include <iostream>
#include <stdlib.h>
using namespace std; 

DEFINE_TOrangeVector_classDescription(PFloatList, "TFloatListList", true, ORANGEMDS_API)

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
}

void TMDS::SMACOFstep(){
    PSymMatrix R = mlnew TSymMatrix(n);
    getDistances();
    float sum=0, s, t;
    for(int i=0;i<n;i++){
        sum=0;
        for(int j=0;j<n;j++){
            if(j==i)
                continue;
            if(projectedDistances->getitem(i,j)>1e-6)
                s=1.0/projectedDistances->getitem(i,j);
            else
                s=0.0;
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
	cout<<"getStress"<<endl;
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
	cout<<"avg. stress: "<<avgStress<<endl;
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
		if(abs(oldStress-stress)<oldStress*eps)
			break;
		if(progressCallback)
			if(!progressCallback->call(float(numIter)/float(iter)))
				break;
		oldStress=stress;
	}
}
