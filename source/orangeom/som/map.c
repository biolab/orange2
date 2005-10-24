#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "lvq_pak.h"
#include "datafile.h"
#include "umat.h"

/* read the map from the file 'mapfile' and 
   allocate required memory for the data structures */

struct umatrix *read_map (char *mapfile, int xswap, int yswap)
{
  int i,j;
  int row, topology;
  struct entries *codes;
  struct data_entry *dtmp;
  eptr p;
  struct umatrix *umat;

  if ((umat = alloc_umat()) == NULL)
    {
      fprintf(stderr, "read_map: can't allocate memory for umatrix\n");
      return NULL;
    }

  if ((codes = open_entries(mapfile)) == NULL)
    {
      fprintf(stderr, "Can't open code file %s\n", mapfile);
      free_umat(umat);
      return NULL;
    }

  umat->codes = codes;
  umat->dim = codes->dimension;
  umat->mxdim = codes->xdim;
  umat->mydim = codes->ydim;
  umat->topol = topology = codes->topol;

  umat->uxdim = 2*umat->mxdim-1;
  umat->uydim = 2*umat->mydim-1;

  /* allocate memory for the needed structures */
  umat->mvalue = (float ***)malloc(umat->mxdim*sizeof(float **));
  for (i = 0; i < umat->mxdim; i++) 
    {
      umat->mvalue[i] = (float **)malloc(umat->mydim*sizeof(float *));
    }

  /* allocate memory for the u-matrix structure */
  umat->uvalue = (float **)malloc(umat->uxdim*sizeof(float *));
  for (i=0;i<umat->uxdim;i++)
    umat->uvalue[i] = (float *)malloc(umat->uydim*sizeof(float));
  
  if ((dtmp = rewind_entries(codes, &p)) == NULL)
    {
      fprintf(stderr, "can't get data\n");
      return NULL;
    }

  row = 1;
  for (j=0;j<umat->mydim;j++)
    for (i=0;i<umat->mxdim;i++) 
      {
	/* read the weight vector of the map element [i][j] */
	umat->mvalue[i][j] = dtmp->points;

	dtmp = next_entry(&p);
      }
  
#if 0
  /* Swap the labels if needed */
  if (xswap) {
    char *tmp;
    for (i=0;i<mat->xdim;i++)
      for (j=0;j<mat->ydim/2;j++) {
	tmp = mat->label[i][j];
	mat->label[i][j] = mat->label[i][mat->ydim-1-j];
	mat->label[i][mat->ydim-1-j] = tmp;
      }
  }
  if (yswap) {
    char **tmp;
    for (i=0;i<mat->xdim/2;i++) {
      tmp = mat->label[i];
      mat->label[i] = mat->label[mat->xdim-1-i];
      mat->label[mat->xdim-1-i] = tmp;
    }
  }
#endif


  return(umat);
}

struct umatrix *alloc_umat(void)
{
  struct umatrix *umat;

  if ((umat = malloc(sizeof(struct umatrix))) == NULL)
    {
      fprintf(stderr, "read_map: can't allocate memory for umatrix\n");
      return NULL;
    }
  umat->mvalue = NULL;
  umat->uvalue = NULL;
  umat->codes = NULL;

  return umat;
}


int free_umat(struct umatrix *umat)
{
  int i;

  if (umat == NULL)
    return 0;

  if (umat->codes)
    close_entries(umat->codes);

  if (umat->mvalue)
    {
      for (i = 0; i < umat->mxdim; i++) 
	if (umat->mvalue[i])
	  free(umat->mvalue[i]);
      free(umat->mvalue);
    }

  if (umat->uvalue)
    {
      for (i = 0; i < umat->uxdim; i++)
	if (umat->uvalue[i])
	  free(umat->uvalue[i]);

      free(umat->uvalue);
    }

  free(umat);

  return 0;
}


/* Comparison function for the sort */
int compar(const void *first, const void *sec)
{
  if(*(double *)first < *(double *)sec) return -1;
  else return *(double *)first > *(double *)sec;
}

/* calculate the u-matrix */
int calc_umatrix(struct umatrix *umat,int xswap,int yswap)
{
  int i,j,k,count,bx,by,bz;
  double dx,dy,dz1,dz2,dz,temp,max=0,min=0, bw;
  double medtable[6];


  /* rectangular topology */
  if (umat->topol == TOPOL_RECT)
    {
      for (j=0;j<umat->mydim;j++)
	for (i=0;i<umat->mxdim;i++)
	  {
	    dx=0;dy=0;dz1=0;dz2=0;count=0;
	    bx=0;by=0;bz=0;
	    for (k=0;k<umat->dim;k++)
	      {
		
		if (i<(umat->mxdim-1))
		  {
		    temp = (umat->mvalue[i][j][k] - umat->mvalue[i+1][j][k]);
		    dx += temp*temp;
		    bx=1;
		  }
		if (j<(umat->mydim-1))
		  {
		    temp = (umat->mvalue[i][j][k] - umat->mvalue[i][j+1][k]);
		    dy += temp*temp;
		    by=1;
		  }
		
		if (j<(umat->mydim-1) && i<(umat->mxdim-1))
		  {
		    temp = (umat->mvalue[i][j][k] - umat->mvalue[i+1][j+1][k]);
		    dz1 += temp*temp;
		    temp = (umat->mvalue[i][j+1][k] - umat->mvalue[i+1][j][k]);
		    dz2 += temp*temp;
		    bz=1;
		  }
	      }
	    dz = (sqrt(dz1)/sqrt((double) 2.0)+sqrt(dz2)/sqrt((double) 2.0))/2;
	    
	    if (bx)
	      umat->uvalue[2*i+1][2*j] = sqrt(dx);
	    if (by)
	      umat->uvalue[2*i][2*j+1] = sqrt(dy);
	    if (bz)
	      umat->uvalue[2*i+1][2*j+1] = dz;
	  }
    }
  else
    /* hexagonal topology */
    {
      for (j=0;j<umat->mydim;j++)
	for (i=0;i<umat->mxdim;i++)
	  {
	    dx=0;dy=0;dz=0;count=0;
	    bx=0;by=0;bz=0;
	    
	    temp=0;
	    if (i<(umat->mxdim-1))
	      {
		for (k=0;k<umat->dim;k++)
		  {
		    temp = (umat->mvalue[i][j][k] - umat->mvalue[i+1][j][k]);
		    dx += temp*temp;
		    bx=1;
		  }
	      }
	    temp=0;
	    if (j<(umat->mydim-1))
	      {
		if (j%2) {
		  for (k=0;k<umat->dim;k++)
		    {
		      temp = (umat->mvalue[i][j][k] - umat->mvalue[i][j+1][k]);
		      dy += temp*temp;
		      by=1;
                    }
		}
		else {
		  if (i>0) {
		    for (k=0;k<umat->dim;k++)
		      {
			temp = (umat->mvalue[i][j][k] - 
				umat->mvalue[i-1][j+1][k]);
			dy += temp*temp;
			by=1;
		      }
		  }
		  else
		    temp=0;
		}

	      }
	    temp=0;
	    if (j<(umat->mydim-1))
	      {
		if (!(j%2)) {
		  for (k=0;k<umat->dim;k++)
		    {
		      temp = (umat->mvalue[i][j][k] - umat->mvalue[i][j+1][k]);
		      dz += temp*temp;
		    }
		   bz=1;
		}
		else
		  {
		    
		    if (i<(umat->mxdim-1)) {
		      for (k=0;k<umat->dim;k++){
			temp = (umat->mvalue[i][j][k] - 
				umat->mvalue[i+1][j+1][k]);
			dz += temp*temp;
		      }
		      bz=1;
		    }
		  }
	      }
	    else
	      temp=0;
	  
	    if (bx)
	      umat->uvalue[2*i+1][2*j] = sqrt(dx);
	    
	    if (by)
	      {
		if (j%2)
		  umat->uvalue[2*i][2*j+1] = sqrt(dy);
		else
		  umat->uvalue[2*i-1][2*j+1] = sqrt(dy);
	      }	    
	   	  
	    if (bz)
	      {
		if (j%2)
		  umat->uvalue[2*i+1][2*j+1] = sqrt(dz);
		else
		  umat->uvalue[2*i][2*j+1] = sqrt(dz);
	      }
	  }
    }

  /* Set the values corresponding to the model vectors themselves
     to medians of the surrounding values */
  if(umat->topol==TOPOL_RECT) {
    /* medians of the 4-neighborhood */
    for (j=0;j<umat->uydim;j+=2)
      for (i=0;i<umat->uxdim;i+=2)
	if(i>0 && j>0 && i<umat->uxdim-1 && j<umat->uydim-1) {
	  /* in the middle of the map */
	  medtable[0]=umat->uvalue[i-1][j];
	  medtable[1]=umat->uvalue[i+1][j];
	  medtable[2]=umat->uvalue[i][j-1];
	  medtable[3]=umat->uvalue[i][j+1];
	  qsort((void *)medtable, 4, sizeof(*medtable), 
		compar);
	  /* Actually mean of two median values */
	  umat->uvalue[i][j]=(medtable[1]+medtable[2])/2.0;
	} else if(j==0 && i>0 && i<umat->uxdim-1) {
	  /* in the upper edge */
	  medtable[0]=umat->uvalue[i-1][j];
	  medtable[1]=umat->uvalue[i+1][j];
	  medtable[2]=umat->uvalue[i][j+1];
	  qsort((void *)medtable, 3, sizeof(*medtable), 
		compar);
	  umat->uvalue[i][j]=medtable[1];
	} else if(j==umat->uydim-1 && i>0 && i<umat->uxdim-1) {
	  /* in the lower edge */
	  medtable[0]=umat->uvalue[i-1][j];
	  medtable[1]=umat->uvalue[i+1][j];
	  medtable[2]=umat->uvalue[i][j-1];
	  qsort((void *)medtable, 3, sizeof(*medtable), 
		compar);
	  umat->uvalue[i][j]=medtable[1];
	} else if(i==0 && j>0 && j<umat->uydim-1) {
	  /* in the left edge */
	  medtable[0]=umat->uvalue[i+1][j];
	  medtable[1]=umat->uvalue[i][j-1];
	  medtable[2]=umat->uvalue[i][j+1];
	  qsort((void *)medtable, 3, sizeof(*medtable), 
		compar);
	  umat->uvalue[i][j]=medtable[1];
	} else if(i==umat->uxdim-1 && j>0 && j<umat->uydim-1) {
	  /* in the right edge */
	  medtable[0]=umat->uvalue[i-1][j];
	  medtable[1]=umat->uvalue[i][j-1];
	  medtable[2]=umat->uvalue[i][j+1];
	  qsort((void *)medtable, 3, sizeof(*medtable), 
		compar);
	  umat->uvalue[i][j]=medtable[1];
	} else if(i==0 && j==0)
	  /* the upper left-hand corner */
	  umat->uvalue[i][j]=(umat->uvalue[i+1][j]+umat->uvalue[i][j+1])/2.0;
	else if(i==umat->uxdim-1 && j==0) {
	  /* the upper right-hand corner */
	  umat->uvalue[i][j]=(umat->uvalue[i-1][j]+umat->uvalue[i][j+1])/2.0;
	} else if(i==0 && j==umat->uydim-1) {
	  /* the lower left-hand corner */
	  umat->uvalue[i][j]=(umat->uvalue[i+1][j]+umat->uvalue[i][j-1])/2.0;
	} else if(i==umat->uxdim-1 && j==umat->uydim-1) {
	  /* the lower right-hand corner */
	  umat->uvalue[i][j]=(umat->uvalue[i-1][j]+umat->uvalue[i][j-1])/2.0;
	}
  } else   /* HEXA */
    for (j=0;j<umat->uydim;j+=2)
      for (i=0;i<umat->uxdim;i+=2)
	if(i>0 && j>0 && i<umat->uxdim-1 && j<umat->uydim-1) {
	  /* in the middle of the map */
	  medtable[0]=umat->uvalue[i-1][j];
	  medtable[1]=umat->uvalue[i+1][j];
	  if(!(j%4)) {
	    medtable[2]=umat->uvalue[i-1][j-1];
	    medtable[3]=umat->uvalue[i][j-1];
	    medtable[4]=umat->uvalue[i-1][j+1];
	    medtable[5]=umat->uvalue[i][j+1];
	  } else {
	    medtable[2]=umat->uvalue[i][j-1];
	    medtable[3]=umat->uvalue[i+1][j-1];
	    medtable[4]=umat->uvalue[i][j+1];
	    medtable[5]=umat->uvalue[i+1][j+1];
	  }
	  qsort((void *)medtable, 6, sizeof(*medtable), 
		compar);
	  /* Actually mean of two median values */
	  umat->uvalue[i][j]=(medtable[2]+medtable[3])/2.0;
	} else if(j==0 && i>0 && i<umat->uxdim-1) {
	  /* in the upper edge */
	  medtable[0]=umat->uvalue[i-1][j];
	  medtable[1]=umat->uvalue[i+1][j];
	  medtable[2]=umat->uvalue[i][j+1];
	  medtable[3]=umat->uvalue[i-1][j+1];
	  qsort((void *)medtable, 4, sizeof(*medtable), 
		compar);
	  /* Actually mean of two median values */
	  umat->uvalue[i][j]=(medtable[1]+medtable[2])/2.0;
	} else if(j==umat->uydim-1 && i>0 && i<umat->uxdim-1) {
	  /* in the lower edge */
	  medtable[0]=umat->uvalue[i-1][j];
	  medtable[1]=umat->uvalue[i+1][j];
	  if(!(j%4)) {
	    medtable[2]=umat->uvalue[i-1][j-1];
	    medtable[3]=umat->uvalue[i][j-1];
	  } else {
	    medtable[2]=umat->uvalue[i][j-1];
	    medtable[3]=umat->uvalue[i+1][j-1];
	  }
	  qsort((void *)medtable, 4, sizeof(*medtable), 
		compar);
	  /* Actually mean of two median values */
	  umat->uvalue[i][j]=(medtable[1]+medtable[2])/2.0;
	} else if(i==0 && j>0 && j<umat->uydim-1) {
	  /* in the left edge */
	  medtable[0]=umat->uvalue[i+1][j];
	  if(!(j%4)) {
	    medtable[1]=umat->uvalue[i][j-1];
	    medtable[2]=umat->uvalue[i][j+1];
	    qsort((void *)medtable, 3, sizeof(*medtable), 
		  compar);
	    umat->uvalue[i][j]=medtable[1];
	  } else {
	    medtable[1]=umat->uvalue[i][j-1];
	    medtable[2]=umat->uvalue[i+1][j-1];
	    medtable[3]=umat->uvalue[i][j+1];
	    medtable[4]=umat->uvalue[i+1][j+1];
	    qsort((void *)medtable, 5, sizeof(*medtable), 
		  compar);
	    umat->uvalue[i][j]=medtable[2];
	  }
	} else if(i==umat->uxdim-1 && j>0 && j<umat->uydim-1) {
	  /* in the right edge */
	  medtable[0]=umat->uvalue[i-1][j];
	  if(j%4) {
	    medtable[1]=umat->uvalue[i][j-1];
	    medtable[2]=umat->uvalue[i][j+1];
	    qsort((void *)medtable, 3, sizeof(*medtable), 
		  compar);
	    umat->uvalue[i][j]=medtable[1];
	  } else {
	    medtable[1]=umat->uvalue[i][j-1];
	    medtable[2]=umat->uvalue[i-1][j-1];
	    medtable[3]=umat->uvalue[i][j+1];
	    medtable[4]=umat->uvalue[i-1][j+1];
	    qsort((void *)medtable, 5, sizeof(*medtable), 
		  compar);
	    umat->uvalue[i][j]=medtable[2];
	  }
	} else if(i==0 && j==0)
	  /* the upper left-hand corner */
	  umat->uvalue[i][j]=(umat->uvalue[i+1][j]+umat->uvalue[i][j+1])/2.0;
	else if(i==umat->uxdim-1 && j==0) {
	  /* the upper right-hand corner */
	  medtable[0]=umat->uvalue[i-1][j];
	  medtable[1]=umat->uvalue[i-1][j+1];
	  medtable[2]=umat->uvalue[i][j+1];
	  qsort((void *)medtable, 3, sizeof(*medtable), 
		compar);
	  umat->uvalue[i][j]=medtable[1];
	} else if(i==0 && j==umat->uydim-1) {
	  /* the lower left-hand corner */
	  if(!(j%4))
	    umat->uvalue[i][j]=(umat->uvalue[i+1][j]+umat->uvalue[i][j-1])/2.0;
	  else {
	    medtable[0]=umat->uvalue[i+1][j];
	    medtable[1]=umat->uvalue[i][j-1];
	    medtable[2]=umat->uvalue[i+1][j-1];
	    qsort((void *)medtable, 3, sizeof(*medtable), 
		  compar);
	    umat->uvalue[i][j]=medtable[1];
	  }
	} else if(i==umat->uxdim-1 && j==umat->uydim-1) {
	  /* the lower right-hand corner */
	  if(j%4)
	    umat->uvalue[i][j]=(umat->uvalue[i-1][j]+umat->uvalue[i][j-1])/2.0;
	  else {
	    medtable[0]=umat->uvalue[i-1][j];
	    medtable[1]=umat->uvalue[i][j-1];
	    medtable[2]=umat->uvalue[i-1][j-1];
	    qsort((void *)medtable, 3, sizeof(*medtable), 
		  compar);
	    umat->uvalue[i][j]=medtable[1];
	  }
	}
  
    
  /* Swap the matrix if needed */
  if (xswap) {
    float tmp;
    for (i=0;i<umat->uxdim;i++)
      for (j=0;j<umat->uydim/2;j++) {
	tmp = umat->uvalue[i][j];
	umat->uvalue[i][j] = umat->uvalue[i][umat->uydim-1-j];
	umat->uvalue[i][umat->uydim-1-j] = tmp;
      }
  }
  if (yswap) {
    float *tmp;
    for (i=0;i<umat->uxdim/2;i++) {
      tmp = umat->uvalue[i];
      umat->uvalue[i] = umat->uvalue[umat->uxdim-1-i];
      umat->uvalue[umat->uxdim-1-i] = tmp;
    }
  }

  /* find the minimum and maximum values */
  max = -FLT_MAX;
  min = FLT_MAX;
  /* min=umat->uvalue[0][0]; */
  for (i=0;i<umat->uxdim;i++)
    for (j=0;j<umat->uydim;j++)
      {
	if (umat->uvalue[i][j] > max)
	  max = umat->uvalue[i][j];
	if (umat->uvalue[i][j] < min)
	  min = umat->uvalue[i][j];
      }

  ifverbose(2)
    {
      fprintf(stderr,"minimum distance between elements : %f\n",min);
      fprintf(stderr,"maximum distance between elements : %f\n",max);
    }

  bw = max - min;
  /* scale values to [0,1] */
  for (i=0;i<umat->uxdim;i++)
    for (j=0;j<umat->uydim;j++)
      umat->uvalue[i][j] = 1.0 - (umat->uvalue[i][j] - min) / bw;

#if 0
      umat->uvalue[i][j] = -umat->uvalue[i][j]/max+1.0;
#endif 	 

  return 0;
}

void swap_umat(struct umatrix *umat, int x,int y)
{
  int i,j;
  if (x) {
    float tmp;
    for (i=0;i<umat->uxdim;i++)
      for (j=0;j<umat->uydim/2;j++) {
	tmp = umat->uvalue[i][j];
	umat->uvalue[i][j] = umat->uvalue[i][umat->uydim-1-j];
	umat->uvalue[i][umat->uydim-1-j] = tmp;
      }
  }
  if (y) {
    float *tmp;
    for (i=0;i<umat->uxdim/2;i++) {
      tmp = umat->uvalue[i];
      umat->uvalue[i] = umat->uvalue[umat->uxdim-1-i];
      umat->uvalue[umat->uxdim-1-i] = tmp;
    }
  }
/* Labels too */
#if 0
  if (x) {
    char *tmp;
    for (i=0;i<mat->xdim;i++)
      for (j=0;j<mat->ydim/2;j++) {
	tmp = mat->label[i][j];
	mat->label[i][j] = mat->label[i][mat->ydim-1-j];
	mat->label[i][mat->ydim-1-j] = tmp;
      }
  }
  if (y) {
    char **tmp;
    for (i=0;i<mat->xdim/2;i++) {
      tmp = mat->label[i];
      mat->label[i] = mat->label[mat->xdim-1-i];
      mat->label[mat->xdim-1-i] = tmp;
    }
  }
#endif

}

int average_umatrix(struct umatrix *umat)
{
  int i,j;
  struct umatrix *umat2;
  
  umat2 = alloc_umat();
  umat2->uxdim = umat->uxdim;
  umat2->uydim = umat->uydim;

  umat2->uvalue = (float **)malloc(umat->uxdim*sizeof(float *));
  for (i=0;i<umat->uxdim;i++)
	umat2->uvalue[i] = (float *)malloc(umat->uydim*sizeof(float));

  /* rectangular topology */
  if (umat->topol==TOPOL_RECT)
    {
      for (j=0;j<umat->uydim;j++)
	for (i=0;i<umat->uxdim;i++)
	  if(i&&j&&(j<umat->uydim-1)&&(i<umat->uxdim-1)){
	    /* Non borders */
	    umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				umat->uvalue[i-1][j]+
				umat->uvalue[i][j]+
				umat->uvalue[i+1][j]+
				umat->uvalue[i][j+1])/5.0);
	  } else if(i && (i<umat->uxdim-1) && !j) {
	    /* West brdr*/
	    umat2->uvalue[i][j]=((umat->uvalue[i-1][j]+
				umat->uvalue[i][j]+
				umat->uvalue[i+1][j]+
				umat->uvalue[i][j+1])/4.0);
	  } else if (!i && j&& (j<umat->uydim-1)) {
	    /*north*/
	    umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				umat->uvalue[i][j]+
				umat->uvalue[i+1][j]+
				umat->uvalue[i][j+1])/4.0);
	  } else if ( i && (i < umat->uxdim-1)&& (j==umat->uydim-1)) {
	    /* south */
	    umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				umat->uvalue[i-1][j]+
				umat->uvalue[i][j]+
				umat->uvalue[i+1][j])/4.0);
	  } else if ( j && (j < umat->uydim-1)&& (i == umat->uxdim-1)) {
	    /* east*/
	    umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				umat->uvalue[i-1][j]+
				umat->uvalue[i][j]+
				umat->uvalue[i][j+1])/4.0);
	  }
      /*corners*/
      umat2->uvalue[0][umat->uydim-1]=(umat->uvalue[1][umat->uydim-1]+ umat->uvalue[0][umat->uydim-1]+ umat->uvalue[0][umat->uydim-2])/3.0;
      umat2->uvalue[umat->uxdim-1][umat->uydim-1]=(umat->uvalue[umat->uxdim-2][umat->uydim-1]+ umat->uvalue[umat->uxdim-1][umat->uydim-1]+ umat->uvalue[umat->uxdim-1][umat->uydim-2])/3.0;
      umat2->uvalue[umat->uxdim-1][0]=(umat->uvalue[umat->uxdim-2][0]+ umat->uvalue[umat->uxdim-1][0]+ umat->uvalue[umat->uxdim-1][1])/3.0;
      umat2->uvalue[0][0]=(umat->uvalue[1][0]+ umat->uvalue[0][1]+ umat->uvalue[0][0])/3.0;


    }
  else
    /* hexagonal topology */
    {
      /*else*/
      for (j=1;j<umat->uydim-1;j++)
	for (i=1;i<umat->uxdim-1;i++)
	  /* Non-borders */
	  {
	    if((j%4)==1){
	      umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				   umat->uvalue[i+1][j-1]+umat->uvalue[i-1][j]+
				   umat->uvalue[i][j]+umat->uvalue[i+1][j]+
				   umat->uvalue[i-1][j+1]+
				   umat->uvalue[i][j+1])/((float) 7.0));
	    } else if((j%4)==2) {
	      umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				   umat->uvalue[i+1][j-1]+umat->uvalue[i-1][j]+
				   umat->uvalue[i][j]+umat->uvalue[i+1][j]+
				   umat->uvalue[i][j+1]+
				   umat->uvalue[i+1][j+1])/((float) 7.0));
	      
	    } else if((j%4)==3) {
	      umat2->uvalue[i][j]=((umat->uvalue[i-1][j-1]+
				   umat->uvalue[i][j-1]+umat->uvalue[i-1][j]+
				   umat->uvalue[i][j]+umat->uvalue[i+1][j]+
				   umat->uvalue[i][j+1]+
				   umat->uvalue[i+1][j+1])/((float) 7.0));
	      
	    }  else if((j%4)==0) {
	      umat2->uvalue[i][j]=((umat->uvalue[i-1][j-1]+
				   umat->uvalue[i][j-1]+umat->uvalue[i-1][j]+
				   umat->uvalue[i][j]+umat->uvalue[i+1][j]+
				   umat->uvalue[i-1][j+1]+
				   umat->uvalue[i][j+1])/((float) 7.0));
	      
	    }		  
	  }
      /* north border */
      j=0;
      for (i=1;i<umat->uxdim-1;i++)
        umat2->uvalue[i][j]=((umat->uvalue[i-1][j]+
				   umat->uvalue[i][j]+umat->uvalue[i+1][j]+
				   umat->uvalue[i-1][j+1]+
				   umat->uvalue[i][j+1])/((float) 5.0));
      /*south border*/
      j=umat->uydim-1;
      for (i=1;i<umat->uxdim-1;i++){
	if((j%4)==1){
	  umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
			       umat->uvalue[i+1][j-1]+umat->uvalue[i-1][j]+
			       umat->uvalue[i][j]+
			       umat->uvalue[i+1][j])/((float) 5.0));
	    } else if((j%4)==2) {
	      umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				   umat->uvalue[i+1][j-1]+umat->uvalue[i-1][j]+
				   umat->uvalue[i][j]+
				   umat->uvalue[i+1][j])/((float) 5.0));
	      
	    } else if((j%4)==3) {
	      umat2->uvalue[i][j]=((umat->uvalue[i-1][j-1]+
				   umat->uvalue[i][j-1]+umat->uvalue[i-1][j]+
				   umat->uvalue[i][j]+
				   umat->uvalue[i+1][j])/((float) 5.0));
	      
	    }  else if((j%4)==0) {
	      umat2->uvalue[i][j]=((umat->uvalue[i-1][j-1]+
				   umat->uvalue[i][j-1]+umat->uvalue[i-1][j]+
				   umat->uvalue[i][j]+
				   umat->uvalue[i+1][j] )/((float) 5.0));
	      
	    }		 
      }
      /*east border*/
      i=umat->uxdim-1;
      for (j=1;j<umat->uydim-1;j++)
	{
	  if((j%4)==1){
	  
	    umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				 umat->uvalue[i-1][j]+
				 umat->uvalue[i][j]+
				 umat->uvalue[i-1][j+1]+
				 umat->uvalue[i][j+1])/((float) 5.0));
	  } else if((j%4)==2) {
	    umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				 umat->uvalue[i-1][j]+
				 umat->uvalue[i][j]+
				   umat->uvalue[i][j+1])/((float) 4.0));
	    
	  } else if((j%4)==3) {
	    umat2->uvalue[i][j]=((umat->uvalue[i-1][j-1]+
				   umat->uvalue[i][j-1]+umat->uvalue[i-1][j]+
				 umat->uvalue[i][j]+
				 umat->uvalue[i][j+1])/((float) 5.0));
	    
	    }  else if((j%4)==0) {
	      umat2->uvalue[i][j]=((umat->uvalue[i-1][j-1]+
				   umat->uvalue[i][j-1]+umat->uvalue[i-1][j]+
				   umat->uvalue[i][j]+
				   umat->uvalue[i-1][j+1]+
				   umat->uvalue[i][j+1])/((float) 6.0));
	      
	    }
	}
      i=0;
      for (j=1;j<umat->uydim-1;j++)

	  /*west border*/
	  {
	    if((j%4)==1){
	      umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				   umat->uvalue[i+1][j-1]+
				   umat->uvalue[i][j]+umat->uvalue[i+1][j]+
				   umat->uvalue[i][j+1])/((float) 5.0));
	    } else if((j%4)==2) {
	      umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				   umat->uvalue[i+1][j-1]+
				   umat->uvalue[i][j]+umat->uvalue[i+1][j]+
				   umat->uvalue[i][j+1]+
				   umat->uvalue[i+1][j+1])/((float) 6.0));
	      
	    } else if((j%4)==3) {
	      umat2->uvalue[i][j]=(( umat->uvalue[i][j-1]+
				   umat->uvalue[i][j]+umat->uvalue[i+1][j]+
				   umat->uvalue[i][j+1]+
				   umat->uvalue[i+1][j+1])/((float) 5.0));
	      
	    }  else if((j%4)==0) {
	      umat2->uvalue[i][j]=((umat->uvalue[i][j-1]+
				   umat->uvalue[i][j]+umat->uvalue[i+1][j]+
				   umat->uvalue[i][j+1])/((float) 4.0));
	      
	    }		  
	  }
    

/*Corners*/
      umat2->uvalue[0][0] = (  umat->uvalue[1][0] + umat->uvalue[0][0] + 
			    umat->uvalue[0][1] )/((float)3.0);
      
      umat2->uvalue[(umat->uxdim-1)][0] = (  umat->uvalue[(umat->uxdim-1)][0] +
				     umat->uvalue[(umat->uxdim-1)][1] + 
				     umat->uvalue[(umat->uxdim-2)][0] +
				     umat->uvalue[(umat->uxdim-2)][1] )/
				       ((float) 4.0);

      umat2->uvalue[(umat->uxdim-1)][(umat->uydim-1)] = (  /* Short cut */
		    umat->uvalue[(umat->uxdim-1)][(umat->uydim-1)] +
		    umat->uvalue[(umat->uxdim-1)][(umat->uydim-2)] + 
		    umat->uvalue[(umat->uxdim-2)][(umat->uydim-1)])/
				       ((float) 3.0);

      umat2->uvalue[0][(umat->uydim-1)] = (  umat->uvalue[0][(umat->uydim-1)] +
				     umat->uvalue[1][(umat->uydim-1)]  + 
				     umat->uvalue[0][(umat->uydim-2)] )/
				       ((float) 3.0); /*Short cut */
    }

  for (j=0;j<umat->uydim;j++)
    for (i=0;i<umat->uxdim;i++){
	umat->uvalue[i][j]=umat2->uvalue[i][j];
    }    

  free_umat(umat2);

  /* find the minimum and maximum values */
  
/*  min=umat->uvalue[0][0];
  for (i=0;i<umat->uxdim;i++)
    for (j=0;j<umat->uydim;j++)
      {
	if (umat->uvalue[i][j] > max)
	  max = umat->uvalue[i][j];
	if (umat->uvalue[i][j] < min)
	  min = umat->uvalue[i][j];
      }
  
  fprintf(stderr,"minimum distance between elements : %f\n",min);
  fprintf(stderr,"maximum distance between elements : %f\n",max);
  
  for (i=0;i<umat->uxdim;i++)
    for (j=0;j<umat->uydim;j++)
      umat->uvalue[i][j] = umat->uvalue[i][j]/max;
  */ 
  return 0;

}

int median_umatrix(struct umatrix *umat)
{
  int i,j;
  struct umatrix *umat2;
  
  umat2 = alloc_umat();
  umat2->uxdim = umat->uxdim;
  umat2->uydim = umat->uydim;

  umat2->uvalue = (float **)malloc(umat->uxdim*sizeof(float *));
  for (i=0;i<umat->uxdim;i++)
	umat2->uvalue[i] = (float *)malloc(umat->uydim*sizeof(float));
  /* rectangular topology */
  if (umat->topol==TOPOL_RECT)
    {
      for (j=0;j<umat->uydim;j++)
	for (i=0;i<umat->uxdim;i++)
	  if(i&&j&&(j<umat->uydim-1)&&(i<umat->uxdim-1)){
	    umat2->uvalue[i][j]=(median5(umat->uvalue[i][j-1],
				umat->uvalue[i-1][j],
				umat->uvalue[i][j],
				umat->uvalue[i+1][j],
				umat->uvalue[i][j+1]));
	  } else if(i && (i<umat->uxdim-1) && !j) {
	    umat2->uvalue[i][j]=(median4(umat->uvalue[i-1][j],
				umat->uvalue[i][j],
				umat->uvalue[i+1][j],
				umat->uvalue[i][j+1]));
	  } else if (!i && j&& (j<umat->uydim-1)) {
	    umat2->uvalue[i][j]=(median4(umat->uvalue[i][j-1],
				umat->uvalue[i][j],
				umat->uvalue[i+1][j],
				umat->uvalue[i][j+1]));
	  } else if ( i && (i < umat->uxdim-1)&& (j==umat->uydim-1)) {
	    umat2->uvalue[i][j]=(median4(umat->uvalue[i][j-1],
				umat->uvalue[i-1][j],
				umat->uvalue[i][j],
				umat->uvalue[i+1][j]));
	  } else if ( j && (j < umat->uydim-1)&& (i==umat->uxdim-1)) {
	    umat2->uvalue[i][j]=(median5(umat->uvalue[i][j-1],
				umat->uvalue[i-1][j],
				umat->uvalue[i-1][j],
				umat->uvalue[i][j],
				umat->uvalue[i][j+1]));
	  }
      umat2->uvalue[0][umat->uydim-1]=median3(umat->uvalue[1][umat->uydim-1], umat->uvalue[0][umat->uydim-1], umat->uvalue[0][umat->uydim-2]);
      umat2->uvalue[umat->uxdim-1][umat->uydim-1]=median3(umat->uvalue[umat->uxdim-2][umat->uydim-1], umat->uvalue[umat->uxdim-1][umat->uydim-1], umat->uvalue[umat->uxdim-1][umat->uydim-2]);
      umat2->uvalue[umat->uxdim-1][0]=median3(umat->uvalue[umat->uxdim-2][0], umat->uvalue[umat->uxdim-1][0], umat->uvalue[umat->uxdim-1][1]);
      umat2->uvalue[0][0]=median3(umat->uvalue[1][0], umat->uvalue[0][1], umat->uvalue[0][0]);


    }
  else
    /* hexagonal topology */
    {
      /*else*/
      for (j=1;j<umat->uydim-1;j++)
	for (i=1;i<umat->uxdim-1;i++)
	  /* Non-borders */
	  {
	    if((j%4)==1){
	      umat2->uvalue[i][j]=(median7(umat->uvalue[i][j-1],
				   umat->uvalue[i+1][j-1],umat->uvalue[i-1][j],
				   umat->uvalue[i][j],umat->uvalue[i+1][j],
				   umat->uvalue[i-1][j+1],
				   umat->uvalue[i][j+1]));
	    } else if((j%4)==2) {
	      umat2->uvalue[i][j]=(median7(umat->uvalue[i][j-1],
				   umat->uvalue[i+1][j-1],umat->uvalue[i-1][j],
				   umat->uvalue[i][j],umat->uvalue[i+1][j],
				   umat->uvalue[i][j+1],
				   umat->uvalue[i+1][j+1]));
	      
	    } else if((j%4)==3) {
	      umat2->uvalue[i][j]=(median7(umat->uvalue[i-1][j-1],
				   umat->uvalue[i][j-1],umat->uvalue[i-1][j],
				   umat->uvalue[i][j],umat->uvalue[i+1][j],
				   umat->uvalue[i][j+1],
				   umat->uvalue[i+1][j+1]));
	      
	    }  else if((j%4)==0) {
	      umat2->uvalue[i][j]=(median7(umat->uvalue[i-1][j-1],
				   umat->uvalue[i][j-1],umat->uvalue[i-1][j],
				   umat->uvalue[i][j],umat->uvalue[i+1][j],
				   umat->uvalue[i-1][j+1],
				   umat->uvalue[i][j+1]));
	      
	    }		  
	  }
      /* north border */
      j=0;
      for (i=1;i<umat->uxdim-1;i++)
        umat2->uvalue[i][j]=(median5(umat->uvalue[i-1][j],
				   umat->uvalue[i][j],umat->uvalue[i+1][j],
				   umat->uvalue[i-1][j+1],
				   umat->uvalue[i][j+1]));
      /*south border*/
      j=umat->uydim-1;
      for (i=1;i<umat->uxdim-1;i++){
	if((j%4)==1){
	  umat2->uvalue[i][j]=(median5(umat->uvalue[i][j-1],
			       umat->uvalue[i+1][j-1],umat->uvalue[i-1][j],
			       umat->uvalue[i][j],
			       umat->uvalue[i+1][j]));
	    } else if((j%4)==2) {
	      umat2->uvalue[i][j]=(median5(umat->uvalue[i][j-1],
				   umat->uvalue[i+1][j-1],umat->uvalue[i-1][j],
				   umat->uvalue[i][j],
				   umat->uvalue[i+1][j]));
	      
	    } else if((j%4)==3) {
	      umat2->uvalue[i][j]=(median5(umat->uvalue[i-1][j-1],
				   umat->uvalue[i][j-1],umat->uvalue[i-1][j],
				   umat->uvalue[i][j],
				   umat->uvalue[i+1][j]));
	      
	    }  else if((j%4)==0) {
	      umat2->uvalue[i][j]=(median5(umat->uvalue[i-1][j-1],
				   umat->uvalue[i][j-1],umat->uvalue[i-1][j],
				   umat->uvalue[i][j],
				   umat->uvalue[i+1][j] ));
	      
	    }		 
      }
      /*east border*/
      i=umat->uxdim-1;
      for (j=1;j<umat->uydim-1;j++)
	{
	  if((j%4)==1){
	  
	    umat2->uvalue[i][j]=(median5(umat->uvalue[i][j-1],
				 umat->uvalue[i-1][j],
				 umat->uvalue[i][j],
				 umat->uvalue[i-1][j+1],
				 umat->uvalue[i][j+1]));
	  } else if((j%4)==2) {
	    umat2->uvalue[i][j]=(median4(umat->uvalue[i][j-1],
				 umat->uvalue[i-1][j],
				 umat->uvalue[i][j],
				   umat->uvalue[i][j+1]));
	    
	  } else if((j%4)==3) {
	    umat2->uvalue[i][j]=(median5(umat->uvalue[i-1][j-1],
				   umat->uvalue[i][j-1],umat->uvalue[i-1][j],
				 umat->uvalue[i][j],
				 umat->uvalue[i][j+1]));
	    
	    }  else if((j%4)==0) {
	      umat2->uvalue[i][j]=(median6(umat->uvalue[i-1][j-1],
				   umat->uvalue[i][j-1],umat->uvalue[i-1][j],
				   umat->uvalue[i][j],
				   umat->uvalue[i-1][j+1],
				   umat->uvalue[i][j+1]));
	      
	    }
	}
      i=0;
      for (j=1;j<umat->uydim-1;j++)

	  /*west border*/
	  {
	    if((j%4)==1){
	      umat2->uvalue[i][j]=(median5(umat->uvalue[i][j-1],
				   umat->uvalue[i+1][j-1],
				   umat->uvalue[i][j],umat->uvalue[i+1][j],
				   umat->uvalue[i][j+1]));
	    } else if((j%4)==2) {
	      umat2->uvalue[i][j]=(median6(umat->uvalue[i][j-1],
				   umat->uvalue[i+1][j-1],
				   umat->uvalue[i][j],umat->uvalue[i+1][j],
				   umat->uvalue[i][j+1],
				   umat->uvalue[i+1][j+1]));
	      
	    } else if((j%4)==3) {
	      umat2->uvalue[i][j]=(median5( umat->uvalue[i][j-1],
				   umat->uvalue[i][j],umat->uvalue[i+1][j],
				   umat->uvalue[i][j+1],
				   umat->uvalue[i+1][j+1]));
	      
	    }  else if((j%4)==0) {
	      umat2->uvalue[i][j]=(median4(umat->uvalue[i][j-1],
				   umat->uvalue[i][j],umat->uvalue[i+1][j],
				   umat->uvalue[i][j+1]));
	      
	    }		  
	  }
      

/*Corners*/
      umat2->uvalue[0][0] = median3(  umat->uvalue[1][0] , umat->uvalue[0][0] , 
				   umat->uvalue[0][1] );
      
      umat2->uvalue[(umat->uxdim-1)][0] = median4(  umat->uvalue[(umat->uxdim-1)][0] ,
				     umat->uvalue[(umat->uxdim-1)][1] , 
				     umat->uvalue[(umat->uxdim-2)][0] ,
				     umat->uvalue[(umat->uxdim-2)][1] );

      umat2->uvalue[(umat->uxdim-1)][(umat->uydim-1)] = median3(  /* Short cut */
		    umat->uvalue[(umat->uxdim-1)][(umat->uydim-1)] ,
		    umat->uvalue[(umat->uxdim-1)][(umat->uydim-2)] , 
		    umat->uvalue[(umat->uxdim-2)][(umat->uydim-1)]);


      umat2->uvalue[0][(umat->uydim-1)] = median3(  umat->uvalue[0][(umat->uydim-1)] ,
				     umat->uvalue[1][(umat->uydim-1)]  , 
				     umat->uvalue[0][(umat->uydim-2)] );
      /*Short cut */
    }

  for (j=0;j<umat->uydim;j++)
    for (i=0;i<umat->uxdim;i++){
      umat->uvalue[i][j]=umat2->uvalue[i][j];
    }    

  free_umat(umat2);

  /* find the minimum and maximum values */
#if 0  
  min=max=umat->uvalue[0][0];
  for (i=0;i<umat->uxdim;i++)
    for (j=0;j<umat->uydim;j++)
      {
	if (umat->uvalue[i][j] > max)
	  max = umat->uvalue[i][j];
	if (umat->uvalue[i][j] < min)
	  min = umat->uvalue[i][j];
      }
  
      
  ifverbose(2)
    {
      fprintf(stderr,"minimum distance between elements : %f\n",min);
      fprintf(stderr,"maximum distance between elements : %f\n",max);
    }
  for (i=0;i<umat->uxdim;i++)
    for (j=0;j<umat->uydim;j++)
      umat->uvalue[i][j] = umat->uvalue[i][j]/max;
#endif  
  return 0;

}

#if 0

/* calculate the input data distribution */
void datamatrix(char *data_file,Map map,Mat mat,char *outputfile,int xswap,
		int yswap)
{
  FILE *in,*out;
  int i,j,k,datadim,xmax,ymax;
  float *data;
  double temp,dist,mindist;
  int lask,row,skip,maxline;
  char *s,*tok;

  /* open the input data file */
  if (!(in=fopen(data_file,"r")))
    {
      fprintf(stderr,"Can't open datafile %s\n",data_file);
      exit(-1);
    }

  /* read the dimension of the input data */
  fscanf(in,"%i",&datadim);

  if (datadim != umat->dim)
    {
      fprintf(stderr,"Dimension of the input data vector must be same\n");
      fprintf(stderr,"than dimension of the weight vector of the map\n");
      exit(-1);
    }

  /* define the maximum lenght of the row in the input data file */
  maxline = 20*datadim+64;
  s = (char *)malloc(maxline*sizeof(char));
  data = (float *)malloc(datadim*sizeof(float));

  lask = -1;
  row = 1;
  getline(in,s,maxline);
  do
    {
      ++lask;
      skip = 0;
      do
	{
	  ++row;
	  getline(in,s,maxline);
	  /* if the row in the input file starts 
	     with # or with newline, skip it     */
	  if (strcmp(s,"")==NULL || s[0]=='#')
	    {
	      fprintf(stderr,"Skipping row %i in %s\n",row,data_file);
	      skip = 1;
	    }
	  else
	    skip = 0;
	}while (skip==1);
      if (!feof(in))
	{
	  tok = strtok(s," ");
	  for (i=0;i<datadim;i++)
	    {
	      if (tok==NULL || sscanf(tok,"%f",&(data[i])) <= 0)
		{
		  fprintf(stderr,"Can't read entry on line %i in %s\n",row,data_file);
		  exit(-1);
		}
	      tok = strtok(NULL," ");
	    }
	  mindist = 10000;

	  /* find the closest map element */
	  for (i=0;i<umat->mxdim;i++)
	    for (j=0;j<umat->mydim;j++)
	      {
		dist=0;
		for (k=0;k<datadim;k++)
		  {
		    temp = umat->mvalue[i][j][k] - data[k];
		    dist += temp*temp;
		    if (dist >= mindist)
		      break;
		  }
		if (dist < mindist)
		  {
		    xmax = i;
		    ymax = j;
		    mindist = dist;
		  }
	      }
	  ++mat->value[xmax][ymax];
	}
    } while (!feof(in));
  fprintf(stderr,"%i data vectors read from %s\n",lask,data_file);

  /* Swap the matrix if needed */
  if (xswap) {
    float tmp;
    for (i=0;i<mat->xdim;i++)
      for (j=0;j<mat->ydim/2;j++) {
	tmp = mat->value[i][j];
	mat->value[i][j] = mat->value[i][mat->ydim-1-j];
	mat->value[i][mat->ydim-1-j] = tmp;
      }
  }
  if (yswap) {
    float *tmp;
    for (i=0;i<mat->xdim/2;i++) {
      tmp = mat->value[i];
      mat->value[i] = mat->value[mat->xdim-1-i];
      mat->value[mat->xdim-1-i] = tmp;
    }
  }

  /* find the largest matrix value */
  xmax=0;
  for (i=0;i<umat->mxdim;i++)
    for (j=0;j<umat->mydim;j++)
      {
	if (mat->value[i][j] > xmax)
	  xmax = mat->value[i][j];
      }

  /* write the input data distribution to a file if requested */
  if (outputfile!=NULL)
    {  
      if ((out=fopen(outputfile,"w")) == NULL)
	{
	  fprintf(stderr,"can't open file %s for input data distribution\n",
		  outputfile);
	  exit(-1);
	}

      for (j=0;j<mat->ydim;j++)  
	{
	  for (i=0;i<mat->xdim;i++)
	    fprintf(out,"%i ",(int) mat->value[i][j]);
	  fprintf(out,"\n");
	}
      fprintf(stderr,"input data distribution matrix written in %s\n",
	      outputfile);
    }

  /* scale the matrix to [0,1] */
  for (i=0;i<umat->mxdim;i++)
    for (j=0;j<umat->mydim;j++)
      mat->value[i][j] = mat->value[i][j]/xmax;

}

#endif






