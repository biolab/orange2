/************************************************************************
 *                                                                      *
 *  Program package 'som_pak':                                          *
 *                                                                      *
 *  som_rout.c                                                          *
 *  - routines needed in some programs in som_pak                       *
 *                                                                      *
 *  Version 3.0                                                         *
 *  Date: 1 Mar 1995                                                    *
 *                                                                      *
 *  NOTE: This program package is copyrighted in the sense that it      *
 *  may be used for scientific purposes. The package as a whole, or     *
 *  parts thereof, cannot be included or used in any commercial         *
 *  application without written permission granted by its producents.   *
 *  No programs contained in this package may be copied for commercial  *
 *  distribution.                                                       *
 *                                                                      *
 *  All comments  concerning this program package may be sent to the    *
 *  e-mail address 'lvq@cochlea.hut.fi'.                                *
 *                                                                      *
 ************************************************************************/

#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "lvq_pak.h"
#include "som_rout.h"
#include "datafile.h"

/*---------------------------------------------------------------------*/

struct entries *randinit_codes(struct entries *data, int topol, int neigh, 
			       int xdim, int ydim)
{
  long noc, i;
  int dim;
  struct entries *codes;
  struct data_entry *entr;
  struct data_entry *maval, *mival;
  eptr p;
  long *compcnt = NULL;

  noc = xdim * ydim;

  if ((codes = alloc_entries()) == NULL)
    {
      fprintf(stderr, "randinit_codes: can't allocate memory for codes\n");
      return NULL;
    }


  dim = data->dimension;
  codes->dimension = dim;
  codes->flags.loadmode = LOADMODE_ALL;
  codes->xdim = xdim;
  codes->ydim = ydim;
  codes->topol = topol;
  codes->neigh = neigh;

  /* allocate codebook entries */
  if ((entr = alloc_entry(codes)) == NULL)
    {
      fprintf(stderr, "randinit_codes: can't allocate memory for codebook vector\n");
      close_entries(codes);
      return NULL;
    }
  codes->dentries = entr;

  for (i = 1; i < noc; i++) 
    {
      entr->next = alloc_entry(codes);
      entr = entr->next;
      if (entr == NULL)
	break;
    }

  if (entr == NULL)
    {
      fprintf(stderr, "randinit_codes: can't allocate codebook\n");
      close_entries(codes);
      return NULL;
    }

  codes->num_entries = noc;
  
  /* Find the maxim and minim values of data */

  if ((compcnt = malloc(sizeof(long) * dim)) == NULL)
    {
      fprintf(stderr, "randinit_codes: can't allocate memory\n");
      close_entries(codes);
      return NULL;
    }

  for (i = 0; i < dim; i++)
    compcnt[i] = 0;

  maval = alloc_entry(data);
  mival = alloc_entry(data);
  if (!(maval && mival))
    {
      close_entries(codes);
      codes = NULL;
      goto end;
    }
      
  for (i = 0; i < data->dimension; i++) {
    maval->points[i] = FLT_MIN;
    mival->points[i] = FLT_MAX;
  }

  if ((entr = rewind_entries(data, &p)) == NULL)
    {
      fprintf(stderr, "randinit_codes: can't get data\n");
      close_entries(codes);
      codes = NULL;
      goto end;
    }

  while (entr != NULL) 
    {
      for (i = 0; i < dim; i++) 
	if (!((entr->mask != NULL) && (entr->mask[i] != 0)))
	  {
	    compcnt[i]++;
	    if (maval->points[i] < entr->points[i])
	      maval->points[i] = entr->points[i];
	    if (mival->points[i] > entr->points[i])
	      mival->points[i] = entr->points[i];
	  }
      entr = next_entry(&p);
    }

  for (i = 0; i < dim; i++)
    if (compcnt[i] == 0)
      fprintf(stderr, "randinit_codes: warning! component %d has no data, using 0.0\n", (int)(i + 1));
  
  /* Randomize the vector values */

  entr = rewind_entries(codes, &p);
  while (entr != NULL) {
    for (i = 0; i < dim; i++) {
      if (compcnt[i] > 0)
	entr->points[i] = mival->points[i] +
          (maval->points[i] - mival->points[i]) * ((float) orand() / 32768.0);
      else
	entr->points[i] = 0.0;
    }
    clear_entry_labels(entr);
    entr = next_entry(&p);
  }

 end:
  if (compcnt)
    free(compcnt);
  free_entry(mival);
  free_entry(maval);

  return(codes);
}


/*---------------------------------------------------------------------*/

void normalize(float *v, int n)
{
  float sum=0.0;
  int j;

  for (j=0; j<n; j++) sum+=v[j]*v[j];
  sum=sqrt(sum);
  for (j=0; j<n; j++) v[j]/=sum;
}


float dotprod(float *v, float *w, int n)
{
  float sum=0.0;
  int j;

  for (j=0; j<n; j++) sum+=v[j]*w[j];
  return (sum);
}


int gram_schmidt(float *v, int n, int e)
{
  int i, j, p, t;
  float sum, *w=(float*)malloc(n*e*sizeof(float));

  if (w == NULL)
    return 1;

  for (i=0; i<e; i++) {
    for (t=0; t<n; t++) {
      sum=v[i*n+t];
      for (j=0; j<i; j++)
	for (p=0; p<n; p++)
	  sum-=w[j*n+t]*w[j*n+p]*v[i*n+p];
      w[i*n+t]=sum;
    }
    normalize(w+i*n, n);
  }
  memcpy(v, w, n*e*sizeof(float));
  free(w);
  return 0;
}

struct data_entry *find_eigenvectors(struct entries *data)
{
  int n=data->dimension;
  float *r=(float*)malloc(n*n*sizeof(float));
  float *m=(float*)malloc(n*sizeof(float));
  float *u=(float*)malloc(2*n*sizeof(float));
  float *v=(float*)malloc(2*n*sizeof(float));
  float mu[2];
  struct data_entry *ptr, *tmp;
  float sum;
  int i, j;
  long *k2, k;
  eptr p;
  char *mask;

  if (r==NULL || m==NULL || u==NULL || v==NULL ) goto everror;

  for (i=0; i<n*n; i++) 
    r[i]=0.0;
  for (i=0; i<n; i++) 
    m[i]=0.0;

  k2 = malloc(n *sizeof(long));
  if (k2 == NULL)
    goto everror;
  memset(k2, 0, n * sizeof(long));

  if ((ptr=rewind_entries(data, &p)) == NULL)
    {
      fprintf(stderr, "find_eigenvectors: can't get data\n");
      goto everror;
    }

  for (k=0; ptr != NULL; k++, ptr=next_entry(&p))
    {
      mask = ptr->mask;
      for (i=0; i<n; i++)
	if ((!mask) || (mask && (mask[i] == 0)))
	  {
	    m[i]+=ptr->points[i]; /* masked components have the value 0 so they
				     don't affect the sum */
	    k2[i]++;
	  }
    }

  if (k<3) goto everror;

  for (i=0; i<n; i++)
    m[i]/=k2[i];

  free(k2); k2 = NULL;

  if ((ptr=rewind_entries(data, &p)) == NULL)
    {
      fprintf(stderr, "find_eigenvectors: can't get data\n");
      goto everror;
    }

  for (; ptr != NULL; ptr=next_entry(&p))
    {
      mask = ptr->mask;
      for (i=0; i<n; i++)
	{
	  /* the components that are masked off are ignored */
	  if (mask && (mask[i] != 0))
	    continue;
	  for (j=i; j<n; j++)
	    if (mask && (mask[j] != 0))
	      continue;
	    else
	      r[i*n+j]+=(ptr->points[i]-m[i])*(ptr->points[j]-m[j]);
	}
    }

  for (i=0; i<n; i++)
    for (j=i; j<n; j++)
      r[j*n+i]=r[i*n+j]/=k;

  for (i=0; i<2; i++) {
    for (j=0; j<n; j++) u[i*n+j]=orand()/16384.0-1.0;
    normalize(u+i*n, n);
    mu[i]=1.0;
  }

  for (k=0; k<10; k++) {
    for (i=0; i<2; i++)
      for (j=0; j<n; j++)
	v[i*n+j]=mu[i]*dotprod(r+j*n, u+i*n, n)+u[i*n+j];

    gram_schmidt(v, n, 2);

    sum=0.0;
    for (i=0; i<2; i++) {
      for (j=0; j<n; j++)
	sum+=fabs(v[i*n+j]/dotprod(r+j*n, v+i*n, n));

      mu[i]=sum/n;
    }

    memcpy(u, v, 2*n*sizeof(float));
  }

  if (mu[0]==0.0 || mu[1]==0.0) goto everror;
  
  ptr = tmp = alloc_entry(data);
  memcpy(tmp->points, m, n*sizeof(float));

  for (i=0; i<2; i++) {
    tmp->next=alloc_entry(data);
    tmp=tmp->next;
    if (tmp == NULL)
      {
	fprintf(stderr, "find_eigenvectors: can't allocate vector\n");
	goto everror;
      }
    memcpy(tmp->points, u+n*i, n*sizeof(float));
    for (j=0; j<n; j++)
      tmp->points[j]/=sqrt(mu[i]);
  }
  tmp->next = NULL;

  ofree(v);
  ofree(u);
  ofree(m);
  ofree(r);
  return (ptr);

 everror:
  if (v!=NULL) ofree(v);
  if (u!=NULL) ofree(u);
  if (m!=NULL) ofree(m);
  if (r!=NULL) ofree(r);
  return (NULL);

}

struct entries *lininit_codes(struct entries *data, int topol, int neigh, 
			      int xdim, int ydim)
{
  long i, number_of_codes;
  int index, dim;
  float xf, yf;
  struct data_entry *mean, *eigen1, *eigen2, *entr;
  struct entries *codes;
  eptr p;

  number_of_codes = xdim * ydim;

  if ((codes = alloc_entries()) == NULL)
    {
      fprintf(stderr, "Can't allocate memory for codes\n");
      return NULL;
    }

  dim = data->dimension;
  codes->dimension = dim;
  codes->flags.loadmode = LOADMODE_ALL;
  codes->xdim = xdim;
  codes->ydim = ydim;
  codes->topol = topol;
  codes->neigh = neigh;
  codes->num_entries = number_of_codes;

  /* Find the middle point and two eigenvectors of the data */
  mean = find_eigenvectors(data);
  if (mean == NULL) {
    fprintf(stderr, "lininit_codes: Can't find eigenvectors\n");
    close_entries(codes);
    return NULL;
  }
  eigen1 = mean->next;
  eigen2 = eigen1->next;

  /* allocate codebook entries */
  if ((entr = alloc_entry(codes)) == NULL)
    {
      fprintf(stderr, "lininit_codes: can't allocate memory for codebook vector\n");
      close_entries(codes);
      return NULL;
    }
  codes->dentries = entr;

  for (i = 1; i < number_of_codes; i++) 
    {
      entr->next = alloc_entry(codes);
      entr = entr->next;
      if (entr == NULL)
	break;
    }

  if (entr == NULL)
    {
      fprintf(stderr, "lininit_codes: can't allocate codebook\n");
      free_entrys(mean);
      close_entries(codes);
      return NULL;
    }

  /* Initialize the units */
  entr = rewind_entries(codes, &p);
  index = 0;
  while (entr != NULL) {
    xf = 4.0 * (float) (index % xdim) / (xdim - 1.0) - 2.0;
    yf = 4.0 * (float) (index / xdim) / (ydim - 1.0) - 2.0;

    for (i = 0; i < dim; i++) {
      entr->points[i] = mean->points[i] 
                + xf * eigen1->points[i] + yf * eigen2->points[i];
    }
    clear_entry_labels(entr);

    entr = next_entry(&p);
    index++;
  }

  free_entrys(mean);

  return(codes);
}


/*---------------------------------------------------------------------*/

float hexa_dist(int bx, int by, int tx, int ty)
{
  float ret, diff;

  diff = bx - tx;

  if (((by - ty) % 2) != 0) {
    if ((by % 2) == 0) {
      diff -= 0.5;
    }
    else {
      diff += 0.5;
    }
  }
  
  ret = diff * diff;
  diff = by - ty;
  ret += 0.75 * diff * diff;
  ret = (float) sqrt((double) ret);

  return(ret);
}

float rect_dist(int bx, int by, int tx, int ty)
{
  float ret, diff;

  diff = bx - tx;
  ret = diff * diff;
  diff = by - ty;
  ret += diff * diff;
  ret = (float) sqrt((double) ret);

  return(ret);
}

/* Adaptation function for bubble-neighborhood */

void bubble_adapt(struct teach_params *teach, struct data_entry *sample,
		  int bx, int by, float radius, float alpha)
{
  long index;
  int tx, ty, xdim, ydim;
  struct entries *codes = teach->codes;
  MAPDIST_FUNCTION *dist = teach->mapdist;
  struct data_entry *codetmp;
  VECTOR_ADAPT *adapt = teach->vector_adapt;
  eptr p;

  xdim = codes->xdim;
  ydim = codes->ydim;
  
  ifverbose(10)
    fprintf(stderr, "Best match in %d, %d\n", bx, by);

  codetmp = rewind_entries(codes, &p);
  index = 0;
  
  while (codetmp != NULL)
    {
      tx = index % xdim;
      ty = index / xdim;
      
      if (dist(bx, by, tx, ty) <= radius) {
	ifverbose(11)
	  fprintf(stderr, "Adapt unit %d, %d\n", tx, ty);
	
	adapt(codetmp, sample, codes->dimension, alpha);

      }
      codetmp = next_entry(&p);
      index++;
    }
}


/* Adaptation function for gaussian neighbourhood */
     
void gaussian_adapt(struct teach_params *teach, struct data_entry *sample,
		    int bx, int by, float radius, float alpha)
{
  long index;
  int tx, ty, xdim, ydim;
  float dd;
  float alp;
  struct entries *codes = teach->codes;
  MAPDIST_FUNCTION *dist = teach->mapdist;
  VECTOR_ADAPT *adapt = teach->vector_adapt;
  struct data_entry *codetmp;
  eptr p;

  xdim = codes->xdim;
  ydim = codes->ydim;

  ifverbose(10)
    fprintf(stderr, "Best match in %d, %d\n", bx, by);

  codetmp = rewind_entries(codes, &p);
  index = 0;

  while (codetmp != NULL)
    {
      tx = index % xdim;
      ty = index / xdim;

      ifverbose(11)
	fprintf(stderr, "Adapt unit %d, %d\n", tx, ty);
      dd = dist(bx, by, tx, ty);

      alp = alpha *
	(float) exp((double) (-dd * dd / (2.0 * radius * radius)));
      
      adapt(codetmp, sample, codes->dimension, alp);

      codetmp = next_entry(&p);
      index++;
    }
}


/* som_training - train a SOM. Radius of the neighborhood decreases 
   linearly from the initial value to one and the learning parameter 
   decreases linearly from its initial value to zero. */

struct entries *som_training(struct teach_params *teach)
{

  NEIGH_ADAPT *adapt;
  WINNER_FUNCTION *find_winner = teach->winner;
  ALPHA_FUNC *get_alpha = teach->alpha_func;
  int dim;
  int bxind, byind;
  float weight;
  float trad, talp;
  struct data_entry *sample;
  struct entries *data = teach->data;
  struct entries *codes = teach->codes;
  long le, length = teach->length;
  float alpha = teach->alpha;
  float radius = teach->radius;
  struct snapshot_info *snap = teach->snapshot;
  struct winner_info win_info;
  eptr p;

  if (set_som_params(teach))
    {
      fprintf(stderr, "som_training: can't set SOM parameters\n");
      return NULL;
    }

  adapt = teach->neigh_adapt;

  if ((sample = rewind_entries(data, &p)) == NULL)
    {
      fprintf(stderr, "som_training: can't get data\n");
      return NULL;
    }

  dim = codes->dimension;
  if (data->dimension != dim)
    {
      fprintf(stderr, "code dimension (%d) != data dimension (%d)\n",
	      dim, data->dimension);
      return NULL;
    }

  time(&teach->start_time);

  for (le = 0; le < length; le++, sample = next_entry(&p)) {
    /* if we are at the end of data file, go back to the start */
    if (sample == NULL)
      {
	sample = rewind_entries(data, &p);
	if (sample == NULL)
	  {
	    fprintf(stderr, "som_training: couldn't rewind data (%ld/%ld iterations done)\n", le, length);
	    return NULL;
	  }
      }

    weight = sample->weight;

    /* Radius decreases linearly to one */
    trad = 1.0 + (radius - 1.0) * (float) (length - le) / (float) length;

    talp = get_alpha(le, length, alpha);

    /* If the sample is weighted, we
       modify the training rate so that we achieve the same effect as
       repeating the sample 'weight' times */
    if ((weight > 0.0) && (use_weights(-1))) {
      talp = 1.0 - (float) pow((double) (1.0 - talp), (double) weight);
    }

    /* Find the best match */
    /* If fixed point and is allowed then use that value */
    if ((sample->fixed != NULL) && (use_fixed(-1))) {
      /* Get the values from fixed-structure */
      bxind = sample->fixed->xfix;
      byind = sample->fixed->yfix;
    }
    else {

      if (find_winner(codes, sample, &win_info, 1) == 0)
	{
	  ifverbose(3)
	    fprintf(stderr, "ignoring empty sample %d\n", le);
	  goto skip_teach; /* ignore empty samples */
	}
      bxind = win_info.index % codes->xdim;
      byind = win_info.index / codes->xdim;
    }

    /* Adapt the units */
    adapt(teach, sample, bxind, byind, trad, talp);

  skip_teach:
    /* save snapshot when needed */
    if ((snap) && ((le % snap->interval) == 0) && (le > 0))
      {
	ifverbose(2)
	  fprintf(stderr, "Saving snapshot, %ld iterations\n", le);
	if (save_snapshot(teach, le))
	  {
	    fprintf(stderr, "snapshot failed, continuing teaching\n");
	  }
      }

    ifverbose(1)
      mprint((long) (length-le));
  }
  time(&teach->end_time);

  ifverbose(1)
    {
      mprint((long) 0);
      fprintf(stderr, "\n");
    }
  return(codes);
}


/*---------------------------------------------------------------------*/

/* find_qerror - calculate quantization error. */

float find_qerror(struct teach_params *teach)
{
  float qerror;
  struct entries *data = teach->data;
  struct entries *codes = teach->codes;
  WINNER_FUNCTION *find_winner = teach->winner;
  struct data_entry *dtmp;
  struct winner_info win_info;
  eptr p;
  int length_known;
  long nod;

  if (set_som_params(teach))
    {
      fprintf(stderr, "find_qerror: can't set SOM parameters\n");
      return -1;
    }

  /* Scan all data entries */

  qerror = 0.0;
  if ((dtmp = rewind_entries(data, &p)) == NULL)
    {
      fprintf(stderr, "find_qerror: can't get data\n");
      return -1.0;
    }

  if ((length_known = data->flags.totlen_known))
    nod = data->num_entries;
  else
    nod = 0;

  for (; dtmp != NULL; dtmp = next_entry(&p)) 
    {
      if (find_winner(codes, dtmp, &win_info, 1) == 0)
	continue; /* ignore empty vectors */
      
      qerror += sqrt((double) win_info.diff);
      
      if (length_known)
	ifverbose(1)
	  mprint((long) nod--);

    }

  if (length_known)
    ifverbose(1)
      {
	mprint((long) 0);
	fprintf(stderr, "\n");
      }
  
  return(qerror);
}


float bubble_qerror(struct teach_params *teach, struct data_entry *sample,
		   int bx, int by, float radius)
{
  long index;
  int tx, ty, xdim, ydim;
  struct entries *codes = teach->codes;
  MAPDIST_FUNCTION *mdist = teach->mapdist;
  DIST_FUNCTION *distance = teach->dist;
  struct data_entry *codetmp;
  eptr p;
  float d, qerror;

  xdim = codes->xdim;
  ydim = codes->ydim;
  
  ifverbose(10)
    fprintf(stderr, "Best match in %d, %d\n", bx, by);

  codetmp = rewind_entries(codes, &p);
  index = 0;
  qerror = 0;

  while (codetmp != NULL)
    {
      tx = index % xdim;
      ty = index / xdim;
      
      if (mdist(bx, by, tx, ty) <= radius) {
	ifverbose(11)
	  fprintf(stderr, "Adapt unit %d, %d\n", tx, ty);
	
	d = distance(codetmp, sample, codes->dimension);
	/* assume that alpha = 1.0 */
	qerror += d*d;

      }
      codetmp = next_entry(&p);
      index++;
    }
  return (qerror);
}


float gaussian_qerror(struct teach_params *teach, struct data_entry *sample,
		     int bx, int by, float radius)
{
  long index;
  int tx, ty, xdim, ydim;
  float dd;
  float alp;
  struct entries *codes = teach->codes;
  MAPDIST_FUNCTION *mdist = teach->mapdist;
  DIST_FUNCTION *distance = teach->dist;
  struct data_entry *codetmp;
  eptr p;
  float d, qerror;

  xdim = codes->xdim;
  ydim = codes->ydim;

  ifverbose(10)
    fprintf(stderr, "Best match in %d, %d\n", bx, by);

  codetmp = rewind_entries(codes, &p);
  index = 0;
  qerror = 0.0;

  while (codetmp != NULL)
    {
      tx = index % xdim;
      ty = index / xdim;

      ifverbose(11)
	fprintf(stderr, "Adapt unit %d, %d\n", tx, ty);
      dd = mdist(bx, by, tx, ty);

      alp = exp((double) (-dd * dd / (2.0 * radius * radius)));
      
      d = distance(codetmp, sample, codes->dimension);

      qerror += alp * d * d;

      codetmp = next_entry(&p);
      index++;
    }

  return (qerror);
}

/* find_qerror2 - calculate quantization error in a different way. */

float find_qerror2(struct teach_params *teach)
{
  float qerror;
  struct entries *data = teach->data;
  struct entries *codes = teach->codes;
  WINNER_FUNCTION *find_winner = teach->winner;
  NEIGH_QERROR *calc_qerror;
  struct data_entry *dtmp;
  struct winner_info win_info;
  int bxind, byind, length_known;
  long nod;
  float radius = teach->radius;
  eptr p;

  if (set_som_params(teach))
    {
      fprintf(stderr, "find_qerror2: can't set SOM parameters\n");
      return -1;
    }

  /* select neighborhood */
  if (teach->codes->neigh == NEIGH_GAUSSIAN)
    {
      ifverbose(3)
	fprintf(stderr, "qmode 1, gaussian neighbourhood\n");
      calc_qerror = gaussian_qerror;
    }
  else
    {
      ifverbose(3)
	fprintf(stderr, "qmode 1, bubble neighbourhood\n");
      calc_qerror = bubble_qerror;
    }

  /* Scan all data entries */

  qerror = 0.0;

  dtmp = rewind_entries(data, &p);

  if ((length_known = data->flags.totlen_known))
    nod = data->num_entries;
  else
    nod = 0;

  for (; dtmp != NULL; dtmp = next_entry(&p)) 
    {
      if (find_winner(codes, dtmp, &win_info, 1) == 0)
	continue; /* ignore empty vectors */

      bxind = win_info.index % codes->xdim;
      byind = win_info.index / codes->xdim;
      
      qerror += calc_qerror(teach, dtmp, bxind, byind, radius);
      
      if (length_known)
	ifverbose(1)
	  mprint((long) nod--);
    }

  if (length_known)
    ifverbose(1)
      {
	mprint((long) 0);
	fprintf(stderr, "\n");
      }
  
  return(qerror);
}

MAPDIST_FUNCTION *get_mapdistf(int topol)
{
  MAPDIST_FUNCTION *dist;

  switch (topol)
    {
    case TOPOL_RECT:
      dist = rect_dist;
      break;
    case TOPOL_HEXA:
      dist = hexa_dist;
      break;
    case TOPOL_LVQ:
    default:
      dist = NULL;
      break;
    }
  return dist;
}

NEIGH_ADAPT *get_nadaptf(int neigh)
{
  NEIGH_ADAPT *nadapt;

  switch (neigh)
    {
    case NEIGH_GAUSSIAN:
      nadapt = gaussian_adapt;
      break;
    case NEIGH_BUBBLE:
      nadapt = bubble_adapt;
      break;
    default:
      nadapt = NULL;
      break;
    }

  return nadapt;
}

/* set_som_params - set functions needed by the SOM algorithm in the
   teach_params structure */

int set_som_params(struct teach_params *params)
{
  if (!params->mapdist)
    if ((params->mapdist = get_mapdistf(params->topol)) == NULL)
      return 1;

  if (!params->neigh_adapt)
    if ((params->neigh_adapt = get_nadaptf(params->neigh)) == NULL)
      return 1;

  return 0;
}
