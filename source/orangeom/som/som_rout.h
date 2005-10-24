#ifndef SOM_ROUT_H
#define SOM_ROUT_H
/************************************************************************
 *                                                                      *
 *  Program package 'som_pak':                                          *
 *                                                                      *
 *  som_rout.h                                                          *
 *  - header file for SOM routines                                      *
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

#include "lvq_pak.h"

typedef float NEIGH_QERROR(struct teach_params *teach,
			   struct data_entry *sample,
			   int bx, int by,
			   float radius);

struct entries *randinit_codes(struct entries *data, int topol, int neigh, int xdim, int ydim);
struct entries *lininit_codes(struct entries *data, int topol, int neigh, int xdim, int ydim);
void normalize(float *v, int n);
float dotprod(float *v, float *w, int n);
int gram_schmidt(float *v, int n, int e);
MAPDIST_FUNCTION hexa_dist, rect_dist, *get_mapdistf(int);
NEIGH_ADAPT bubble_adapt, gaussian_adapt, *get_nadaptf(int);
struct entries *som_training(struct teach_params *teach);
float find_qerror(struct teach_params *teach);
float find_qerror2(struct teach_params *teach);
NEIGH_QERROR bubble_qerror, gaussian_qerror;
int set_som_params(struct teach_params *params);

#endif /* SOM_ROUT_H */
