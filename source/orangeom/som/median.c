#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float median7(float yy,float kaa, float koo, float nee, float vii, float kuu,float see)
{
  float array[7];   float tmp;
  int i,j;

  array[0]=yy;  array[1]=kaa;  array[2]=koo;  array[3]=nee;  array[4]=vii;
  array[5]=kuu;  array[6]=see;
  
  for(j=0;j<7-1;j++)
    for(i=0;i<7-1-j;i++){
      if(array[i]>array[i+1]) {
	tmp=array[i];
	array[i] = array[i+1];
	array[i+1]=tmp;
      }
    }
  return(array[3]);
}

float median6(float yy,float kaa, float koo, float nee, float vii, float kuu)
{
  float array[6];   float tmp;
  int i,j;

  array[0]=yy;  array[1]=kaa;  array[2]=koo;  array[3]=nee;  array[4]=vii;
  array[5]=kuu;  
  
  for(j=0;j<6-1;j++)
    for(i=0;i<6-1-j;i++){
      if(array[i]>array[i+1]) {
	tmp=array[i];
	array[i] = array[i+1];
	array[i+1]=tmp;
      }
    }
  return(array[3]);
}

float median5(float yy,float kaa, float koo, float nee, float vii)
{
  float array[5];   float tmp;
  int i,j;

  array[0]=yy;  array[1]=kaa;  array[2]=koo;  array[3]=nee;  array[4]=vii;
 
  
  for(j=0;j<5-1;j++)
    for(i=0;i<5-1-j;i++){
      if(array[i]>array[i+1]) {
	tmp=array[i];
	array[i] = array[i+1];
	array[i+1]=tmp;
      }
    }
  return(array[2]);
}

float median4(float yy,float kaa, float koo, float nee)
{
  float array[4];   float tmp;
  int i,j;

  array[0]=yy;  array[1]=kaa;  array[2]=koo;  array[3]=nee;  
 
  
  for(j=0;j<4-1;j++)
    for(i=0;i<4-1-j;i++){
      if(array[i]>array[i+1]) {
	tmp=array[i];
	array[i] = array[i+1];
	array[i+1]=tmp;
      }
    }
  return(array[2]);
}


float median3(float yy,float kaa, float koo)
{
  float array[3];   float tmp;
  int i,j;

  array[0]=yy;  array[1]=kaa;  array[2]=koo;  
 
  
  for(j=0;j<3-1;j++)
    for(i=0;i<3-1-j;i++){
      if(array[i]>array[i+1]) {
	tmp=array[i];
	array[i] = array[i+1];
	array[i+1]=tmp;
      }
    }
  return(array[1]);
}


