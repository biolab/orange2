/********************************************************************
*
*   Name:                modul error
*
*   Description:  reports errors in uniform way throughout the system
*
*********************************************************************/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "general.h"

void error(const char* Msg1, const char* Msg2)
{
    fprintf(stderr, "\nERROR: %s", Msg1) ;
    if (strlen(Msg1) + strlen(Msg2) + 8 > 80)
       fprintf(stderr, "\n") ;
    fprintf(stderr," %s\n", Msg2) ;
    fflush(stderr) ;
}

void stop(const char* Msg1, const char* Msg2)
{
    fprintf(stderr, "\nFATAL ERROR: %s", Msg1) ;
    if (strlen(Msg1) + strlen(Msg2) + 8 > 80)
       fprintf(stderr, "\n") ;
    fprintf(stderr," %s\n", Msg2) ;
    fflush(stderr) ;
    exit(1);
}
