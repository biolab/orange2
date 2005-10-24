#ifndef SOMPAK_FILEIO_H
#define SOMPAK_FILEIO_H
/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  fileio.h                                                            *
 *   - header file for fileio.c. 
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
#include <stdlib.h>
#include "errors.h"
#include "config.h"

/* these should be moved to config.h or something similar */
/* default commands for compressing and uncompressing files */

#ifndef DEF_COMPRESS_COM
#define DEF_COMPRESS_COM "gzip -9 -c >%s"
#endif /* DEF_COMPRESS_COM */
#ifndef DEF_UNCOMPRESS_COM
#define DEF_UNCOMPRESS_COM "gzip -d -c %s"
#endif /* DEF_UNCOMPRESS_COM */

#ifndef NO_PIPED_COMMANDS
extern char *compress_command, *uncompress_command;
#endif /* NO_PIPED_COMMANDS */

struct file_info {
  char *name;
  FILE *fp;
  struct {
    unsigned int compressed : 1; /* is the file compressed */
    unsigned int pipe : 1;       /* the file is a pipe */
    unsigned int eof : 1;        /* has end of line been reached */
  } flags;
  int error;                     /* error code or 0 if OK */
  long lineno;                   /* line number we are on */
};

#define fi2fp(fi) ((fi != NULL) ? (fi)->fp : NULL)

#ifndef STR_LNG
#define STR_LNG 1000
#endif
#ifndef ABS_STR_LNG
#define ABS_STR_LNG 10000
#endif

/* prototypes */

struct file_info *open_file(char *name, char *fmode);
int close_file(struct file_info *fi);

#define getline getline_som
char *getline(struct file_info *fi);

/* for getting the program name */
char *setprogname(char *argv0);
#define getprogname() setprogname(NULL);

#endif /* SOMPAK_FILEIO_H */
