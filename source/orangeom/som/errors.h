#ifndef _LVQ_ERRORS_H
#define _LVQ_ERRORS_H
/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  errors.h                                                            *
 *   - error codes used by some functions. These aren't used very       *
 *     much yet.                                                        *
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

/* error codes for SOM_PAK and LVQ_PAK */

#define ERR_OK         0  /* no error */
#define ERR_NOMEM      1  /* can't allocate memory */
#define ERR_FILEMODE   2  /* incorrect file mode */
#define ERR_NOPIPES    3  /* operations on pipes are not supported (on 
			     those systems that don't have popen() */
#define ERR_OPENFILE   4  /* can't open file. look at errno for more info */
#define ERR_COMMAND    5  /* can't execute command */
#define ERR_REWINDFILE  6 /* can't rewind file */
#define ERR_REWINDPIPE  7 /* can't rewind regular pipe */
#define ERR_LINETOOLONG 8 /* input line too long */
#define ERR_FILEERR    9  /* file error, see erno for more info */
#define ERR_HEADER     10 /* error in file headers */
#define ERR_FILEFORMAT 11 /* error in data file */
#define WARN_EMPTYENTRY 12 /* loaded entry was empty (all comps. masked off) */

#define clear_err() (lvq_errno = 0)
#define ERROR(n) (lvq_errno = n)

#endif /* _LVQ_ERRORS_H */
