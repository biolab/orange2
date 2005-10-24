#ifndef _INCLUDED_CONFIG_H
#define _INCLUDED_CONFIG_H
/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  config.h                                                            *
 *   - configuration options for SOM/LVQ_PAK                            *
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

/* In files giving an x instead of a value marks that component to be
   ignored in calculations. The default is 'x'. If you want to change
   the string remove the comments around the line below and change it
   to whatever you like. Defining MASKED_VALUE here overrides the
   default in file datafile.h. This value can also be overridden with
   a command line option or with environment variable, see docs for
   details. */

#ifndef MASKED_VALUE
/* #define MASKED_VALUE "x" */
#endif /* MASKED_VALUE */

/* Machine / OS dependent options */

/* Default commands for compressing and uncompressing files in
   environments that support it. Defining these values here override
   the defaults in file fileio.h. These values can also be with a
   command line option or with environment variables, see docs for
   details. */

#ifndef DEF_COMPRESS_COM
#define DEF_COMPRESS_COM "gzip -9 -c >%s"
#endif /* DEF_COMPRESS_COM */
#ifndef DEF_UNCOMPRESS_COM
#define DEF_UNCOMPRESS_COM "gzip -d -c %s"
#endif /* DEF_UNCOMPRESS_COM */

/* options for MSDOS */

#ifdef __MSDOS__
#ifndef MSDOS
#define MSDOS
#endif
#endif /* __MSDOS__ */

#ifdef MSDOS

/* MSDOS doesn't have popen so compression and piped commands don't
   work. Undefine this if you have popen */

#define NO_PIPED_COMMANDS

/* Borland C doesn't have strcasecmp but has the function strcmpi that
   does the same thing */

#define strcasecmp(s1,s2) strcmpi(s1,s2)

#endif /* MSDOS */

/* definitions needed to get the program name in various environments */

/* the character that separates different directories in path name */
#ifndef DIRSEPARATOR
#if defined MSDOS
#define DIRSEPARATOR '\\'
#else /* other OSes */
#define DIRSEPARATOR '/'
#endif 
#endif

/* character that separates the drive part from the directory part */
#ifndef DRIVESEPARATOR
#if defined MSDOS
#define DRIVESEPARATOR ':'
#elif defined __amigados__
#define DRIVESEPARATOR ':'
#else /* other OSes */
/* not used */
#endif 
#endif

/* character that separates the filename suffix from the filename */
#ifndef SUFFIXSEPARATOR
#if defined MSDOS
#define SUFFIXSEPARATOR '.'
#else /* other OSes */
/* not used */
#endif 
#endif

#endif /* _INCLUDED_CONFIG_H */
