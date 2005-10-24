/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  fileio.c                                                            *
 *   - routines for reading and writing files. Features include         *
 *     transparent use of compression and decompression, stdin/stdout   *
 *     and reading/writing from/to piped commands.                      *
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
#include <string.h>
#include "fileio.h"

/* prototypes for local functions */
static char *check_for_compression(char *name);
static struct file_info *alloc_file_info(void);

#define FM_READ 1
#define FM_WRITE 2

#ifdef _MSC_VER
#define popen(a,b) NULL
#define pclose(a) NULL
#endif

/* set default commands for compressing and uncompressing of files */

#ifndef NO_PIPED_COMMANDS
char *compress_command = DEF_COMPRESS_COM;
char *uncompress_command = DEF_UNCOMPRESS_COM;
#endif /* NO_PIPED_COMMANDS */

static char *stdin_name = "(stdin)";
static char *stdout_name = "(stdout)";

/* open_file - open a file for reading or writing. If name == NULL,
   uses standard input or output. Allowed characters in fmode are 'r'
   for reading, 'w' for writing, 'z' for compressed files and 'p' for
   piping output/input to/from a command (name is command string). 'P'
   and 'z' are not required in the fmode because they can be guessed
   from the filename: if the name and with the suffix .gz, .z or .Z
   the z-mode is assumed. If the name starts with the unix pipe
   character '|', the p-mode is assumed and the rest of the name is
   used as the command to be run. */

struct file_info *open_file(char *name, char *fmode)
{
  FILE *fp;
  struct file_info *fi = NULL;
  char buf[1000], *s, fmode2[10];
  int mode = 0;       /* FM_READ = read, FM_WRITE = write */
  int compress = 0;   /* 0 = no compression, 1 = compress */
  int piped_com = 0;  /* piped command? */
  int len;

  fi = alloc_file_info();
  if (fi == NULL)
    return NULL;


  /* copy mode string as we may change it later */
  strcpy(fmode2, fmode); 
  fmode = fmode2;

  /* get mode: read/write, compression? */

  if (strchr(fmode, 'r'))
    mode = FM_READ;
  else if (strchr(fmode, 'w'))
    mode = FM_WRITE;
  else
    {
      /* one of the above must be used */
      fprintf(stderr, "open_file: incorrect file mode: %s for file %s\n ", 
	      fmode, name);
      close_file(fi);
      return NULL;
    }
    
  /* compression? */
  if ((s = strchr(fmode, 'z')))
    {
      compress = 1;
      strcpy(s, s + 1); /* remove z from mode */
    }

  /* piped command? */
  if ((s = strchr(fmode, 'p')))
    {
      piped_com = 1;
      compress = 0; /* do not allow compress and pipe together */
      strcpy(s, s + 1); /* remove p from mode */
    }

  /* if filename starts with '|' use piped command */
  if (name && (name[0] == '|'))
    {
      piped_com = 1;
      compress = 0;  /* do not allow compress and pipe together */
      name++; 
    }

  if (!piped_com)
    if (check_for_compression(name))
      compress = 1;

  /* "-" as name means use stdin/out */
  if (name)
    if (strcmp("-", name) == 0)
      name = NULL;

  /* use stdin/stdout if name == NULL */
  if (name == NULL)
    {
      fp = (mode == FM_READ) ? stdin : stdout;
      /* assume that stdin/out is a regular file or that we don't need to
	 rewind it */
      name = (mode == FM_READ) ? stdin_name : stdout_name;
    }
  else 
    if (compress || piped_com)
      {
#ifndef NO_PIPED_COMMANDS
	if (compress)
	  {
	    /* compressed files */
	    sprintf(buf, (mode == FM_READ) ? 
		    uncompress_command : compress_command, 
		    name);
	    fi->flags.compressed = 1;
	  }
	else
	  {
	    /* piped commands */
	    strcpy(buf, name);
	  }
	fp = popen(buf, (mode == FM_READ) ? "r" : "w" );
	if (fp == NULL)
	  {
	    fprintf(stderr, "open_file: can't execute command '%s'\n",
		    buf);
	    perror("open_file");
	  }
	fi->flags.pipe = 1;
#else /* !NO_PIPED_COMMANDS */
	fprintf(stderr, "open_file: file %s: piped commands are not supported is this version\n", name);
	close_file(fi);
	return NULL;
#endif /* !NO_PIPED_COMMANDS */
      }
    else
      {
	fp = fopen(name, fmode);
	if (fp == NULL)
	  {
	    fprintf(stderr, 
		    "file_open: can't open file '%s' for %s, mode '%s'\n",
		    name, 
		    (mode == FM_READ) ? "reading" : "writing",
		    fmode);
	    perror("file_open");
	    close_file(fi);
	    return NULL;
	  }
      }
  fi->fp = fp;

  fi->name = NULL;

  /* copy name */
  if (name)
    {
      len = strlen(name);
      if (len > 0)
	{
	  fi->name = malloc(len + 1);
	  if (fi->name == NULL)
	    {
	      fprintf(stderr, "open_file: can't allocate mem for name\n");
	      close_file(fi);
	      return NULL;
	    }

	  strcpy(fi->name, name);
	}
    }

  return fi;
}


/* alloc_file_info - allocate and initialize file_info structure */

static struct file_info *alloc_file_info(void)
{
  struct file_info *fi;

  fi = malloc(sizeof(struct file_info));
  if (fi == NULL)
    {
      perror("alloc_file_info");
      return NULL;
    }

  /* initialize structure members to reasonable values */
  fi->name = NULL;
  fi->fp = NULL;
  fi->flags.pipe = 0;
  fi->flags.compressed = 0;
  fi->flags.eof = 0;
  fi->lineno = 0;
  return fi;
}

/* close_file - close file and deallocate file_info */

int close_file(struct file_info *fi)
{
  int retcode = 0;

  if (fi)
    {
      if (fi->name)
	free(fi->name);
      if (fi->fp)
#ifndef NO_PIPED_COMMANDS
	if (fi->flags.pipe) /* piped commands + compressed files */
	  retcode = pclose(fi->fp);
	else
#endif /* NO_PIPED_COMMANDS */
	  retcode = fclose(fi->fp);
      free(fi);
    }
  return retcode;
}

/* check_for_compression - check if name indicates compression,
   i.e. the ending is one of .gz, .z, .Z. If suffix is found, returns
   a pointer to the dot, otherwise returns NULL */

static char *check_for_compression(char *name)
{
  char *s;
  
  if (name == NULL)
    return NULL;

  /* look for the last '.' in name */
  s = strrchr(name, '.');
  
  if (s == NULL) /* no suffix */
    return NULL;

  if (strcmp(s, ".gz") == 0) /* compressed with gzip */
    return s;

  if (strcmp(s, ".z") == 0)  /* compressed with gzip (older version) */
    return s;

  if (strcmp(s, ".Z") == 0)  /* compressed with compress */
    return s;
    
  /* unknown suffix */
  return NULL;
}    


/* getline - get a line from file. Returns a char * to the line (a
   static buffer), NULL on error. */
    
char *getline(struct file_info *fi)
{
  static char *stre = NULL;
  static long strl = 0;
  long len;
  int c;
  char *tstr;
  FILE *fp = fi2fp(fi);

  fi->error = 0;
  fi->flags.eof = 0;
  /* allocate memory for line buffer */

  if (stre == NULL) 
    {
      strl = STR_LNG;
      stre = (char *) malloc(sizeof(char) * strl);
      if (stre == NULL)
	{
	  perror("getline");
	  fi->error = ERR_NOMEM;
	  return NULL;
	}
    }

  len = 0;
  /* increment file line number */
  fi->lineno += 1;
  tstr = stre;

  while (1)
    {

      /* if line buffer is too small, increase its size */
      if (len >= strl)
	{
	  strl += STR_LNG;
	  tstr = stre = realloc(stre, sizeof(char) * strl);
	  if (stre == NULL)
	    {
	      perror("getline");
	      fi->error = ERR_NOMEM;
	      return NULL;
	    }

	  if (strl > ABS_STR_LNG) 
	    {
	      fprintf(stderr, "getline: Too long lines in file %s (max %d)\n",
		      fi->name, ABS_STR_LNG);
	      fi->error = ERR_LINETOOLONG;
	      return NULL;
	    }
	}

      /* get next character */
      c = fgetc(fp);

      /* end of line? */
      if (c == '\n')
	{
	  tstr[len] = '\0';
	  break;
	}

      /* end of file / error? */
      if (c == EOF)
	{
	  tstr[len] = '\0';

	  /* really an EOF? */
	  if (feof(fp))
	    {
	      /* we are at the end of file */
	      fi->flags.eof = 1;
	      if (len == 0)
		tstr = NULL;
	    }
	  else
	    {
	      /* read error */
	      tstr = NULL;
	      fi->error = ERR_FILEERR;
	      fprintf(stderr, "getline: read error on line %d of file %s\n", fi->lineno, fi->name);
	      perror("getline");
	    }
	  break;
	}

      tstr[len++] = c;
    }
  
  return tstr;
}

/* *********** routines for getting the program name ********** */

static char progname_real[512];

char *setprogname(char *argv0)
{
  char *s, *s2;
  static char *progname = NULL;

  if (argv0 == NULL)
    return progname;

  s = argv0;
#ifdef DRIVESEPARATOR
  /* remove the drive part from the filename */
  s2 = strchr(s, DRIVESEPARATOR);
  if (s2)
    s = s2 + 1;
#endif /* DRIVESEPARATOR */

  /* remove all directories from name, leaving only the filename */
  s2 = strrchr(s, DIRSEPARATOR);
  if (s2)
    s = s2 + 1;

  s = strcpy(progname_real, s);
  
#ifdef SUFFIXSEPARATOR
  /* remove suffix from filename */
  s2 = strrchr(s, SUFFIXSEPARATOR);
  if (s2)
    *s2 = '\0';
#endif /* SUFFIXSEPARATOR */

  progname = s;
  return progname;
}
