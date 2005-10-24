/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  lvq_pak.c                                                           *
 *   -very general routines needed in many places                       *
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
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "lvq_pak.h"
#include "datafile.h"

/* find_winner_euc - finds the winning entry (1 nearest neighbour) in
   codebook using euclidean distance. Information about the winning
   entry is saved in the winner_info structure. Return 1 (the number
   of neighbours) when successful and 0 when winner could not be found
   (for example, all components of data vector have been masked off) */

int find_winner_euc(struct entries *codes, struct data_entry *sample,
			struct winner_info *win, int knn)
{
  struct data_entry *codetmp;
  int dim, i, masked;
  float diffsf, diff, difference;
  eptr p;

  dim = codes->dimension;
  win->index = -1;
  win->winner = NULL;
  win->diff = -1.0;

  /* Go through all code vectors */
  codetmp = rewind_entries(codes, &p);
  diffsf = FLT_MAX;

  while (codetmp != NULL) {
    difference = 0.0;
    masked = 0;

    /* Compute the distance between codebook and input entry */
    for (i = 0; i < dim; i++)
      {
	if ((sample->mask != NULL) && (sample->mask[i] != 0))
	  {
	    masked++;
	    continue; /* ignore vector components that have 1 in mask */
	  }
	diff = codetmp->points[i] - sample->points[i];
	difference += diff * diff;
	if (difference > diffsf) break;
      }

    if (masked == dim)
      return 0; /* can't calculate winner, empty data vector */

    /* If distance is smaller than previous distances */
    if (difference < diffsf) {
      win->winner = codetmp;
      win->index = p.index;
      win->diff = difference;
      diffsf = difference;
    }

    codetmp = next_entry(&p);
  }

  if (win->index < 0)
    ifverbose(3)
      fprintf(stderr, "find_winner_euc: can't find winner\n");

  return 1; /* number of neighbours */
}

/* find_winner_knn - finds the winning entrys (k nearest neighbours)
   in codebook using euclidean distance. Information about the winning
   entry is saved in the winner_info structures provided by the
   caller. Return k (the number of neighbours) when successful and 0
   when winner could not be found (for example, all components of data
   vector have been masked off) */

int find_winner_knn(struct entries *codes, struct data_entry *sample,
		    struct winner_info *win, int knn)
{
  struct data_entry *codetmp;
  int dim, i, j, masked;
  float difference, diff;
  eptr p;

  if (knn == 1) /* might be a little faster */
    return find_winner_euc(codes, sample, win, 1);

  dim = codes->dimension;

  for (i = 0; i < knn; i++)
    {
      win[i].index = -1;
      win[i].winner = NULL;
      win[i].diff = FLT_MAX;
    }
  /* Go through all code vectors */

  codetmp = rewind_entries(codes, &p);

  while (codetmp != NULL) {
    difference = 0.0;

    masked = 0;
    /* Compute the distance between codebook and input entry */
    for (i = 0; i < dim; i++)
      {
	/* pitaisiko ottaa huomioon myos codebookissa olevat?? */
	if ((sample->mask != NULL) && (sample->mask[i] != 0))
	  {
	    masked++;
	    continue; /* ignore vector components that have 1 in mask */
	  }
	diff = codetmp->points[i] - sample->points[i];
	difference += diff * diff;
	if (difference > win[knn-1].diff) break;
      }

    if (masked == dim)
      return 0;

    /* If distance is smaller than previous distances */
    for (i = 0; (i < knn) && (difference > win[i].diff); i++);

    if (i < knn)
      {
	for (j = knn - 1; j > i; j--)
	  {
	    win[j].diff = win[j - 1].diff;
	    win[j].index = win[j - 1].index;
	    win[j].winner = win[j - 1].winner;
	  }

	win[i].diff = difference;
	win[i].index = p.index;
	win[i].winner = codetmp;
      }

    codetmp = next_entry(&p);
  }

  if (win->index < 0)
    ifverbose(3)
      fprintf(stderr, "find_winner_knn: can't find winner\n");

  return knn; /* number of neighbours */
}

/* vector_dist_euc - compute distance between two vectors is euclidean
   metric. Returns < 0 if distance couldn't be calculated (all components
   were masked off */

float vector_dist_euc(struct data_entry *v1, struct data_entry *v2, int dim)
{
  float diff, difference;
  int i, masked = 0;

  difference = 0.0;
  for (i = 0; i < dim; i++)
    {
      if (((v1->mask != NULL) && (v1->mask[i] != 0)) ||
	  ((v2->mask != NULL) && (v2->mask[i] != 0)))
	{
	  masked++;
	  /* ignore vector components that have 1 in mask */
	}
      else
	{
	  diff = v1->points[i] - v2->points[i];
	  difference += diff * diff;
	}
    }

  if (masked == dim)
    return -1;

  return sqrt(difference);
}

/* adapt_vector - move a codebook vector towards another vector */

void adapt_vector(struct data_entry *codetmp, struct data_entry *sample,
		  int dim, float alpha)
{
  int i;

  for (i = 0; i < dim; i++)
    if ((sample->mask != NULL) && (sample->mask[i] != 0))
      continue; /* ignore vector components that have 1 in mask */
    else
      codetmp->points[i] += alpha *
	(sample->points[i] - codetmp->points[i]);

}

/*******************************************************************
 * Routines for general usage                                      *
 *******************************************************************/

/* package errors */

int lvq_errno;

void errormsg(char *msg)
{
  fprintf(stderr, "%s\n", msg);
}

/* My own free routine */
void ofree(void *data)
{
  if (data != NULL)
    free(data);
}

/* oalloc - my malloc allocation routine with some error checking. Not
   used much any more as it exits if an error occurs. */

void *oalloc(unsigned int len)
{
  void *tmp;

  if ((tmp = malloc(len)) == NULL) {
    fprintf(stderr, "Can't allocate memory");
    exit(-1);
  }

  return(tmp);
}

/* orealloc - my realloc allocation routine with some error
   checking. Not used much any more as it exits if an error occurs. */

void *orealloc(void *po, unsigned int len)
{
  void *tmp;

  if ((tmp = realloc(po, len)) == NULL) {
    fprintf(stderr, "Can't reallocate memory");
    exit(-1);
  }

  return(tmp);
}


/* Print dots indicating that a job is in progress */
void mprint(long rlen)
{
#ifndef time_t
#define time_t long
#endif
  static time_t startt, prevt;
  time_t currt;
  static long totlen = 0;
  long t1, t2;
  int i;

  currt=time(NULL);

  if (!totlen) {
    totlen=rlen;
    startt=currt;
    fprintf(stderr, "               ");
    for (i=0; i<10; i++) fprintf(stderr, "------");
  }

  if (currt!=prevt || !rlen) {
    t1=currt-startt;
    if (rlen!=totlen) t2=(currt-startt)*(float)totlen/(totlen-rlen);
    else t2=0;
    if (t2>9999) {
      t1/=60;
      t2/=60;
      i=0;
    } else i=1;
    fprintf(stderr, "\r%4u/%4u %4s ", (int)t1, (int)t2, i?"sec.":"min.");
    if (totlen) {
      i=(int) (60*(float)(totlen-rlen)/totlen);
      while (i--) fprintf(stderr, ".");
    }
    fflush(stderr);
    prevt=currt;
  }
  if (!rlen) totlen=0;
}


static unsigned long next = 1;
#define RND_MAX 32767L

/* May define my own random generator */
void osrand(int i)
{
  next = i;
}

int orand()
{
  return((int) ((next = (next * 23) % 100000001) % RND_MAX));
}

/* init_random - initialize own random number generator with seed.
   If seed is 0, uses current time as seed. */

void init_random(int seed)
{
  if (!seed)
    osrand((int) time(NULL));
  else
    osrand(seed);
}

int verbose_level = 1;

int verbose(int level)
{
  if (level >= 0) {
    verbose_level = level;
  }

  return(verbose_level);
}

int silent(int level)
{
  static int silent_level = 0;

  if (level >= 0) {
    silent_level = level;
  }

  return(silent_level);
}

int use_fixed(int level)
{
  static int fixed_level = 0;

  if (level >= 0) {
    fixed_level = level;
  }

  return(fixed_level);
}

int use_weights(int level)
{
  static int weights_level = 0;

  if (level >= 0) {
    weights_level = level;
  }

  return(weights_level);
}

/* ostrdup - return a pointer to a duplicate of string str. If no memory
   for new string cannot be allocated, return NULL. */

char *ostrdup(char *str)
{
  char *tmp;
  int len;

  /* allocate mem for duplicate */
  len = strlen(str);
  if (len < 0)
    return NULL;
  tmp = malloc(sizeof(char) * (len + 1));
  if (tmp == NULL)
    {
      fprintf(stderr, "ostrdup: Can't allocate mem.\n");
      perror("ostrdup");
      return NULL;
    }

  /* copy string */
  strcpy(tmp, str);

  return(tmp);
}

/*******************************************************************
 * Routines to get the parameter values                            *
 *******************************************************************/

static int no_parameters = -1;

int parameters_left(void)
{
  return(no_parameters);
}

long oatoi(char *str, long def)
{
  if (str == (char *) NULL)
    return(def);
  else
    return(atoi(str));
}

float oatof(char *str, float def)
{
  if (str == (char *) NULL)
    return(def);
  else
    return((float) atof(str));
}

char *extract_parameter(int argc, char **argv, char *param, int when)
{
  int i = 0;

  if (no_parameters == -1)
    no_parameters = argc - 1;

  while ((i < argc) && (strcmp(param, argv[i]))) {
    i++;
  }

  if ((i <= argc - 1) && (when == OPTION2))
    {
      no_parameters -= 1;
      return "";
    }

  if (i < argc-1) {
    no_parameters -= 2;
    return(argv[i+1]);
  }
  else {
    if (when == ALWAYS) {
      fprintf(stderr, "Can't find asked option %s\n", param);
      exit(-1);
    }
  }

  return((char *) NULL);
}


/* global_options - handle some options that are common to all
   programs. Also read some environment variables. */

int global_options(int argc, char **argv)
{
  char *s;

  /* set program name */
  setprogname(argv[0]);

#ifndef NO_PIPED_COMMANDS
  /* command for compressing */

  s = getenv("LVQSOM_COMPRESS_COMMAND");
  if (s)
    compress_command = s;

  s = extract_parameter(argc, argv, "-compress_cmd", OPTION);
  if (s)
    compress_command = s;

  /* command for uncompressing */
  s = getenv("LVQSOM_UNCOMPRESS_COMMAND");
  if (s)
    uncompress_command = s;

  s = extract_parameter(argc, argv, "-uncompress_cmd", OPTION);
  if (s)
    uncompress_command = s;
#endif /* NO_PIPED_COMMANDS */

  /* string that identifies a vector component to be ignored in files */
  s = getenv("LVQSOM_MASK_STR");
  if (s)
    masked_string = s;

  s = extract_parameter(argc, argv, "-mask_str", OPTION);
  if (s)
    masked_string = s;

  if (extract_parameter(argc, argv, "-version", OPTION2))
    fprintf(stderr, "Version: %s\n", get_version());

  verbose(oatoi(extract_parameter(argc, argv, VERBOSE, OPTION), 1));

  return 0;
}

/* save_snapshot - save a snapshot of the codebook */

int save_snapshot(struct teach_params *teach, long iter)
{
  struct entries *codes = teach->codes;
  struct snapshot_info *shot = teach->snapshot;
  char filename[1024]; /* hope this is enough */
  char comments[1024];

  /* make filename */
  sprintf(filename, shot->filename, iter);

  /* open file for writing */

  fprintf(stderr, "saving snapshot: file '%s', type '%s'\n", filename,
	  get_str_by_id(snapshot_list, shot->type));

  sprintf(comments, "#SNAPSHOT FILE\n#iterations: %ld (%ld total)\n",
	  iter, teach->length);
  if (save_entries_wcomments(codes, filename, comments))
    return 1; /* error */

  return 0;
}

/* get_snapshot - allocate and initialize snapshot info */

struct snapshot_info *get_snapshot(char *filename, int interval, int type)
{
  struct snapshot_info *shot;

  shot = malloc(sizeof(struct snapshot_info));
  if (shot == NULL)
    {
      fprintf(stderr, "get_snapshot: Can't allocate structure\n");
      perror("get_snapshot");
      return NULL;
    }

  /* allocate room for string */
  shot->filename = NULL;
  if (filename)
    if ((shot->filename = malloc(strlen(filename) + 1)) == NULL)
      {
	fprintf(stderr, "get_snapshot: Can't allocate mem for string\n");
	perror("get_snapshot");
	free(shot);
	return NULL;
      }
    else
      strcpy(shot->filename, filename);

  shot->interval = interval;
  shot->type = type;

  fprintf(stderr, "snapshot: filename: '%s', interval: %d, type: %s\n",
	  shot->filename, shot->interval, get_str_by_id(snapshot_list, type));

  return shot;
}

/* free_snapshot - deallocate snapshot info */

void free_snapshot(struct snapshot_info *shot)
{
  if (shot)
    {
      if (shot->filename)
	free(shot->filename);
      free(shot);
    }
}

/* get_type_by_id - search typelist for id */

struct typelist *get_type_by_id(struct typelist *types, int id)
{
  for (;types->str; types++)
    if (types->id == id)
      break;

  return types;
}

/* get_type_by_str - search typelist for string */

struct typelist *get_type_by_str(struct typelist *types, char *str)
{
  for (;types->str; types++)
    if (str) /* NULL str gets the last item */
      //if (strcasecmp(types->str, str) == 0)
	break;

  return types;
}

/* -------  different alpha functions -------------- */

/* alpha functions */

struct typelist alpha_list[] = {
  {ALPHA_LINEAR, "linear", linear_alpha},
  {ALPHA_INVERSE_T, "inverse_t", inverse_t_alpha},
  {ALPHA_UNKNOWN, NULL, NULL}};      /* default */

/* linearly decreasing alpha */

float linear_alpha(long iter, long length, float alpha)
{
  return (alpha * (float) (length - iter) / (float) length);
}

#define INV_ALPHA_CONSTANT 100.0

float inverse_t_alpha(long iter, long length, float alpha)
{
  float c;

  c = length / INV_ALPHA_CONSTANT;

  return (alpha * c / (c + iter));
}

/* print_lines */

int print_lines(FILE *fp, char **lines)
{
  char *line;
  while ((line = *lines++))
    fputs(line, fp);

  return 0;
}

