/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  datafile.c                                                          *
 *   - routines for manipulating codebooks and data files               *
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
#include "lvq_pak.h"
#include "fileio.h"
#include "datafile.h"

/* open_data_file - opens a data file for reading. Returns a pointer to
   entries-structure or NULL on error. If name is NULL, just allocates
   the data structure but doesn't open any file. */

#ifdef _MSC_VER
#define popen(a,b) NULL
#define pclose(a) NULL
#endif

struct entries *open_data_file(char *name)
{
  struct entries *entries;

  lvq_errno = 0;

  entries = alloc_entries();
  if (entries == NULL)
    return NULL;

  if (name)
    {
      if ((entries->fi = open_file(name, "r")) == NULL)
	{
	  fprintf(stderr, "Can't open file %s", name);
	  return NULL;
	}
    }

  return entries;
}

/* alloc_entries - allocate and initialize an entries-structure. Return NULL
   on error. */

struct entries *alloc_entries(void)
{
  struct entries *en;

  if ((en = (struct entries*)malloc(sizeof(struct entries))) == NULL)
    {
      perror("alloc_entries");
      return NULL;
    }

  /* initialize to reasonable values */
  en->dimension = 0;
  en->topol = TOPOL_UNKNOWN;
  en->neigh = NEIGH_UNKNOWN;
  en->xdim = en->ydim = 0;
  en->current = en->dentries = NULL;
  en->num_loaded = 0;
  en->num_entries = 0;

  en->lap = 0;
  en->buffer = 0;
  en->flags.loadmode = LOADMODE_ALL;
  en->flags.totlen_known = 0;
  en->flags.random_order = 0;
  en->flags.skip_empty = 1;
  en->flags.labels_needed = (!label_not_needed(-1));
  return en;
}

/* copy_entries - create a new entries structure with the same parameters
   as the original. Doesn't copy data_entrys */
//changed new -> mnew
struct entries *copy_entries(struct entries *entr)
{
  struct entries *mnew;

  mnew = alloc_entries();
  if (mnew == NULL)
    return NULL;

  mnew->dimension = entr->dimension;
  mnew->topol = entr->topol;
  mnew->neigh = entr->neigh;
  mnew->xdim = entr->xdim;
  mnew->ydim = entr->ydim;

  return mnew;
}


/* read_headers - reads the header information from file and sets the
   entries variables accordingly. Return a non-zero value on error. */

int read_headers(struct entries *entries)
{
  struct file_info *fi = entries->fi;
  char *iline;
  int dim, sta;
  long row = 0;
  int error = 0;

  clear_err();
  /* Find the first not-comment line */

  do
    {
      iline = getline(fi);
      row++;
      if (iline == NULL) {
	fprintf(stderr, "Can't read file %s", fi->name);
	return lvq_errno;
      }

    } while (iline[0] == '#');

  /* get dimension */
  sta = sscanf(iline, "%d", &dim);
  if (sta <= 0) {
    fprintf(stderr, "Can't read dimension parameter in file %s", fi->name);
    return ERR_HEADER;
  }

  entries->dimension = dim;
  entries->topol = get_topol(iline);
  entries->neigh = get_neigh(iline);
  entries->xdim = get_xdim(iline);
  entries->ydim = get_ydim(iline);

  return error;
}

/* skip_headers - skip over headers of a file. Used when a file is re-opened.
   Returns a non-zero value on error. */

int skip_headers(struct file_info *fi)
{
  char *iline;

  /* Currently all header information is on the first non-comment line, so
     we just skip it. */

  while ((iline = getline(fi)) != NULL)
    if (iline[0] != '#')
      break;

  if (iline == NULL)
    {
      fprintf(stderr, "skip_headers: error reading file %s\n", fi->name);
      return lvq_errno;
    }

  return 0;
}

/* rewind_file - go to the beginning of file to the point where the first
   data entry is. If file is an ordinary file, seeks to the start of file.
   If the file is a compressed file, closes the old file and runs the
   uncompressing command again. Returns 0 on success, error code otherwise. */

int rewind_file(struct file_info *fi)
{
  char buf[512];

  if (!fi->flags.pipe)
    {
      /* not a pipe, so assume that it is a regular file */
      rewind(fi->fp);
    }
  else
    {
#ifndef NO_PIPED_COMMANDS
      if (fi->flags.compressed)
	{
	  /* close old process */
	  pclose(fi->fp);
	  /* reopen compressed file */
	  /* Compressed rewind works only for reading */
	  sprintf(buf, uncompress_command, fi->name);
	  fi->fp = popen(buf, "r");
	  if (fi->fp == NULL)
	    {
	      fprintf(stderr, "rewind_file: can't execute command '%s'\n",
		      buf);
	      perror("open_file");
	      return ERR_COMMAND;
	    }
	}
      else
	{
	  fprintf(stderr, "can't rewind piped command\n");
	  return ERR_REWINDPIPE;
	}
#else /* NO_PIPED_COMMANDS */
      fprintf(stderr, "rewind_file: Only regular files supported\n");
      return ERR_REWINDPIPE;
#endif /* NO_PIPED_COMMANDS */
    }
  /* set flags */
  fi->flags.eof = 0;
  fi->error = 0;
  fi->lineno = 0;

  /* skip the headers to to go the first entry */
  return skip_headers(fi);
}

/* open_entries - open a data file. Returns a pointer to a ready-to-use
   entries -structure. Return NULL on error. */

struct entries *open_entries(char *name)
{
  struct entries *entries;

  /* open file */
  if ((entries = open_data_file(name)) == NULL)
    return NULL;

  /* read headers */
  if (read_headers(entries))
    {
      close_entries(entries);
      return NULL;
    }

  return entries;
}

/* close_entries - deallocates an entries-file. Frees memory allocated for
   entry-list and closes the file associated with entries if there is one. */

void close_entries(struct entries *entries)
{
  if (entries)
    {
      /* deallocate data */
      if (entries->dentries)
	free_entrys(entries->dentries);

      /* close file */
      if (entries->fi)
	close_file(entries->fi);

      /* free memory allocated for structure */
      free(entries);
    }
}

/* read_entries - reads data from file to memory. If LOADMODE_ALL is
   used the whole file is loaded into memory at once and the file is
   closed. If buffered loading (LOADMODE_BUFFER) is used at most N
   data vectors are read into memory at one time. The buffer size N is
   given in the entries->buffer field. In both cases if there are any
   previous entries in the entries structure, they are overwritten and
   the memory space allocated for them is reused. */

struct entries *read_entries(struct entries *entries)
{
  long noc = 0;
  struct data_entry *entr, *prev, *next;
  struct file_info *fi = entries->fi;

  /* get first entry */

  next = entries->dentries;
  entr = next;
  if (next)
    next = next->next;

  entr = load_entry(entries, entr);
  prev = entr;

  entries->dentries = entr;
  if (entr == NULL)
    {
      if (lvq_errno)
	{
	  fprintf(stderr, "read_entries: Error loading from file %s\n",
		  fi->name);
	}
      return NULL;
    }

  noc++;
  do
    {
      /* if buffering is wanted and enough has been loaded */
      if ((entries->flags.loadmode == LOADMODE_BUFFER) &&
	  (noc >= entries->buffer))
	break;

      entr = next;
      if (next)
	next = next->next;

      entr = load_entry(entries, entr);
      prev->next = entr;

      prev = entr;
      if (entr)
	noc++;
    } while (entr != NULL);

  /* deallocate remaining entries in list */
  {
    long freed = 0;
    while (next)
      {
	entr = next->next;
	free_entry(next);
	next = entr;
	freed++;
      }
  }

  if (lvq_errno)
    {
      /* error loading entry */
      fprintf(stderr, "read_entries: error loading entry from file %s, aborting loading\n", fi->name);
      return NULL;
    }

  entries->num_loaded = noc;

  /* If all were loaded, close file */
  if (entries->flags.loadmode == LOADMODE_ALL)
    {
      entries->num_entries = noc;
      entries->flags.totlen_known = 1;
      close_file(entries->fi);
      entries->fi = NULL;
    }
  else
    {
      /* With buffered loading we dont know the length of the file
	 until it has been read once. The following trys to do this */

      /* if the total length is file is not known, add the number of
	 vectors loaded just now to the total count */
      if (!entries->flags.totlen_known)
	entries->num_entries += noc;

      /* if we are at the end of the file we can stop counting as we
         now know the total length */
      if (entries->fi->flags.eof)
	{
	  entries->flags.totlen_known = 1;
	  if (noc == entries->num_entries)
	    {
	      fprintf(stderr, "read_entries: file %s; size less than buffer size, switching buffering off\n", fi->name);
	      close_file(entries->fi);
	      entries->fi = NULL;
	      entries->flags.loadmode = LOADMODE_ALL;
	    }
	}
    }

  /* Randomize entry order if wanted. */

  if (entries->flags.random_order)
    entries->dentries = randomize_entry_order(entries->dentries);

  return(entries);
}

 /******************************************************************
 * Routines to store codebook vectors                              *
 *******************************************************************/

/* save_entries_wcomments - saves data to a file with optional comment
   or header lines. Returns a non-zero value on error. */

int save_entries_wcomments(struct entries *codes, char *out_code_file, char *comments)
{
  struct file_info *fi;
  struct data_entry *entry;
  eptr p;
  int error = 0;

  fi = open_file(out_code_file, "w");
  if (fi == NULL) {
    fprintf(stderr, "save_entries: Can't open file '%s'\n", out_code_file);
    return 1;
  }

  /* write header */
  if (write_header(fi, codes))
    {
      fprintf(stderr, "save_entries: Error writing headers\n");
      error = 1;
      goto end;
    }
  /* write comments if there are any */
  if (comments)
    fputs(comments, fi2fp(fi));

  /* write entries */
  for (entry = rewind_entries(codes, &p); entry != NULL; entry = next_entry(&p))
    {
      if (write_entry(fi, codes, entry))
	{
	  fprintf(stderr, "save_entries: Error writing entry, aborting\n");
	  error = 1;
	  goto end;
	}
    }

 end:
  if (fi)
    close_file(fi);
  return error;
}

/* write_header - writes header information to datafile */

int write_header(struct file_info *fi, struct entries *codes)
{
  FILE *fp = fi2fp(fi);

  if (codes != NULL) {
    fprintf(fp, "%d", codes->dimension);
    if (codes->topol > TOPOL_DATA) {
      fprintf(fp, " %s", topol_str(codes->topol));
      if (codes->topol > TOPOL_LVQ) {
	fprintf(fp, " %d", codes->xdim);
	fprintf(fp, " %d", codes->ydim);
	fprintf(fp, " %s", neigh_str(codes->neigh));
      }
    }
    fprintf(fp, "\n");
  }

  /* Some kind of error checking could be added... */
  return 0;
}


/* write_entry - writes one data entry to file. */

int write_entry(struct file_info *fi, struct entries *entr,
		struct data_entry *entry)
{
  FILE *fp = fi2fp(fi);
  int i, label;

  /* write vector */
  for (i = 0; i < entr->dimension; i++)
    if ((entry->mask != NULL) && (entry->mask[i] != 0 ))
      fprintf(fp, "%s ", masked_string);
    else
      fprintf(fp, "%g ", entry->points[i]);


  /* Write labels. The last label is empty */
  for (i = 0;;i++)
    {
      label = get_entry_labels(entry, i);
      if (label != LABEL_EMPTY)
	fprintf(fp, "%s ", find_conv_to_lab(label));
      else
	break;
    }
  fprintf(fp, "\n");

  /* Some kind of error checking could be added ... */
  return 0;
}

/* initialize and possibly allocate room for data_entry. If entry is NULL,
   a new entry is allocated and initialized. If entry is a pointer to an
   old entry, the old entry is re-initialized. Return NULL on error. */

struct data_entry *init_entry(struct entries *entr, struct data_entry *entry)
{
  clear_err();
  if (entry == NULL)
    {
      entry = (struct data_entry*)malloc(sizeof(struct data_entry));
      if (entry == NULL)
	{
	  perror("alloc_entry");
	  ERROR(ERR_NOMEM);
	  return NULL;
	}

      entry->fixed = NULL;
      entry->next = NULL;
      entry->mask = NULL;
      entry->lab.label_array = NULL;
      entry->num_labs = 0;

      entry->points = (float*)calloc(entr->dimension, sizeof(float));
      if (entry->points == NULL)
	{
	  free_entry(entry);
	  perror("alloc_entry");
	  ERROR(ERR_NOMEM);
	  return NULL;
	}
    }

  /* discard mask */
  if (entry->mask)
    {
      free(entry->mask);
      entry->mask = NULL;
    }

  /* discard fixed point */
  if (entry->fixed)
    {
      free(entry->fixed);
      entry->fixed = NULL;
    }

  clear_entry_labels(entry);
  entry->weight = 0;

  return entry;
}

/* free_entry - deallocates a data_entry. */

void free_entry(struct data_entry *entry)
{
  if (entry)
    {
      if (entry->points)
	free(entry->points);
      if (entry->fixed)
	free(entry->fixed);
      if (entry->mask)
	free(entry->mask);
      clear_entry_labels(entry);
      free(entry);
    }
}

/* set_mask - sets a value in mask */
//changed to explicit casting
char *set_mask(char *mask, int dim, int n)
{
  clear_err();

  if (mask == NULL)
    {
      mask = (char*)malloc(dim);
      if (mask == NULL)
	{
	  fprintf(stderr, "set_mask: failed to allocate mask\n");
	  perror("set_mask");
	  ERROR(ERR_NOMEM);
	  return NULL;
	}
      memset(mask, 0, dim);
    }

  if (n >= 0)
    mask[n] = 1;
  return mask;
}


/* the string that indicates a vector component that should be ignored */

char *masked_string = MASKED_VALUE;

/* load_entry - loads one data_entry from file associated with entr. If
   entry is non-NULL, an old data_entry is reused, otherwise a new entry
   is allocated. Returns NULL on error. */

struct data_entry *load_entry(struct entries *entr, struct data_entry *entry)
{
  int i;
  float ent;
  char lab[STR_LNG];
  char *toke, *line;
  long row;
  int dim, label_found;
  int entry_is_new = !entry;
  char *mask = NULL;
  int maskcnt;  /* now many components are masked */
  struct file_info *fi = entr->fi;
  dim = entr->dimension;

  clear_err();

  /* read next line */
 read_next_line:
  line = NULL;
  while (!line)
    {
      /* get line from file */
      line = getline(fi);

      /* The caller should check the entr->fi->error for errors or end
	 of file */

      if (line == NULL)
	{
	  ERROR(fi->error);
	  return NULL;
	}

      /* skip comments */
      if (line[0] == '#')
	line = NULL;
    }

  row = entr->fi->lineno;

  /* If entry is given, a new entry is loaded on over the old one. If
     entry == NULL, room for the new entry is allocated */

  entry = init_entry(entr, entry);
  if (entry == NULL)
    return NULL;

  maskcnt = 0;
  toke = strtok(line, SEPARATOR_CHARS);
  /* Read the first vector value */
  if (strcmp(toke, masked_string) == 0)
    {
      mask = set_mask(mask, dim, 0);
      if (mask == NULL)
	{
	  if (entry_is_new)
	    free_entry(entry);
	  return NULL;
	}

      maskcnt++;
      ent = 0.0;
    }
  else
    if (sscanf(toke, "%f", &ent) <= 0) {
      fprintf(stderr, "Can't read entry on line %d, component 0\n", row);
      if (entry_is_new)
	free_entry(entry);

      ERROR(ERR_FILEFORMAT);
      return NULL;
    }
  entry->points[0] = ent;


  /* Read the other vector values */
  for (i = 1; i < dim; i++) {
    toke = strtok(NULL, SEPARATOR_CHARS);
    if (toke == NULL) {
      fprintf(stderr, "load_entry: can't read entry in file %s on line %d, component %d\n",
	      fi->name, row, i);
      if (entry_is_new)
	free_entry(entry);
      ERROR(ERR_FILEFORMAT);
      return NULL;
    }

    if (strcmp(toke, masked_string) == 0)
      {
	mask = set_mask(mask, dim, i);
	if (mask == NULL)
	  {
	    if (entry_is_new)
	      free_entry(entry);
	    return NULL;
	  }
	maskcnt++;
	ent = 0.0;
      }
    else
      if (sscanf(toke, "%f", &ent) <= 0) {
	fprintf(stderr, "load_entry: can't read entry in file %s on line %d, component %d\n",
	      fi->name, row, i);
	if (entry_is_new)
	  free_entry(entry);
	ERROR(ERR_FILEFORMAT);
	return NULL;
      }
    entry->points[i] = ent;
  }

  /* Entries with all components masked off are normally discarded but
     they are loaded if the skip_empty-flag is set in the flags of the
     file. */

  if (maskcnt == dim)
    {
      if (entr->flags.skip_empty)
	{
	  ifverbose(3)
	    fprintf(stderr, "load_entry: skipping line %d of file %s, all components are masked off\n", row, fi->name);
	  free(mask);
	  mask = NULL;
	  goto read_next_line; /* load next line */
	}
      else
	ifverbose(3)
	  fprintf(stderr, "load_entry: loading line %d of file %s, all components are masked off\n", row, fi->name);
    }

  if (mask)
    {
      entry->mask = mask;
      mask = NULL;
    }

  /* Now the following tokens (if any) are label,
     weight term and fixed point description.
     Sometimes label is not needed. Other terms are never
     needed */

  label_found = 0;

  while ((toke = strtok(NULL, SEPARATOR_CHARS)) != NULL)
    {
      if (strncmp(toke, "weight=", 7) == 0)
	entry->weight = get_weight(toke);
      else if (strncmp(toke, "fixed=", 6) == 0)
	{
	  if ((entry->fixed = get_fixed(toke)) == NULL)
	    {
	      fprintf(stderr, "bad fixed point, line %d of file %s\n",
		      row, fi->name);
	      if (entry_is_new)
		free_entry(entry);
	      ERROR(ERR_FILEFORMAT);
	      return NULL;
	    }
	}
      else
	{
	  if (sscanf(toke, "%s", lab) <= 0) {
	    fprintf(stderr, "Can't read entry label on line %dof file %s\n",
		    row, fi->name);
	    if (entry_is_new)
	      free_entry(entry);
	    ERROR(ERR_FILEFORMAT);
	    return NULL;
	  }

	  add_entry_label(entry, find_conv_to_ind(lab));
	  label_found++;
	}
    }

  if ((entr->flags.labels_needed) && (!label_found))
    {
      fprintf(stderr, "Required label missing on line %d of file %s\n",
	      row, fi->name);
      if (entry_is_new)
	free_entry(entry);
      ERROR(ERR_FILEFORMAT);
      return NULL;
    }

  return(entry);
}

/* next_entry - Get next entry from the entries table. Returns NULL when
   at end of table or end of file is encountered. If loadmode is buffered,
   loads more data from file when needed. */

struct data_entry *next_entry(eptr *ptr)
{
  struct entries *entries = ptr->parent;
  struct data_entry *next, *current;

  current = ptr->current;

  if (current == NULL)
    return NULL;

  next = current->next;

  if (next == NULL)
    if (entries->flags.loadmode == LOADMODE_BUFFER)
      {
	if (entries->fi->flags.eof)
	  next = NULL; /* end of file, no more lines */
	else
	  {
	    /* load more lines */
	    if (read_entries(entries) == NULL)
	      next = NULL;
	    else
	      next = entries->dentries;
	  }
      }
  ptr->current = next;
  if (next)
    ptr->index++;
  return next;
}

/* rewind_entries - go to the first entry in entries list. Returns pointer
   to first data_entry. Loads data from file if it hasn't been loaded yet and
   rewinds file if we are using buffered reading. */

struct data_entry *rewind_entries(struct entries *entries, eptr *ptr)
{
  struct data_entry *current = entries->dentries;
  struct file_info *fi;

  ptr->parent = entries;

  if (entries->flags.loadmode == LOADMODE_BUFFER)
    {

      /* buffered loading */
      fi = entries->fi;
      if ((fi->flags.eof) || (current != NULL))
	{
	  /* if we are at the end of file, need to rewind the file */
	  if (rewind_file(fi))
	    {
	      fprintf(stderr, "error rewinding file\n");
	      return NULL;
	    }
	}

      if (!read_entries(entries))
	{
	  fprintf(stderr, "rewind_entries failed\n");
	  return NULL;
	}
      current = entries->dentries;

    }
  else
    {
      /* not buffered */
      if ((current == NULL) && (!entries->flags.totlen_known))
	{
	  /* file not loaded into memory */
	  if (!read_entries(entries))
	    {
	      fprintf(stderr, "rewind_entries failed\n");
	      return NULL;
	    }
	  current = entries->dentries;
	}
    }

  ptr->current = current;
  ptr->index = 0;
  entries->lap++;

  return current;
}

/* free_entrys - Free a list of entries */

void free_entrys(struct data_entry *data)
{
  struct data_entry *tmp;

  for (;data != NULL; data = tmp)
    {
      tmp = data->next;
      free_entry(data);
    }
}

/* copy_entry - Copy one entry (next==NULL) */

struct data_entry *copy_entry(struct entries *entries, struct data_entry *data)
{
  int i;
  struct data_entry *tmp;

  clear_err();
  /* allocate memory for the copy */
  tmp = alloc_entry(entries);
  if (tmp == NULL)
    return NULL;

  /* copy data vector */
  for (i = 0; i < entries->dimension; i++)
    tmp->points[i] = data->points[i];

  /* copy labels */
  copy_entry_labels(tmp, data);

  /* copy mask */
  if (data->mask)
    {
      if ((tmp->mask = (char*)malloc(entries->dimension)) == NULL)
	{
	  fprintf(stderr, "Can't allocate memory for mask\n");
	  free_entry(tmp);
	  ERROR(ERR_NOMEM);
	  return NULL;
	}
      memcpy(tmp->mask, data->mask, entries->dimension);
    }

  /* copy other stuff */
  tmp->weight = data->weight;

  /* copy fixed point */
  if (data->fixed)
    {
      if ((tmp->fixed = (struct fixpoint*)malloc(sizeof(struct fixpoint))) == NULL)
	{
	  free_entry(tmp);
	  ERROR(ERR_NOMEM);
	  return NULL;
	}
      memcpy(tmp->fixed, data->fixed, sizeof(struct fixpoint));
    }
  tmp->next = NULL;

  return(tmp);
}


/* some routines for getting values from datafile headers */

int get_weight(char *str)
{
  return(atoi(&(str[7])));
}

struct fixpoint *get_fixed(char *str)
{
  char *dup, *sav;
  struct fixpoint *tmp;

  tmp = (struct fixpoint*)malloc(sizeof(struct fixpoint));
  if (tmp == NULL)
    return NULL;

  dup = ostrdup(str);
  if (dup == NULL)
    {
      free(tmp);
      return NULL;
    }

  tmp->xfix = atoi(&(dup[6]));
  sav = strchr(dup, ',');
  tmp->yfix = atoi(&(sav[1]));

  free(dup);


  if ((tmp->xfix < 0) || (tmp->yfix < 0)) {
    fprintf(stderr, "Fixed point incorrect");
    free(tmp);
    tmp = NULL;
  }

  return(tmp);
}

int get_topol(char *str)
{
  int ret = TOPOL_UNKNOWN;
  char *dup;
  char *tok;

  dup = ostrdup(str);
  strtok(dup, " ");
  tok = strtok(NULL, " ");

  if (tok != NULL) {
    ret = topol_type(tok);
  }

  ofree(dup);
  return(ret);
}

int get_neigh(char *str)
{
  int ret = NEIGH_UNKNOWN;
  char *dup;
  char *tok;

  dup = ostrdup(str);
  strtok(dup, " ");
  strtok(NULL, " ");
  strtok(NULL, " ");
  strtok(NULL, " ");
  tok = strtok(NULL, " ");

  if (tok != NULL) {
    ret = neigh_type(tok);
  }

  ofree(dup);
  return(ret);
}

int get_xdim(char *str)
{
  int ret = 0;
  char *dup;
  char *tok;

  dup = ostrdup(str);
  strtok(dup, " ");
  strtok(NULL, " ");
  tok = strtok(NULL, " ");

  if (tok != NULL) {
    ret = atoi(tok);
  }

  ofree(dup);
  return(ret);
}

int get_ydim(char *str)
{
  int ret = 0;
  char *dup;
  char *tok;

  dup = ostrdup(str);
  strtok(dup, " ");
  strtok(NULL, " ");
  strtok(NULL, " ");
  tok = strtok(NULL, " ");

  if (tok != NULL) {
    ret = atoi(tok);
  }

  ofree(dup);
  return(ret);
}


/*******************************************************************
 * Routines to handle files that contain the learning rate values  *
 *******************************************************************/

int alpha_read(float *alpha, long noc, char *infile)
{
  long i;
  char basename[STR_LNG];
  FILE *fp;
  struct file_info *fi;

  strcpy(basename, infile);
  strtok(basename, ".");
  strcat(basename, ".lra");

  fi = open_file(basename, "r");
  if (fi == NULL) {
    ifverbose(1) {
      fprintf(stderr, "Can't open alpha file %s", basename);
    }
    return(0);
  }

  fp = fi2fp(fi);

  for (i = 0; i < noc; i++) {
    if (fscanf(fp, "%g\n", &(alpha[i])) < 0)
      return(0);
  }

  close_file(fi);

  return(1);
}

int alpha_write(float *alpha, long noc, char *outfile)
{
  long i;
  char basename[STR_LNG];
  FILE *fp;
  struct file_info *fi;

  strcpy(basename, outfile);
  strtok(basename, ".");
  strcat(basename, ".lra");

  fi = open_file(basename, "w+");
  if (fi == NULL) {
    fprintf(stderr, "Can't open alpha file %s for writing", basename);
    return 0;
  }

  fp = fi2fp(fi);

  for (i = 0; i < noc; i++) {
    fprintf(fp, "%g\n", alpha[i]);
  }

  close_file(fi);
  return 0;
}

void invalidate_alphafile(char *outfile)
{
  int i;
  char basename[STR_LNG];
  FILE *fp;

  strcpy(basename, outfile);
  strtok(basename, ".");
  strcat(basename, ".lra");

  fp = fopen(basename, "r");
  if (fp != NULL) {
    ifverbose(1)
      fprintf(stdout, "Removing the learning rate file %s\n", basename);
    fclose(fp);
    i = remove(basename);
    if (i) {
      fprintf(stderr, "Can not remove %s", basename);
    }
  }
}


/*******************************************************************
 * Other routines                                                  *
 *******************************************************************/

/* randomize_entry_order - arrange a list of entrys to random order. */

struct data_entry *randomize_entry_order(struct data_entry *entry)
{
  long i, nol;
  struct data_entry *temp, fake, newlist, *prev;

  if (entry == NULL)
    return(NULL);

  temp = entry;

  for (nol = 0; temp != NULL; temp = temp->next)
    nol++;

  fake.next = entry;
  newlist.next = NULL;
  prev = &newlist;

  for (;nol; nol--)
    {
      temp = &fake;
      for (i = orand() % nol; i > 0; i--)
	temp = temp->next;
      prev->next = temp->next;
      prev = prev->next;
      temp->next = temp->next->next;
      prev->next= NULL;
    }

  return(newlist.next);
}

/* set_buffer - sets the buffer size of an entries-file or turns on
   LOADMODE_ALL if buffer == 0. */

void set_buffer(struct entries *ent, long buffer)
{

  /* set buffered input */
  if (buffer > 0)
    {
      ent->flags.loadmode = LOADMODE_BUFFER;
      ent->buffer = buffer;
    }
  else
    ent->flags.loadmode = LOADMODE_ALL;
}



/* set_teach_params - sets values in teaching parameter structure based
   on values given in codebook and data files */

int set_teach_params(struct teach_params *params, struct entries *codes,
		     struct entries *data, long dbuffer)
{
  int error = 0;

  /* set data buffer size */
  if (data != NULL)
    set_buffer(data, dbuffer);

  /* Load all codes to memory. This should be on for codebook files */
  set_buffer(codes, 0);

  params->topol = codes->topol;
  params->mapdist = NULL;
  params->neigh = codes->neigh;
  params->neigh_adapt = NULL;

  /* these two might change when using a different variation, for
     example, dot product */

  params->winner = find_winner_euc;
  params->dist = vector_dist_euc;
  params->vector_adapt = adapt_vector;

  if (codes)
    params->codes = codes;
  if (data)
    params->data = data;
  params->snapshot = NULL;

  return error;
}

struct typelist topol_list[] = {
  {TOPOL_DATA, "data", NULL},
  {TOPOL_LVQ, "lvq", NULL},
  {TOPOL_HEXA, "hexa", NULL},
  {TOPOL_RECT, "rect", NULL},
  {TOPOL_UNKNOWN, NULL, NULL}};

/* neighborhood types */

struct typelist neigh_list[] = {
  {NEIGH_BUBBLE, "bubble", NULL},
  {NEIGH_GAUSSIAN, "gaussian", NULL},
  {NEIGH_UNKNOWN, NULL, NULL}};

/* snapshots */

struct typelist snapshot_list[] = {
  {SNAPSHOT_SAVEFILE, "file", NULL},
#ifndef NO_PIPED_COMMANDS
  {SNAPSHOT_EXEC_CMD, "command", NULL},
  {SNAPSHOT_EXEC_CMD_ASYNC, "command_async", NULL},
#endif /* NO_PIPED_COMMANDS */
  {SNAPSHOT_SAVEFILE, NULL, NULL}}; /* default */

int label_not_needed(int level)
{
  static int label_level = 0;

  if (level >= 0) {
    label_level = level;
  }

  return(label_level);
}
