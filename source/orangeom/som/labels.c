/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  labels.c                                                            *
 *   - routines for manipulating labels                                 *
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

#include <string.h>
#include "lvq_pak.h"
#include "labels.h"
#include "errors.h"

/******************************************************************* 
 * Conversion between labels and indices                           * 
 *******************************************************************/

#ifndef LABEL_ARRAY_SIZE
#define LABEL_ARRAY_SIZE 100
#endif /* LABEL_ARRAY_SIZE */

static char **labels = NULL;
static int num_labs = 0;
static int lab_array_size = 0;
  
/* enlarge_array - enlarges (or allocates) the array used to store 
   the labels. */

static int enlarge_array()
{
  char **labs = NULL;

  lab_array_size += LABEL_ARRAY_SIZE;
  labs = realloc(labels, sizeof(char *) * lab_array_size);
  
  if (labs == NULL)
    {
      fprintf(stderr, "Can't allocate memory for labeltable \n");
      return ERR_NOMEM;
    }
  labels = labs;
  return 0;
}

/* free_labels - Free all the memory used by label list */

void free_labels()
{
  if (labels)
    free(labels);

  labels = NULL;
  num_labs = lab_array_size = 0;
}

/* find_conv_to_ind - Give the corresponding index; if the label is
   not yet there, add it to table. Empty label is always 0 */

int find_conv_to_ind(char *lab)
{
  int i, label;
  char *tmp;

  /* no string == empty label */
  if (lab == NULL)
    return LABEL_EMPTY;

  /* empty string == empty label */
  if (lab[0] == '\0')
    return LABEL_EMPTY;

  /* check if the label is already in the table */
  label = -1;
  for (i = 0; i < num_labs; i++)
    if (strcmp(labels[i], lab) == 0)
      {
	label = i;
	break;
      }
  
  if (label < 0)
    {
      /* label not found in array. Add it. */
      label = num_labs;
      if (label >= lab_array_size)
	if (enlarge_array())
	  return -1;
      
      if ((tmp = ostrdup(lab)) == NULL)
	return -1;
      
      labels[label] = tmp;
      num_labs++;
    }

  return(label + 1);
}

/* find_conv_to_lab - Give the corresponding label; if the index is
   not yet there, return NULL */

char *find_conv_to_lab(int ind)
{
  
  if (ind == LABEL_EMPTY)
    return NULL;

  if ((ind > num_labs) || (ind < 0))
    return NULL;

  return labels[ind - 1];
}

/* number_of_labels - Give the number of entries in the label table
   (including the empty label *) */

int number_of_labels()
{
  return (num_labs + 1); /* number of labels in array + empty label */
}

/* ********** Routines for manipulating labels ******************** */

/* labes are stored in the following way: If there is only one label,
   store it in the labels pointer.  If there are multiple labels, the
   labels entry is a pointer to an array of labels. The num_labs field
   in the entry tells the number of labels in that entry */

/* set_entry_label - sets the label on one data entry. All previous 
   labels are discarded. */

int set_entry_label(struct data_entry *entry, int label)
{
  clear_entry_labels(entry); /* remove previous labels if any */
  if (label != LABEL_EMPTY)  /* empty label == no label */
    {
      entry->lab.label = label;
      entry->num_labs = 1;
    }
  return 0;
}

/* get_entry_labels - get i:th label from entry (i starts from 0). */

int get_entry_labels(struct data_entry *entry, int i)
{

  if ((entry->num_labs <= 1) && (i == 0))
    return entry->lab.label;
  
  if (i >= entry->num_labs)
    return LABEL_EMPTY;
  else
    return entry->lab.label_array[i];
}

/* clear_entry_label - remove all labels from entry. */

void clear_entry_labels(struct data_entry *entry)
{
  if (entry->num_labs > 1)
    free(entry->lab.label_array);

  entry->num_labs = 0;
  entry->lab.label = LABEL_EMPTY;
}

#define ATABLE_INCREMENT 8

/* add_entry_label - add a label to entry. Returns non-zero on error. */

int add_entry_label(struct data_entry *entry, int label)
{
  int *atable = NULL;

  clear_err();

  /* adding an empty label does nothing */
  if (label == LABEL_EMPTY)
    return 0;
  
  /* add first label to entry */
  if (entry->num_labs == 0)
    {
      entry->num_labs = 1;
      entry->lab.label = label;
      return 0;
    }

  /* if there is already one label, we need to allocate a table for the 
     labels */

  if (entry->num_labs == 1)
    {
      atable = malloc(sizeof(int) * ATABLE_INCREMENT);
      if (atable == NULL)
	return ERR_NOMEM;
      
      /* move old entry to table and add the new label */
      entry->num_labs++;
      atable[0] = entry->lab.label;
      atable[1] = label;
      entry->lab.label_array = atable;
      return 0;
    }

  atable = entry->lab.label_array;

  /* enlarge label array if needed */
  if ((entry->num_labs % ATABLE_INCREMENT) == 0)
    {
      /* need more space */
      atable = realloc(atable,
		       sizeof(int) * (entry->num_labs + ATABLE_INCREMENT));
      if (atable == NULL)
	return ERR_NOMEM;
      entry->lab.label_array = atable;
    }

  atable[entry->num_labs++] = label;
      
  return 0;
}

/* copy_entry_labels - copy all labels from one entry to the other. */

int copy_entry_labels(struct data_entry *dest, struct data_entry *source)
{
  int blocks, size, *atable;
  
  clear_err();

  clear_entry_labels(dest); /* remove old labels first */

  if (source->num_labs <= 1)
    dest->lab.label = source->lab.label;
  else
    {
      blocks = (source->num_labs + ATABLE_INCREMENT - 1) / ATABLE_INCREMENT;
      size = sizeof(int) * blocks * ATABLE_INCREMENT;
      atable = malloc(size);
      if (atable == NULL)
	return ERR_NOMEM;
      memcpy(atable, source->lab.label_array, size);
      dest->lab.label_array = atable;
    }
  dest->num_labs = source->num_labs;

  return 0;
}

/* ********************************************************************** */

/* hitlists - hitlist is a list of label,frequency pairs that is kept sorted 
   so that the label with the highest frequency is always the first in the
   list. */

/* initialize a new hitlist */

struct hitlist *new_hitlist(void)
{
  struct hitlist *hl;

  clear_err();

  hl = malloc(sizeof(struct hitlist));
  if (hl == NULL)
    {
      ERROR(ERR_NOMEM);
      return NULL;
    }

  hl->head = hl->tail = NULL;
  hl->entries = 0;
  
  return hl;
}

/* clear_hitlist - clears all entries from a hitlist */

void clear_hitlist(struct hitlist *hl)
{
  struct hit_entry *he, *next;
  if (hl)
    {
      for (he = hl->head; he; he = next)
	{
	  next = he->next;
	  free(he);
	}

      hl->head = hl->tail = NULL;
    }
}

/* free_hitlist - deallocate a hitlist */

void free_hitlist(struct hitlist *hl)
{
  if (hl)
    {
      clear_hitlist(hl); /* deallocate hits */
      free(hl);
    }
}

/* find_hit - find a hit_entry corresponding to a certain label from
   list. Returns NULL if there is no entry for the label */

struct hit_entry *find_hit(struct hitlist *hl, long label)
{
  struct hit_entry *he;

  for(he = hl->head; he; he = he->next)
    if (label == he->label)
      break;

  return he;
}

/* hit_swapwprev - (internal) exchange a hit entry with the previous
   hit in the list. Used to keep the list in order */

static struct hit_entry *hit_swapwprev(struct hitlist *hl, struct hit_entry *he)
{
  struct hit_entry *prev;
  
  prev = he->prev;

  if (prev)
    {
      prev->next = he->next;
      he->prev = prev->prev;
      he->next = prev;
      prev->prev = he;
      if (prev->next == NULL)
	hl->tail = prev;
      else
	prev->next->prev = prev;
      if (he->prev == NULL)
	hl->head = he;
      else 
	he->prev->next = he;
    }

  return he;
}

/* add_hit - add a hit in the list for a label */

int add_hit(struct hitlist *hl, long label)
{
  struct hit_entry *he;

  he = find_hit(hl, label);
  
  if (he)
    {
      /* found in list, increase counter */
      he->freq++;

      /* keep list in order, higher frequencies are in the beginning. */
      while (he->prev)
	if (he->prev->freq < he->freq)
	  he = hit_swapwprev(hl, he);
	else
	  break;
    }
  else
    {
      he = malloc(sizeof(struct hit_entry));
      if (he == NULL)
	{
	  ERROR(ERR_NOMEM);
	  return 0;
	}
      /* add to end of list */
      he->next = NULL;
      he->prev = hl->tail;
      he->label = label;
      he->freq = 1;
      hl->tail = he;
      if (he->prev)
	he->prev->next = he;
      else
	hl->head = he;
      hl->entries++;
    }

  return he->freq;
}

/* print_hitlist - prints the contents of a hitlist */

void print_hitlist(struct hitlist *hl, FILE *fp)
{
  struct hit_entry *he;
  int lab;

  for (he = hl->head; he; he = he->next)
    {
      lab = he->label;
      fprintf(fp, "%s,%d", 
	      (lab != LABEL_EMPTY) ? find_conv_to_lab(lab) : "EMPTY",
	      he->freq);
      if (he->next)
	fputs(", ", fp);
    }
  fputs("\n", fp);
}

/* hitlist_label_freq - returns the counter value for a label. If the
   label is not in the list returns 0 (naturally) */

int hitlist_label_freq(struct hitlist *hl, long label)
{
  struct hit_entry *he;

  for (he = hl->head; he != NULL; he = he->next)
    if (he->label == label)
      return he->freq;
  
  return 0;
}

