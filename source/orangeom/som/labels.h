#ifndef _LVQ_LABELS_H
#define _LVQ_LABELS_H
/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  labels.h                                                            *
 *   - header file for label manipulation functions                     *
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

#define LABEL_EMPTY 0

struct hit_entry {
  struct hit_entry *next, *prev;
  long label; 
  long freq;    /* frequency of this label */
};

struct hitlist {
  struct hit_entry *head, *tail;
  long entries;  /* number of entries */
};

int find_conv_to_ind(char *str);
char *find_conv_to_lab(int ind);
int number_of_labels();
void free_labels();
int set_entry_label(struct data_entry *entry, int label);
int get_entry_labels(struct data_entry *entry, int i);
#define get_entry_label(e) get_entry_labels((e), 0)

void clear_entry_labels(struct data_entry *entry);
int add_entry_label(struct data_entry *entry, int label);
int copy_entry_labels(struct data_entry *dest, struct data_entry *source);

/* hitlists */
struct hitlist *new_hitlist(void);
void clear_hitlist(struct hitlist *hl);
void free_hitlist(struct hitlist *hl);
struct hit_entry *find_hit(struct hitlist *hl, long label);
int add_hit(struct hitlist *hl, long label);
void print_hitlist(struct hitlist *hl, FILE *fp);
int hitlist_label_freq(struct hitlist *hl, long label);


#endif /* _LVQ_LABELS_H */
