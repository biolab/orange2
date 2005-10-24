#ifndef SOMPAK_DATAFILE_H
#define SOMPAK_DATAFILE_H
/************************************************************************
 *                                                                      *
 *  Program packages 'lvq_pak' and 'som_pak' :                          *
 *                                                                      *
 *  datafile.h                                                          *
 *   - header file for datafile.c: prototypes for functions             *
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
#include "lvq_pak.h"
#include "errors.h"
#include "fileio.h"

/* in files giving an x instead of a value marks that component to be
   ignored in calculations */

#ifndef MASKED_VALUE
#define MASKED_VALUE "x"
#endif /* MASKED_VALUE */

/* characters that separate vector components on a line, typically
   whitespace characters */

#ifndef SEPARATOR_CHARS
/* space and tab */
#define SEPARATOR_CHARS " \t"
#endif /* SEPARATOR_CHARS */

extern char *masked_string;

struct entries *open_data_file(char *name);
int read_headers(struct entries *entries);
struct entries *read_entries(struct entries *entries);

int save_entries_wcomments(struct entries *codes, char *out_code_file, char *comments);
#define save_entries(codes,name) save_entries_wcomments((codes),(name),NULL)
int write_entry(struct file_info *, struct entries *, struct data_entry *);
int write_header(struct file_info *fi, struct entries *codes);

struct data_entry *init_entry(struct entries *entr, struct data_entry *entry);
#define alloc_entry(entr) init_entry((entr), NULL)
void free_entry(struct data_entry *entry);
struct data_entry *load_entry(struct entries *entr, struct data_entry *entry);
struct data_entry *next_entry(eptr *ptr);
struct data_entry *rewind_entries(struct entries *, eptr *);

void close_entries(struct entries *entries);
struct entries *open_entries(char *name);
int rewind_file(struct file_info *fi);
int skip_headers(struct file_info *fi);

struct entries *alloc_entries(void);
struct entries *copy_entries(struct entries *entr);

#define free_entries(e) close_entries(e)

struct data_entry *copy_entry(struct entries *entries, struct data_entry *data);
void free_entrys(struct data_entry *data);

int get_topol(char *);
int get_neigh(char *);
int get_xdim(char *);
int get_ydim(char *);

#define topol_type(s) get_id_by_str(topol_list, s)
#define neigh_type(s) get_id_by_str(neigh_list, s)

#define topol_str(i) get_str_by_id(topol_list, i)
#define neigh_str(i) get_str_by_id(neigh_list, i)

int get_weight(char *str);
int label_not_needed(int level);
int use_weights(int level);
int use_fixed(int level);

int alpha_read(float *alphas, long noc, char *infile);
int alpha_write(float *alphas, long noc, char *outfile);
void invalidate_alphafile(char *outfile);

/* struct entries *new_entry_order(struct entries *data); */
struct data_entry *randomize_entry_order(struct data_entry *entry);

void set_buffer(struct entries *ent, long buffer);
int set_teach_params(struct teach_params *params, struct entries *codes, struct entries *data, long dbuffer);
char *set_mask(char *mask, int dim, int n);

extern struct typelist topol_list[], neigh_list[], snapshot_list[];

#endif /* SOMPAK_FILEIO_H */
