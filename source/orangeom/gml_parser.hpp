#ifndef gml_parser_h
#define gml_parser_h

#include "gml_scanner.hpp"

union GML_pair_val {
    long integer;
    double floating;
    char* string;
    struct GML_pair* list;
};

struct GML_pair {
    char* key;
    GML_value kind;
    union GML_pair_val value;
    struct GML_pair* next;
};

struct GML_list_elem {
    char* key;
    struct GML_list_elem* next;
};

struct GML_stat {
    struct GML_error err;
    struct GML_list_elem* key_list;
};

/*
 * returns list of KEY - VALUE pairs. Errors and a pointer to a list
 * of key-names are returned in GML_stat. Previous information contained
 * in GML_stat, i.e. the key_list, will be *lost*. 
 */

struct GML_pair* GML_parser (FILE*, struct GML_stat*, int);

/*
 * free memory used in a list of GML_pair
 */

void GML_free_list (struct GML_pair*, struct GML_list_elem*);


/*
 * debugging 
 */

void GML_print_list (struct GML_pair*, int);

#endif




