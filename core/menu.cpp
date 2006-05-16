/********************************************************************
*
*   Name:    		 module MENU
*
*   Description:      deals with menus
*
*********************************************************************/

#include <stdio.h>
#include <string.h>

#include "general.h"
#include "menu.h"

const int ScreenWidth = 80 ;


int textMenu(const char* Title, const Pchar Item[],int NoItems)
{
    int MaxLen = strlen(Title) ;
    int i, pos ;
    for( i = 0 ; i < NoItems ; i++)
       if ((pos = strlen(Item[i])) > MaxLen)
          MaxLen = pos ;

    int tablen = (ScreenWidth - MaxLen - 4) / 2 ;
    char *tab = new char[tablen + 1] ;
    for (i = 0 ; i < tablen ; i++)
       tab[i] = ' ' ;
    tab[i] = '\0' ;

    printf("\n%s%s\n%s",tab, Title, tab) ;
    pos = strlen(Title) ;
    for (i = 0 ; i < pos ; i++)
      printf("-") ;
    printf("\n") ;

    i = 0 ;
    while (i<NoItems)
    {
       if (i<9)
         printf(" ") ;
       printf("%s%d. %s\n", tab, (i+1), Item[i]) ;
       i++ ;
    }
    delete [] tab ;

    int choice, nread ;
    do {
       printf(">") ;
       fflush(stdout) ;
       nread = scanf("%d", &choice) ;
       if (nread == 0) 
		   scanf("%*s") ;
       else if (nread == EOF) 
         return -1 ;
    } while (choice<1 || choice > NoItems) ;

    return choice ;
}
