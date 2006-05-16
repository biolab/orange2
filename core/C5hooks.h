#if !defined(READC5_H)
#define READC5_H


typedef  unsigned char	Boolean ;

void GetNames(FILE *Nf) ;
void GetData(FILE *Df, Boolean Train) ;
void GetMCosts(FILE *Cf) ;
void FreeC5(void) ;

#endif
