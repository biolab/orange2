

/*************************************************************************/
/*									 */
/*	Source code for use with See5/C5.0 Release 1.15			 */
/*	-----------------------------------------------			 */
/*		      Copyright RuleQuest Research 2001			 */
/*									 */
/*	This code is provided "as is" without warranty of any kind,	 */
/*	either express or implied.  All use is at your own risk.	 */
/*									 */
/*************************************************************************/


#include "general.h"
#include "C5defns.h"

#define Inc 2048


/*************************************************************************/
/*									 */
/*		General data for See5/C5.0				 */
/*									 */
/*************************************************************************/


	Attribute	ClassAtt=0,	/* attribute to use as class */
			LabelAtt=0;	/* attribute used as case label */
	int		MaxAtt=0,	/* max att number */
			MaxClass=0,	/* max class number */
			ErrMsgs=0,      /* errors found */
			AttExIn=0,	/* attribute exclusions/inclusions */
			TSBase=0,	/* base day for time stamps */
			LineNo=0,	/* input line number */
			Delimiter;	/* character at end of name */
	DiscrValue	*MaxAttVal=0;	/* number of values for each att */
	char		*SpecialStatus=0;	/* special att treatment */
	Definition	*AttDef=0;	/* attribute definitions */
	String		*ClassName=0,	/* class names */
		  	*AttName=0,	/* att names */
		  	**AttValName=0;	/* att value names */
	double		Confidence,	/* set by classify() */
			*Vote=0,	/* total votes for classes */
			**MCost=0;	/* misclass cost [pred][real] */
	Tree		*Pruned=0;	/* decision trees */
	CRuleSet	*RuleSet=0;	/* rulesets */
	FILE		*TRf=0;		/* file pointer for tree and rule i/o */
	char		*FileStem="?",	/* file stem */
			Fn[500];	/* file name with extension */
	Boolean		*SomeMiss=Nil,	/* att has missing values */
			*SomeNA=Nil;	/* att has N/A values */

	ItemNo		MaxItem=-1;	/* max data item number */

	Description	*Item;		/* data items */

/*************************************************************************/
/*									 */
/*		Global parameters for See5/C5.0				 */
/*									 */
/*************************************************************************/


//	int		TRIALS=1;	/* number of classifiers */

//	Boolean		RULES=0,	/* true if use rulesets */
//			PROBTHRESH=0;	/* true if to use soft thresholds */


/*************************************************************************/
/*									 */
/*	  Global data for See5/C5.0 used for building decision trees	 */
/*									 */
/*************************************************************************/


// int	Trial;		/* trial number for boosting */

// double	*ClassSum=0;	/* class weights during classification */


/*************************************************************************/
/*									 */
/*	Global data for constructing and applying rules			 */
/*									 */
/*************************************************************************/


// CRule	*Rule=0;		/* current rules */

// RuleNo	NRules;		/* number of rules */

// int	RuleSpace;	/* space currently allocated for rules */

// ClassNo	Default;	/* default class associated with ruleset */


/*************************************************************************/
/*									 */
/*	Read a name from file f into string s, setting Delimiter.	 */
/*									 */
/*	- Embedded periods are permitted, but periods followed by space	 */
/*	  characters act as delimiters.					 */
/*	- Embedded spaces are permitted, but multiple spaces are	 */
/*	  replaced by a single space.					 */
/*	- Any character can be escaped by '\'.				 */
/*	- The remainder of a line following '|' is ignored.		 */
/*									 */
/*************************************************************************/


Boolean ReadName(FILE *f, String s, int n, char ColonOpt)
/*      --------  */
{
    register char *Sp=s;
    register int  c;
    char	  Msg[2];

    /*  Skip to first non-space character  */

    while ( (c = InChar(f)) == '|' || Space(c) )
    {
	if ( c == '|' ) SkipComment;
    }

    /*  Return false if no names to read  */

    if ( c == EOF )
    {
	Delimiter = EOF;
	return false;
    }

    /*  Read in characters up to the next delimiter  */

    while ( c != ColonOpt && c != ',' && c != '\n' && c != '|' && c != EOF )
    {
	if ( --n <= 0 )
	{
	    if ( Of ) Error(LONGNAME, "", "");
	}

	if ( c == '.' )
	{
	    if ( (c = InChar(f)) == '|' || Space(c) || c == EOF ) break;
	    *Sp++ = '.';
	    continue;
	}

	if ( c == '\\' )
	{
	    c = InChar(f);
	}

	if ( Space(c) )
	{
	    *Sp++ = ' ';

	    while ( ( c = InChar(f) ) == ' ' || c == '\t' )
		;
	}
	else
	{
	    *Sp++ = c;
	    c = InChar(f);
	}
    }

    if ( c == '|' ) SkipComment;
    Delimiter = c;

    /*  Special case for ':='  */

    if ( Delimiter == ':' )
    {
	c = InChar(f);
	if ( c == '=' )
	{
	    Delimiter = c;
	}
	else
	{
	    ungetc(c, f);
	}
    }

    /*  Strip trailing spaces  */

    while ( Sp > s && Space(*(Sp-1)) ) Sp--;

    if ( Sp == s )
    {
	Msg[0] = ( Space(c) ? '.' : c );
	Msg[1] = '\00';
	Error(MISSNAME, Fn, Msg);
    }

    *Sp++ = '\0';
    return true;
}



/*************************************************************************/
/*									 */
/*	Read names of classes, attributes and legal attribute values.	 */
/*	On completion, names are stored in:				 */
/*	  ClassName	-	class names				 */
/*	  AttName	-	attribute names				 */
/*	  AttValName	-	attribute value names			 */
/*	with:								 */
/*	  MaxAttVal	-	number of values for each attribute	 */
/*									 */
/*	Other global variables set are:					 */
/*	  MaxAtt	-	maximum attribute number		 */
/*	  MaxClass	-	maximum class number			 */
/*									 */
/*	Note:  until the number of attributes is known, the name	 */
/*	       information is assembled in local arrays			 */
/*									 */
/*************************************************************************/


void GetNames(FILE *Nf)
/*   --------  */
{
    char	Buffer[1000]="";
    int		AttCeiling=100, ClassCeiling=100;
    Attribute	Att;

    ErrMsgs = AttExIn = 0;
    LineNo  = 1;

    /*  Get class names from names file  */

    ClassName = AllocZero(ClassCeiling, String);
    MaxClass = ClassAtt = LabelAtt = 0;
    do
    {
	ReadName(Nf, Buffer, 1000, ':');

	if ( ++MaxClass >= ClassCeiling)
	{
	    ClassCeiling += 100;
	    Realloc(ClassName, ClassCeiling, String);
	}
	ClassName[MaxClass] = CopyString(Buffer);
    }
    while ( Delimiter == ',' );

    /*  Get attribute and attribute value names from names file  */

    AttName = AllocZero(AttCeiling, String);
    MaxAttVal = AllocZero(AttCeiling, DiscrValue);
    AttValName = AllocZero(AttCeiling, String *);
    SpecialStatus = AllocZero(AttCeiling, char);
    AttDef = AllocZero(AttCeiling, Definition);

    MaxAtt = 0;
    while ( ReadName(Nf, Buffer, 1000, ':') )
    {
	if ( Delimiter != ':' && Delimiter != '=' )
	{
	    Error(BADATTNAME, Buffer, "");
	}

	/*  Check for include/exclude instruction  */

	if ( ( *Buffer == 'a' || *Buffer == 'A' ) &&
	     ! memcmp(Buffer+1, "ttributes ", 10) &&
	     ! memcmp(Buffer+strlen(Buffer)-6, "cluded", 6) )
	{
	    AttExIn = ( ! memcmp(Buffer+strlen(Buffer)-8, "in", 2) ? 1 : -1 );
	    if ( AttExIn == 1 )
	    {
		ForEach(Att, 1, MaxAtt)
		{
		    SpecialStatus[Att] |= SKIP;
		}
	    }

	    while ( ReadName(Nf, Buffer, 1000, ':') )
	    {
		Att = Which(Buffer, AttName, 1, MaxAtt);
		if ( ! Att )
		{
		    Error(UNKNOWNATT, Buffer, Nil);
		}
		else
		if ( AttExIn == 1 )
		{
		    SpecialStatus[Att] -= SKIP;
		}
		else
		{
		    SpecialStatus[Att] |= SKIP;
		}
	    }

	    break;
	}

	if ( Which(Buffer, AttName, 1, MaxAtt) > 0 )
	{
	    Error(DUPATTNAME, Buffer, Nil);
	}

	if ( ++MaxAtt >= AttCeiling )
	{
	    AttCeiling += 100;
	    Realloc(AttName, AttCeiling, String);
	    Realloc(MaxAttVal, AttCeiling, DiscrValue);
	    Realloc(AttValName, AttCeiling, String *);
	    Realloc(SpecialStatus, AttCeiling, char);
	    Realloc(AttDef, AttCeiling, Definition);
	}

	AttName[MaxAtt] = CopyString(Buffer);
	SpecialStatus[MaxAtt] = Nil;
	AttDef[MaxAtt] = Nil;
	MaxAttVal[MaxAtt] = 0;

	if ( Delimiter == '=' )
	{
	    if ( MaxClass == 1 && ! strcmp(ClassName[1], AttName[MaxAtt]) )
	    {
		Error(BADDEF3, Nil, Nil);
	    }

	    ImplicitAtt(Nf);
	}
	else
	{
	    ExplicitAtt(Nf);
	}
    }

    /*  Check whether class is one of the attributes  */

    if ( MaxClass == 1 )
    {
	ClassAtt = Which(ClassName[1], AttName, 1, MaxAtt);

	/*  Make sure not excluding class attribute  */

	if ( Skip(ClassAtt) ) SpecialStatus[ClassAtt] -= SKIP;

	/*  Class attribute must be defined and must be discrete  */

	if ( ClassAtt <= 0 )
	{
	    Error(NOTARGET, ClassName[1], "");
	}
	else
	if ( Continuous(ClassAtt) ||
	     StatBit(ClassAtt, DISCRETE|EXCLUDE) )
	{
	    Error(BADTARGET, ClassName[1], "");
	}

	free(ClassName[1]);
	free(ClassName);

	MaxClass  = MaxAttVal[ClassAtt];
	ClassName = AttValName[ClassAtt];
    }
    else
	   ClassName[0] = CopyString("?");

    fclose(Nf);

    if ( ErrMsgs > 0 ) Goodbye(1);
}



/*************************************************************************/
/*									 */
/*	Continuous or discrete attribute				 */
/*									 */
/*************************************************************************/


void ExplicitAtt(FILE *Nf)
/*   -----------  */
{
    char	Buffer[1000]="", *p;
    DiscrValue	v;
    int		ValCeiling=100, BaseYear;
    time_t	clock;

    /*  Read attribute type or first discrete value  */

    if ( ! ( ReadName(Nf, Buffer, 1000, ':') ) )
    {
	Error(EOFINATT, AttName[MaxAtt], "");
    }

    MaxAttVal[MaxAtt] = 0;

    if ( Delimiter != ',' )
    {
	/*  Typed attribute  */

	if ( ! strcmp(Buffer, "continuous") )
	{
	}
	else
	if ( ! strcmp(Buffer, "timestamp") )
	{
	    SpecialStatus[MaxAtt] = TSTMPVAL;

	    /*  Set the base date if not done already  */

	    if ( ! TSBase )
	    {
		clock = time(0);
		BaseYear = gmtime(&clock)->tm_year + 1900;
		SetTSBase(BaseYear);
	    }
	}
	else
	if ( ! strcmp(Buffer, "date") )
	{
	    SpecialStatus[MaxAtt] = DATEVAL;
	}
	else
	if ( ! strcmp(Buffer, "time") )
	{
	    SpecialStatus[MaxAtt] = STIMEVAL;
	}
	else
	if ( ! memcmp(Buffer, "discrete", 8) )
	{
	    SpecialStatus[MaxAtt] = DISCRETE;

	    /*  Read max values and reserve space  */

	    v = atoi(&Buffer[8]);
	    if ( v < 2 )
	    {
		Error(BADDISCRETE, AttName[MaxAtt], "");
	    }

	    AttValName[MaxAtt] = Alloc(v+3, String);
	    AttValName[MaxAtt][0] = (char *) (int) v+1;
	    AttValName[MaxAtt][(MaxAttVal[MaxAtt]=1)] = CopyString("N/A");
	}
	else
	if ( ! strcmp(Buffer, "ignore") )
	{
	    SpecialStatus[MaxAtt] = EXCLUDE;
	}
	else
	if ( ! strcmp(Buffer, "label") )
	{
	    LabelAtt = MaxAtt;
	    SpecialStatus[MaxAtt] = EXCLUDE;
	}
	else
	{
	    /*  Cannot have only one discrete value for an attribute  */

	    Error(SINGLEATTVAL, AttName[MaxAtt], Buffer);
	}
    }
    else
    {
	/*  Discrete attribute with explicit values  */

	AttValName[MaxAtt] = AllocZero(ValCeiling, String);

	/*  Add "N/A" unless this attribute is the class  */

	if ( MaxClass > 1 || strcmp(ClassName[1], AttName[MaxAtt]) )
	{
	    AttValName[MaxAtt][(MaxAttVal[MaxAtt]=1)] = CopyString("N/A");
	}
	else
	{
	    MaxAttVal[MaxAtt] = 0;
	}

	p = Buffer;

	/*  Special check for ordered attribute  */

	if ( ! memcmp(Buffer, "[ordered]", 9) )
	{
	    SpecialStatus[MaxAtt] = ORDERED;

	    for ( p = Buffer+9 ; Space(*p) ; p++ )
		;
	}

	/*  Record first real explicit value  */

	AttValName[MaxAtt][++MaxAttVal[MaxAtt]] = CopyString(p);

	/*  Record remaining values  */

	do
	{
	    if ( ! ( ReadName(Nf, Buffer, 1000, ':') ) )
	    {
		Error(EOFINATT, AttName[MaxAtt], "");
	    }

	    if ( ++MaxAttVal[MaxAtt] >= ValCeiling )
	    {
		ValCeiling += 100;
		Realloc(AttValName[MaxAtt], ValCeiling, String);
	    }

	    AttValName[MaxAtt][MaxAttVal[MaxAtt]] = CopyString(Buffer);
	}
	while ( Delimiter == ',' );

	/*  Cancel ordered status if <3 real values  */

	if ( Ordered(MaxAtt) && MaxAttVal[MaxAtt] <= 3 )
	{
	    SpecialStatus[MaxAtt] = 0;
	}
    }
}



/*************************************************************************/
/*									 */
/*	Locate value Val in List[First] to List[Last]			 */
/*									 */
/*************************************************************************/


int Which(String Val, String *List, int First, int Last)
/*  -----  */
{
    int	n=First;

    while ( n <= Last && strcmp(Val, List[n]) ) n++;

    return ( n <= Last ? n : First-1 );
}



/*************************************************************************/
/*									 */
/*	Allocate space then copy string into it				 */
/*									 */
/*************************************************************************/


String CopyString(String S)
/*     ----------  */
{
    char *p;

    p = (char*)malloc(strlen(S)+1);
    strcpy(p, S);
    return p;
}



/*************************************************************************/
/*									 */
/*	Read next char keeping track of line numbers			 */
/*									 */
/*************************************************************************/


int InChar(FILE *f)
/*  ------  */
{
    static Boolean	NewLine=false;
    register int	c;

    if ( NewLine ) LineNo++;

    NewLine = ( (c = getc(f)) == '\n' );

    return c;
}



/*************************************************************************/
/*									 */
/*  Read a raw case description from file Df.				 */
/*									 */
/*  For each attribute, read the attribute value from the file.		 */
/*  If it is a discrete valued attribute, find the associated no.	 */
/*  of this attribute value (if the value is unknown this is 0).	 */
/*									 */
/*  Returns the Description of the case (i.e. the array of		 */
/*  attribute values).							 */
/*									 */
/*************************************************************************/


#define XError(a,b,c)	Error(a,b,c)


Description GetDescription(FILE *Df, Boolean Train)
/*          --------------  */
{
    Attribute	Att;
    char	name[1000], *endname;
    int		Dv;
    // Description	Dummy;
	Description DVec;
    ContValue	Cv;
    Boolean	FirstValue=true;


    if ( ReadName(Df, name, 1000, '\00') )
    {
	DVec = AllocZero(MaxAtt+2, AttValue);
	//Dummy = AllocZero(MaxAtt+2, AttValue);
	//DVec = &Dummy[1];
	ForEach(Att, 1, MaxAtt)
	{
	    if ( AttDef[Att] )
	    {
		DVec[Att] = EvaluateDef(AttDef[Att], DVec);
		if ( SomeMiss )
		{
		    SomeMiss[Att] |= Unknown(DVec, Att);
		    SomeNA[Att]   |= NotApplic(DVec, Att);
		}
		continue;
	    }

	    /*  Get the attribute value if don't already have it  */

	    if ( ! FirstValue && ! ReadName(Df, name, 1000, '\00') )
	    {
		XError(HITEOF, Fn, "");
		FreeLastCase(DVec);
		return Nil;
	    }
	    FirstValue = false;

	    if ( Exclude(Att) )
	    {
		/*  Record the value as a string  */

		SVal(DVec,Att) = Alloc(strlen(name)+1, char);
		strcpy(SVal(DVec,Att), name);
	    }
	    else
	    if ( ! strcmp(name, "?") )
	    {
		/*  Set marker to indicate missing value  */

		DVal(DVec, Att) = UNKNOWN;
		if ( SomeMiss ) SomeMiss[Att] = true;
	    }
	    else
	    if ( Att != ClassAtt && ! strcmp(name, "N/A") )
	    {
		/*  Set marker to indicate not applicable  */

		DVal(DVec, Att) = NA;
		if ( SomeNA ) SomeNA[Att] = true;
	    }
	    else
	    if ( Discrete(Att) )
	    {
		/*  Discrete attribute  */

		Dv = Which(name, AttValName[Att], 1, MaxAttVal[Att]);
		if ( ! Dv )
		{
		    if ( StatBit(Att, DISCRETE) )
		    {
			if ( Train )
			{
			    /*  Add value to list  */

			    Dv = ++MaxAttVal[Att];
			    if ( Dv > (int) AttValName[Att][0] )
			    {
				XError(TOOMANYVALS, AttName[Att],
					 (char *) AttValName[Att][0]);
			    }
			    AttValName[Att][Dv]   = CopyString(name);
			    AttValName[Att][Dv+1] = "<other>"; /* no free */
			}
			else
			{
			    /*  Set value to "<other>"  */

			    Dv = MaxAttVal[Att] + 1;
			}
		    }
		    else
		    {
			XError(BADATTVAL, AttName[Att], name);
			Dv = UNKNOWN;
		    }
		}
		DVal(DVec, Att) = Dv;
	    }
	    else
	    {
		/*  Continuous value  */

		if ( TStampVal(Att) )
		{
		    CVal(DVec, Att) = Cv = TStampToMins(name);
		    if ( Cv >= 1E9 )	/* long time in future */
		    {
			XError(BADTSTMP, AttName[Att], name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}
		else
		if ( DateVal(Att) )
		{
		    CVal(DVec, Att) = Cv = DateToDay(name);
		    if ( Cv < 1 )
		    {
			XError(BADDATE, AttName[Att], name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}
		else
		if ( TimeVal(Att) )
		{
		    CVal(DVec, Att) = Cv = TimeToSecs(name);
		    if ( Cv < 0 )
		    {
			XError(BADTIME, AttName[Att], name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}
		else
		{
		    CVal(DVec, Att) = strtod(name, &endname);
		    if ( endname == name || *endname != '\0' )
		    {
			XError(BADATTVAL, AttName[Att], name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}
	    }
	}

	if ( ClassAtt )
	{
	    Class(DVec) = XDVal(DVec, ClassAtt);
	}
	else
	{
	    if ( ! ReadName(Df, name, 1000, '\00') )
	    {
		XError(HITEOF, Fn, "");
		FreeLastCase(DVec);
		return Nil;
	    }

	    Class(DVec) = Dv = Which(name, ClassName, 1, MaxClass);
	}


	return DVec;
    }
    else
    {
	return Nil;
    }
}



/*************************************************************************/
/*									 */
/*	Routines to handle implicitly-defined attributes		 */
/*									 */
/*************************************************************************/


char	*Buff;			/* buffer for input characters */
int	BuffSize, BN;		/* size and index of next character */

EltRec	*TStack;		/* expression stack model */
int	TStackSize, TSN;	/* size of stack and index of next entry */

int	DefSize, DN;		/* size of definition and next element */

Boolean PreviousError;		/* to avoid parasytic errors */

AttValue _UNK,			/* quasi-constant for unknown value */
	 _NA;			/* ditto for not applicable */


#define FailSyn(Msg)	 {DefSyntaxError(Msg); return false;}
#define FailSem(Msg)	 {DefSemanticsError(Fi, Msg, OpCode); return false;}

#define	cval		_cont_val
#define	sval		_string_val
#define	dval		_discr_val



/*************************************************************************/
/*									 */
/*	A definition is handled in two stages:				 */
/*	  - The definition is read (up to a line ending with a period)	 */
/*	    replacing multiple whitespace characters with one space	 */
/*	  - The definition is then read (using a recursive descent	 */
/*	    parser), building up a reverse polish expression		 */
/*	Syntax and semantics errors are flagged				 */
/*									 */
/*************************************************************************/


void ImplicitAtt(FILE *Nf)
/*   -----------  */
{
#ifdef CUBIST
    _UNK.cval = UNKNOWN;
    _UNK.dval = 0;
    _NA.dval  = NA;
    memcpy(&_NA.cval, &_NA.dval, sizeof(int));
#else
    _UNK.dval = UNKNOWN;
    _NA.dval  = NA;
#endif

    /*  Get definition as a string in Buff  */

    ReadDefinition(Nf);

    PreviousError = false;
    BN = 0;

    /*  Allocate initial stack and attribute definition  */

    TStack = Alloc(TStackSize=50, EltRec);
    TSN = 0;

    AttDef[MaxAtt] = Alloc(DefSize = 100, DefElt);
    DN = 0;

    /*  Parse Buff as an expression terminated by a period  */

    Expression();
    if ( ! Find(".") ) DefSyntaxError("'.' ending definition");

    /*  Final check -- defined attribute must not be of type String  */

    if ( ! PreviousError )
    {
	if ( DN == 1 )
	{
	    Error(SAMEATT, AttName[ (int) DefSVal(AttDef[MaxAtt][0]) ], Nil);
	    PreviousError = true;
	}

	if ( TStack[0].Type == 'B' )
	{
	    /*  Defined attributes should never have a value N/A  */

	    MaxAttVal[MaxAtt] = 3;
	    AttValName[MaxAtt] = AllocZero(4, String);
	    AttValName[MaxAtt][1] = CopyString("??");
	    AttValName[MaxAtt][2] = CopyString("t");
	    AttValName[MaxAtt][3] = CopyString("f");
	}
	else
	{
	    MaxAttVal[MaxAtt] = 0;
	}
    }

    if ( PreviousError )
    {
	DN = 0;
	SpecialStatus[MaxAtt] = EXCLUDE;
    }

    /*  Write a terminating marker  */

    DefOp(AttDef[MaxAtt][DN]) = OP_END;

    free(Buff);
    free(TStack);
}



/*************************************************************************/
/*									 */
/*	Read the text of a definition.  Skip comments, collapse		 */
/*	multiple whitespace characters.					 */
/*									 */
/*************************************************************************/


void ReadDefinition(FILE *f)
/*   --------------  */
{
    Boolean	LastWasPeriod=false;
    char	c;

    Buff = Alloc(BuffSize=50, char);
    BN = 0;

    while ( true )
    {
	c = InChar(f);

	if ( c == '|' ) SkipComment;

	if ( c == EOF || c == '\n' && LastWasPeriod )
	{
	    /*  The definition is complete.  Add a period if it's
		not there already and terminate the string  */

	    if ( ! LastWasPeriod ) Append('.');
	    Append(0);

	    return;
	}

	if ( Space(c) )
	{
	    Append(' ');
	}
	else
	if ( c == '\\' )
	{
	    /*  Escaped character -- bypass any special meaning  */

	    Append(InChar(f));
	}
	else
	{
	    LastWasPeriod = ( c == '.' );
	    Append(c);
	}
    }
}



/*************************************************************************/
/*									 */
/*	Append a character to Buff, resizing it if necessary		 */
/*									 */
/*************************************************************************/


void Append(char c)
/*   ------  */
{
    if ( c == ' ' && (! BN || Buff[BN-1] == ' ' ) ) return;

    if ( BN >= BuffSize )
    {
	Realloc(Buff, BuffSize += 50, char);
    }

    Buff[BN++] = c;
}



/*************************************************************************/
/*									 */
/*	Recursive descent parser with syntax error checking.		 */
/*	The reverse polish is built up by calls to Dump() and DumpOp(),	 */
/*	which also check for semantic validity.				 */
/*									 */
/*	For possible error messages, each routine also keeps track of	 */
/*	the beginning of the construct that it recognises (in Fi).	 */
/*									 */
/*************************************************************************/


Boolean Expression()
/*      ----------  */
{
    int		Fi=BN;

    if ( Buff[BN] == ' ' ) BN++;

    if ( ! Conjunct() ) FailSyn("expression");

    while ( Find("or") )
    {
	BN += 2;

	if ( ! Conjunct() ) FailSyn("expression");

	DumpOp(OP_OR, Fi);
    }

    return true;
}



Boolean Conjunct()
/*      --------  */
{
    int		Fi=BN;

    if ( ! SExpression() ) FailSyn("expression");

    while ( Find("and") )
    {
	BN += 3;

	if ( ! SExpression() ) FailSyn("expression");

	DumpOp(OP_AND, Fi);
    }

    return true;
}



String RelOps[] = {">=", "<=", "!=", ">", "<", "=", "<>", (String) 0};

Boolean SExpression()
/*      -----------  */
{
    int		o, Fi=BN;

    if ( ! AExpression() ) FailSyn("expression");

    if ( (o = FindOne(RelOps)) >= 0 )
    {
	BN += strlen(RelOps[o]);

	if ( ! AExpression() ) FailSyn("expression");

	DumpOp(( o == 0 ? OP_GE :
		 o == 1 ? OP_LE :
		 o == 4 ? OP_GT :
		 o == 5 ? OP_LT :
		 o == 2 || o == 3 ?
			( TStack[TSN-1].Type == 'S' ? OP_SNE : OP_NE ) :
			( TStack[TSN-1].Type == 'S' ? OP_SEQ : OP_EQ ) ), Fi);
    }

    return true;
}



String AddOps[] = {"+", "-", (String) 0};

Boolean AExpression()
/*      -----------  */
{
    int		o, Fi=BN;

    if ( Buff[BN] == ' ' ) BN++;

    if ( (o = FindOne(AddOps)) >= 0 )
    {
	BN += 1;
    }

    if ( ! Term() ) FailSyn("expression");

    if ( o == 1 ) DumpOp(OP_UMINUS, Fi);

    while ( (o = FindOne(AddOps)) >= 0 )
    {
	BN += 1;

	if ( ! Term() ) FailSyn("arithmetic expression");

	DumpOp((char)(OP_PLUS + o), Fi);
    }

    return true;
}



String MultOps[] = {"*", "/", "%", (String) 0};

Boolean Term()
/*      ----  */
{
    int		o, Fi=BN;

    if ( ! Factor() ) FailSyn("expression");

    while ( (o = FindOne(MultOps)) >= 0 )
    {
	BN += 1;

	if ( ! Factor() ) FailSyn("arithmetic expression");

	DumpOp((char)(OP_MULT + o), Fi);
    }

    return true;
}



Boolean Factor()
/*      ----  */
{
    int		Fi=BN;

    if ( ! Primary() ) FailSyn("value");

    while ( Find("^") )
    {
	BN += 1;

	if ( ! Primary() ) FailSyn("exponent");

	DumpOp(OP_POW, Fi);
    }

    return true;
}



Boolean Primary()
/*      -------  */
{
    if ( Atom() )
    {
	return true;
    }
    else
    if ( Find("(") )
    {
	BN++;
	if ( ! Expression() ) FailSyn("expression in parentheses");
	if ( ! Find(")") ) FailSyn("')'");
	BN++;
	return true;
    }
    else
    {
	FailSyn("attribute, value, or '('");
    }
}



String Funcs[] = {"sin", "cos", "tan", "log", "exp", "int", (String) 0};

Boolean Atom()
/*      ----  */
{
    char	*EndPtr, *Str, Date[11], Time[9];
    int		o, FirstBN, Fi=BN;
    ContValue	F;
    Attribute	Att;

    if ( Buff[BN] == ' ' ) BN++;

    if ( Buff[BN] == '"' )
    {
	FirstBN = ++BN;
	while ( Buff[BN] != '"' )
	{
	    if ( ! Buff[BN] ) FailSyn("closing '\"'");
	    BN++;
	}

	/*  Make a copy of the string without double quotes  */

	Buff[BN] = '\00';
	Str = CopyString(Buff + FirstBN);

	Buff[BN++] = '"';
	Dump(OP_STR, 0, Str, Fi);
    }
    else
    if ( (Att = FindAttName()) )
    {
	BN += strlen(AttName[Att]);

	Dump(OP_ATT, 0, (String) Att, Fi);
    }
    else
    if ( isdigit(Buff[BN]) )
    {
	/*  Check for date or time first  */

	if ( ( Buff[BN+4] == '/' && Buff[BN+7] == '/' ||
	       Buff[BN+4] == '-' && Buff[BN+7] == '-' )&&
	     isdigit(Buff[BN+1]) && isdigit(Buff[BN+2]) &&
		isdigit(Buff[BN+3]) &&
	     isdigit(Buff[BN+5]) && isdigit(Buff[BN+6]) &&
	     isdigit(Buff[BN+8]) && isdigit(Buff[BN+9]) )
	{
	    memcpy(Date, Buff+BN, 10);
	    Date[10] = '\00';
	    if ( (F = DateToDay(Date)) == 0 )
	    {
		Error(BADDEF1, Date, "date");
	    }

	    BN += 10;
	}
	else
	if ( Buff[BN+2] == ':' && Buff[BN+5] == ':' &&
	     isdigit(Buff[BN+1]) &&
	     isdigit(Buff[BN+3]) && isdigit(Buff[BN+4]) &&
	     isdigit(Buff[BN+6]) && isdigit(Buff[BN+7]) )
	{
	    memcpy(Time, Buff+BN, 8);
	    Time[8] = '\00';
	    if ( (F = TimeToSecs(Time)) == 0 )
	    {
		Error(BADDEF1, Time, "time");
	    }

	    BN += 8;
	}
	else
	{
	    F = strtod(Buff+BN, &EndPtr);

	    /*  Check for period after integer  */

	    if ( EndPtr > Buff+BN+1 && *(EndPtr-1) == '.' )
	    {
		EndPtr--;
	    }

	    BN = EndPtr - Buff;
	}

	Dump(OP_NUM, F, Nil, Fi);
    }
    else
    if ( (o = FindOne(Funcs)) >= 0 )
    {
	BN += 3;

	if ( ! Find("(") ) FailSyn("'(' after function name");
	BN++;

	if ( ! Expression() ) FailSyn("expression");

	if ( ! Find(")") ) FailSyn("')' after function argument");
	BN++;

	DumpOp((char)(OP_SIN + o), Fi);
    }
    else
    if ( Buff[BN] == '?' )
    {
	BN++;
	if ( TStack[TSN-1].Type == 'N' )
	{
	    Dump(OP_NUM, _UNK._cont_val, Nil, Fi);
	}
	else
	{
	    Dump(OP_STR, 0, Nil, Fi);
	}
    }
    else
    if ( ! memcmp(Buff+BN, "N/A", 3) )
    {
	BN += 3;
	if ( TStack[TSN-1].Type == 'N' )
	{
	    Dump(OP_NUM, _NA._cont_val, Nil, Fi);
	}
	else
	{
	    Dump(OP_STR, 0, CopyString("N/A"), Fi);
	}
    }
    else
    {
	return false;
    }

    return true;
}



/*************************************************************************/
/*									 */
/*	Skip spaces and check for specific string			 */
/*									 */
/*************************************************************************/


Boolean Find(String S)
/*      ----  */
{
    if ( Buff[BN] == ' ' ) BN++;

    return ( ! Buff[BN] ? false : ! memcmp(Buff+BN, S, strlen(S)) );
}



/*************************************************************************/
/*									 */
/*	Find one of a zero-terminated list of alternatives		 */
/*									 */
/*************************************************************************/


int FindOne(String *Alt)
/*  -------  */
{
    int	a;

    for ( a = 0 ; Alt[a] ; a++ )
    {
	if ( Find(Alt[a]) ) return a;
    }

    return -1;
}



/*************************************************************************/
/*									 */
/*	Find an attribute name						 */
/*									 */
/*************************************************************************/


Attribute FindAttName()
/*        -----------  */
{
    Attribute	Att, LongestAtt=0;

    ForEach(Att, 1, MaxAtt-1)
    {
	if ( ! Exclude(Att) && Find(AttName[Att]) )
	{
	    if ( ! LongestAtt ||
		 strlen(AttName[Att]) > strlen(AttName[LongestAtt]) )
	    {
		LongestAtt = Att;
	    }
	}
    }

    return LongestAtt;
}



/*************************************************************************/
/*									 */
/*	Error message routines.  Syntax errors come from the		 */
/*	recursive descent parser, semantics errors from the routines	 */
/*	that build up the equivalent polish				 */
/*									 */
/*************************************************************************/


void DefSyntaxError(String Msg)
/*   --------------  */
{
    String	RestOfText;
    int		i=10;

    if ( ! PreviousError )
    {
	RestOfText = Buff + BN;

	/*  Abbreviate text if longer than 12 characters  */

	if ( CharWidth(RestOfText) > 12 )
	{
#ifdef UTF8
	    /*  Find beginning of UTF-8 character  */

	    for ( ; (RestOfText[i] & 0x80) ; i++)
		;
#endif
	    RestOfText[i] = RestOfText[i+1] = '.';
	}

	Error(BADDEF1, RestOfText, Msg);
	PreviousError = true;
    }
}



void DefSemanticsError(int Fi, String Msg, int OpCode)
/*   -----------------  */
{
    char	Exp[1000], XMsg[1000], Op[1000];

    if ( ! PreviousError )
    {
	/*  Abbreviate the input if necessary  */

	if ( BN - Fi > 23 )
	{
	    sprintf(Exp, "%.10s...%.10s", Buff+Fi, Buff+BN-10);
	}
	else
	{
	    sprintf(Exp, "%.*s", BN - Fi, Buff+Fi);
	}

	switch ( OpCode )
	{
	    case OP_AND:	sprintf(Op, "%s", "and"); break;
	    case OP_OR:		sprintf(Op, "%s", "or"); break;
	    case OP_SEQ:
	    case OP_EQ:		sprintf(Op, "%s", "="); break;
	    case OP_SNE:
	    case OP_NE:		sprintf(Op, "%s", "<>"); break;
	    case OP_GT:		sprintf(Op, "%s", ">"); break;
	    case OP_GE:		sprintf(Op, "%s", ">="); break;
	    case OP_LT:		sprintf(Op, "%s", "<"); break;
	    case OP_LE:		sprintf(Op, "%s", "<="); break;
	    case OP_PLUS:	sprintf(Op, "%s", "+"); break;
	    case OP_MINUS:	sprintf(Op, "%s", "-"); break;
	    case OP_UMINUS:	sprintf(Op, "%s", "unary -"); break;
	    case OP_MULT:	sprintf(Op, "%s", "*"); break;
	    case OP_DIV:	sprintf(Op, "%s", "/"); break;
	    case OP_MOD:	sprintf(Op, "%s", "%"); break;
	    case OP_POW:	sprintf(Op, "%s", "^"); break;
	    case OP_SIN:	sprintf(Op, "%s", "sin"); break;
	    case OP_COS:	sprintf(Op, "%s", "cos"); break;
	    case OP_TAN:	sprintf(Op, "%s", "tan"); break;
	    case OP_LOG:	sprintf(Op, "%s", "log"); break;
	    case OP_EXP:	sprintf(Op, "%s", "exp"); break;
	    case OP_INT:	sprintf(Op, "%s", "int");
	}

	sprintf(XMsg, "%s with '%s'", Msg, Op);
	Error(BADDEF2, Exp, XMsg);
	PreviousError = true;
    }
}



/*************************************************************************/
/*									 */
/*	Reverse polish routines.  These use a model of the stack	 */
/*	during expression evaluation to detect type conflicts etc	 */
/*									 */
/*************************************************************************/



void Dump(char OpCode, ContValue F, String S, int Fi)
/*   ----  */
{
    if ( Buff[Fi] == ' ' ) Fi++;

    if ( ! UpdateTStack(OpCode, F, S, Fi) ) return;

    /*  Make sure enough room for this element  */

    if ( DN >= DefSize-1 )
    {
	Realloc(AttDef[MaxAtt], DefSize += 100, DefElt);
    }

    DefOp(AttDef[MaxAtt][DN]) = OpCode;
    if ( OpCode == OP_ATT || OpCode == OP_STR )
    {
	DefSVal(AttDef[MaxAtt][DN]) = S;
    }
    else
    {
	DefNVal(AttDef[MaxAtt][DN]) = F;
    }

    DN++;
}



void DumpOp(char OpCode, int Fi)
/*   ------  */
{
    Dump(OpCode, 0, Nil, Fi);
}



Boolean UpdateTStack(char OpCode, ContValue F, String S, int Fi)
/*      ------------  */
{
    if ( TSN >= TStackSize )
    {
	Realloc(TStack, TStackSize += 50, EltRec);
    }

    switch ( OpCode )
    {
	case OP_ATT:
		TStack[TSN].Type = ( Continuous((int) S) ? 'N' : 'S' );
		break;

	case OP_NUM:
		TStack[TSN].Type = 'N';
		break;

	case OP_STR:
		TStack[TSN].Type = 'S';
		break;

	case OP_AND:
	case OP_OR:
		if ( TStack[TSN-2].Type != 'B' || TStack[TSN-1].Type != 'B' )
		{
		    FailSem("non-logical value");
		}
		TSN -= 2;
		break;

	case OP_EQ:
	case OP_NE:
		if ( TStack[TSN-2].Type != TStack[TSN-1].Type )
		{
		    FailSem("incompatible values");
		}
		TSN -= 2;
		TStack[TSN].Type = 'B';
		break;

	case OP_GT:
	case OP_GE:
	case OP_LT:
	case OP_LE:
		if ( TStack[TSN-2].Type != 'N' || TStack[TSN-1].Type != 'N' )
		{
		    FailSem("non-arithmetic value");
		}
		TSN -= 2;
		TStack[TSN].Type = 'B';
		break;

	case OP_SEQ:
	case OP_SNE:
		if ( TStack[TSN-2].Type != 'S' || TStack[TSN-1].Type != 'S' )
		{
		    FailSem("incompatible values");
		}
		TSN -= 2;
		TStack[TSN].Type = 'B';
		break;

	case OP_PLUS:
	case OP_MINUS:
	case OP_MULT:
	case OP_DIV:
	case OP_MOD:
	case OP_POW:
		if ( TStack[TSN-2].Type != 'N' || TStack[TSN-1].Type != 'N' )
		{
		    FailSem("non-arithmetic value");
		}
		TSN -= 2;
		break;

	case OP_UMINUS:
		if ( TStack[TSN-1].Type != 'N' )
		{
		    FailSem("non-arithmetic value");
		}
		TSN--;
		break;

	case OP_SIN:
	case OP_COS:
	case OP_TAN:
	case OP_LOG:
	case OP_EXP:
	case OP_INT:
		if ( TStack[TSN-1].Type != 'N' )
		{
		    FailSem("non-arithmetic argument");
		}
		TSN--;
    }

    TStack[TSN].Fi = Fi;
    TStack[TSN].Li = BN-1;
    TSN++;

    return true;
}



/*************************************************************************/
/*									 */
/*	Evaluate an implicit attribute for a case			 */
/*									 */
/*************************************************************************/

#define UNA(a)	(UnknownVal(XStack[a]) || NotApplicVal(XStack[a]))
#define	C1(x)	(UNA(XSN-1) ? _UNK.cval : (x))
#define	C2(x)	(UNA(XSN-1) || UNA(XSN-2) ? _UNK.cval : (x))
#define	D2(x)	(UNA(XSN-1) || UNA(XSN-2) ? _UNK.dval : (x))
#define	D2(x)	(UNA(XSN-1) || UNA(XSN-2) ? _UNK.dval : (x))
#define	SetCVal
#define	SetDVal


AttValue EvaluateDef(Definition D, Description Case)
/*       -----------  */
{
    AttValue	XStack[100];			/* allows 100-level nesting  */
    int		XSN=0, DN, bv1, bv2, Mult;
    ContValue	cv1, cv2;
    String	sv1, sv2;
    Attribute	Att;
    DefElt	DElt;

    for ( DN = 0 ; ; DN++)
    {
	switch ( DefOp((DElt = D[DN])) )
	{
	    case OP_ATT:
		    Att = (int) DefSVal(DElt);

//#if defined PREDICT || defined SEE5 && defined WIN32 && ! defined _CONSOLE
//		    GetValue(Att);
//#endif
		    if ( Continuous(Att) )
		    {
			XStack[XSN++].cval =
			    ( NotApplic(Case, Att) ? _NA.cval :
			      Unknown(Case, Att) ? _UNK.cval :
			      CVal(Case, Att) );
			SetDVal;
		    }
		    else
		    {
			XStack[XSN++].sval =
			    ( Unknown(Case, Att) && ! NotApplic(Case, Att) ? 0 :
			      AttValName[Att][XDVal(Case, Att)] );
			SetCVal;
		    }
		    break;

	    case OP_NUM:
		    XStack[XSN++].cval = DefNVal(DElt);
		    SetDVal;
		    break;

	    case OP_STR:
		    XStack[XSN++].sval = DefSVal(DElt);
		    SetCVal;
		    break;

	    case OP_AND:
		    bv1 = XStack[XSN-2].dval;
		    bv2 = XStack[XSN-1].dval;
		    XStack[XSN-2].dval = ( bv1 == 3 || bv2 == 3 ? 3 :
					   D2(bv1 == 2 && bv2 == 2 ? 2 : 3) );
		    XSN--;
		    SetCVal;
		    break;

	    case OP_OR:
		    bv1 = XStack[XSN-2].dval;
		    bv2 = XStack[XSN-1].dval;
		    XStack[XSN-2].dval = ( bv1 == 2 || bv2 == 2 ? 2 :
					   D2(bv1 == 2 || bv2 == 2 ? 2 : 3) );
		    XSN--;
		    SetCVal;
		    break;

	    case OP_EQ:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = ( cv1 == cv2 ? 2 : 3 );
		    XSN--;
		    SetCVal;
		    break;

	    case OP_NE:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = ( cv1 != cv2 ? 2 : 3 );
		    XSN--;
		    SetCVal;
		    break;

	    case OP_GT:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = D2(cv1 > cv2 ? 2 : 3);
		    XSN--;
		    SetCVal;
		    break;

	    case OP_GE:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = D2(cv1 >= cv2 ? 2 : 3);
		    XSN--;
		    SetCVal;
		    break;

	    case OP_LT:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = D2(cv1 < cv2 ? 2 : 3);
		    XSN--;
		    SetCVal;
		    break;

	    case OP_LE:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].dval = D2(cv1 <= cv2 ? 2 : 3);
		    XSN--;
		    SetCVal;
		    break;

	    case OP_SEQ:
		    sv1 = XStack[XSN-2].sval;
		    sv2 = XStack[XSN-1].sval;
		    XStack[XSN-2].dval =
			( ! sv1 && ! sv2 ? 2 :
			  ! sv1 || ! sv2 ? 3 :
			  ! strcmp(sv1, sv2) ? 2 : 3 );
		    XSN--;
		    SetCVal;
		    break;

	    case OP_SNE:
		    sv1 = XStack[XSN-2].sval;
		    sv2 = XStack[XSN-1].sval;
		    XStack[XSN-2].dval =
			( ! sv1 && ! sv2 ? 3 :
			  ! sv1 || ! sv2 ? 2 :
			  strcmp(sv1, sv2) ? 2 : 3 );
		    XSN--;
		    SetCVal;
		    break;

	    case OP_PLUS:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].cval = C2(cv1 + cv2);
		    XSN--;
		    SetDVal;
		    break;

	    case OP_MINUS:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].cval = C2(cv1 - cv2);
		    XSN--;
		    SetDVal;
		    break;

	    case OP_MULT:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].cval = C2(cv1 * cv2);
		    XSN--;
		    SetDVal;
		    break;

	    case OP_DIV:
		    /*  Note: have to set precision of result  */

		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    if ( ! cv2 ||
			 UnknownVal(XStack[XSN-2]) ||
			 UnknownVal(XStack[XSN-1]) ||
			 NotApplicVal(XStack[XSN-2]) ||
			 NotApplicVal(XStack[XSN-1]) )
		    {
			XStack[XSN-2] = _UNK;
		    }
		    else
		    {
			Mult = Denominator(cv1);
			cv1 = cv1 / cv2;
			while ( fabs(cv2) > 1 )
			{
			    Mult *= 10;
			    cv2 /= 10;
			}
			XStack[XSN-2].cval = rint(cv1 * Mult) / Mult;
		    }
		    XSN--;
		    SetDVal;
		    break;

	    case OP_MOD:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].cval = C2(fmod(cv1, cv2));
		    XSN--;
		    SetDVal;
		    break;

	    case OP_POW:
		    cv1 = XStack[XSN-2].cval;
		    cv2 = XStack[XSN-1].cval;
		    XStack[XSN-2].cval =
			( UNA(XSN-1) || UNA(XSN-2) ||
			  ( cv1 < 0 && ceil(cv2) != cv2 ) ? _UNK.cval :
			  pow(cv1, cv2) );
		    XSN--;
		    SetDVal;
		    break;

	    case OP_UMINUS:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(-cv1);
		    SetDVal;
		    break;

	    case OP_SIN:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(sin(cv1));
		    SetDVal;
		    break;

	    case OP_COS:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(cos(cv1));
		    SetDVal;
		    break;

	    case OP_TAN:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(tan(cv1));
		    SetDVal;
		    break;

	    case OP_LOG:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval =
			( UNA(XSN-1) || cv1 <= 0 ? _UNK.cval : log(cv1) );
		    SetDVal;
		    break;

	    case OP_EXP:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(exp(cv1));
		    SetDVal;
		    break;

	    case OP_INT:
		    cv1 = XStack[XSN-1].cval;
		    XStack[XSN-1].cval = C1(rint(cv1));
		    SetDVal;
		    break;

	    case OP_END:
		    return XStack[0];
	}
    }
}



/*************************************************************************/
/*									 */
/*	Routines for reading model files				 */
/*	--------------------------------				 */
/*									 */
/*************************************************************************/

/*
Boolean	BINARY=false;
int	Entry;

char*	Prop[]={"null",
		"att",
		"class",
		"cut",
		"conds",
		"elts",
		"entries",
		"forks",
		"freq",
		"id",
		"type",
		"low",
		"mid",
		"high",
		"result",
		"rules",
		"val",
		"lift",
		"cover",
		"ok",
		"default",
		"sample",
		"init"
	       };

char	PropName[20],
	*PropVal=Nil,
	*Unquoted;
int	PropValSize=0;

#define	PROPS 22

#define ERRORP		0
#define ATTP		1
#define CLASSP		2
#define CUTP		3
#define	CONDSP		4
#define ELTSP		5
#define ENTRIESP	6
#define FORKSP		7
#define FREQP		8
#define IDP		9
#define TYPEP		10
#define LOWP		11
#define MIDP		12
#define HIGHP		13
#define RESULTP		14
#define RULESP		15
#define VALP		16
#define LIFTP		17
#define COVERP		18
#define OKP		19
#define DEFAULTP	20
#define SAMPLEP		21
#define INITP		22

*/

/*************************************************************************/
/*									 */
/*	Read header information and decide whether model files are	 */
/*	in ASCII or binary format					 */
/*									 */
/*************************************************************************/

/*
void ReadFilePrefix(String Extension)
{
#if defined WIN32 || defined _CONSOLE
    if ( ! (TRf = GetFile(Extension, "rb")) ) Error(NOFILE, Fn, "");
#else
    if ( ! (TRf = GetFile(Extension, "r")) ) Error(NOFILE, Fn, "");
#endif

    StreamIn((char *) &TRIALS, sizeof(int));
    if ( memcmp((char *) &TRIALS, "id=", 3) != 0 )
    {
	BINARY = true;
	BinRecoverDiscreteNames();
    }
    else
    {
	BINARY = false;
	rewind(TRf);
	ReadHeader();
    }
}

*/

/*************************************************************************/
/*								  	 */
/*	Read the header information (id, saved names, models)		 */
/*								  	 */
/*************************************************************************/
/*
void ReadHeader()
{
    Attribute	Att;
    DiscrValue	v;
    char	*p, Dummy;
    int		Year, Month, Day;

    while ( true )
    {
	switch ( ReadProp(&Dummy) )
	{
	    case ERRORP:
		return;

	    case IDP: //  Recover year run and set base date for timestamps  

		if ( sscanf(PropVal + strlen(PropVal) - 11,
			    "%d-%d-%d\"", &Year, &Month, &Day) == 3 )
		{
		    SetTSBase(Year);
		}
		break;
	    case ATTP:
		Unquoted = RemoveQuotes(PropVal);
		Att = Which(Unquoted, AttName, 1, MaxAtt);
		if ( ! Att || Exclude(Att) )
		{
		    Error(MODELFILE, E_MFATT, Unquoted);
		}
		break;

	    case ELTSP:
		MaxAttVal[Att] = 1;
		AttValName[Att][1] = CopyString("N/A");

		for ( p = PropVal ; *p ; )
		{
		    p = RemoveQuotes(p);
		    v = ++MaxAttVal[Att];
		    AttValName[Att][v] = Alloc(strlen(p)+1, char);
		    strcpy(AttValName[Att][v], p);

		    for ( p += strlen(p) ; *p != '"' ; p++ )
			;
		    p++;
		    if ( *p == ',' ) p++;
		}
		AttValName[Att][MaxAttVal[Att]+1] = "<other>";
		break;

	    case ENTRIESP:
		sscanf(PropVal, "\"%d\"", &TRIALS);
		Entry = 0;
		return;
	}
    }
}
*/


/*************************************************************************/
/*									 */
/*	Retrieve decision tree with extension Extension			 */
/*									 */
/*************************************************************************/

/*
Tree GetTree(String Extension)
{
    CheckFile(Extension, false);

    return ( BINARY ? BinInTree() : InTree() );
}
*/

/*
Tree InTree()
{
    Tree	T;
    DiscrValue	v, Subset=0;
    char	Delim, *p;
    ClassNo	c;
    int		X;
    double	XD;

    T = (Tree) AllocZero(1, TreeRec);

    do
    {
	switch ( ReadProp(&Delim) )
	{
	    case ERRORP:
		return 0;

	    case TYPEP:
		sscanf(PropVal, "\"%d\"", &X); T->NodeType = X;
		break;

	    case CLASSP:
		Unquoted = RemoveQuotes(PropVal);
		T->Leaf = Which(Unquoted, ClassName, 1, MaxClass);
		if ( ! T->Leaf ) Error(MODELFILE, E_MFCLASS, Unquoted);
		break;

	    case ATTP:
		Unquoted = RemoveQuotes(PropVal);
		T->Tested = Which(Unquoted, AttName, 1, MaxAtt);
		if ( ! T->Tested || Exclude(T->Tested) )
		{
		    Error(MODELFILE, E_MFATT, Unquoted);
		}
		break;

	    case CUTP:
		sscanf(PropVal, "\"%lf\"", &XD);	T->Cut = XD;
		T->Lower = T->Mid = T->Upper = T->Cut;
		break;

	    case LOWP:
		sscanf(PropVal, "\"%lf\"", &XD);	T->Lower = XD;
		break;

	    case MIDP:
		sscanf(PropVal, "\"%lf\"", &XD);	T->Mid = XD;
		break;

	    case HIGHP:
		sscanf(PropVal, "\"%lf\"", &XD);	T->Upper = XD;
		break;

	    case FORKSP:
		sscanf(PropVal, "\"%d\"", &T->Forks);
		break;

	    case FREQP:
		T->ClassDist = Alloc(MaxClass+1, ItemCount);
		p = PropVal+1;

		ForEach(c, 1, MaxClass)
		{
		    T->ClassDist[c] = strtod(p, &p);
		    T->Items += T->ClassDist[c];
		    p++;
		}
		break;

	    case ELTSP:
		if ( ! Subset++ )
		{
		    T->Subset = AllocZero(T->Forks+1, Set);
		}

		T->Subset[Subset] = MakeSubset(T->Tested);
		break;
	}
    }
    while ( Delim == ' ' );

    if ( T->ClassDist )
    {
	T->Errors = T->Items - T->ClassDist[T->Leaf];
    }
    else
    {
	T->ClassDist = Alloc(1, ItemCount);
    }

    if ( T->NodeType )
    {
	T->Branch = AllocZero(T->Forks+1, Tree);
	ForEach(v, 1, T->Forks)
	{
	    T->Branch[v] = InTree();
	}
    }

    return T;
}
*/


/*************************************************************************/
/*									 */
/*	Retrieve ruleset with extension Extension			 */
/*	(Separate functions for ruleset, single rule, single condition)	 */
/*									 */
/*************************************************************************/

/*
CRuleSet GetRules(String Extension)
{
    CheckFile(Extension, false);

    return ( BINARY ? BinInRules() : InRules() );
}

*/
/*

CRuleSet InRules()
{
    CRuleSet	RS;
    RuleNo	r;
    char	Delim;

    RS = Alloc(1, RuleSetRec);

    do
    {
	switch ( ReadProp(&Delim) )
	{
	    case ERRORP:
		return 0;

	    case RULESP:
		sscanf(PropVal, "\"%d\"", &RS->SNRules);
		break;

	    case DEFAULTP:
		Unquoted = RemoveQuotes(PropVal);
		RS->SDefault = Which(Unquoted, ClassName, 1, MaxClass);
		if ( ! RS->SDefault ) Error(MODELFILE, E_MFCLASS, Unquoted);
		break;
	}
    }
    while ( Delim == ' ' );

    //  Read each rule  

    RS->SRule = Alloc(RS->SNRules+1, CRule);
    ForEach(r, 1, RS->SNRules)
    {
	if ( RS->SRule[r] = InRule() )
	{
	    RS->SRule[r]->RNo = r;
	    RS->SRule[r]->TNo = Entry;
	}
    }

    Entry++;
    return RS;
}
*/

/*
CRule InRule()
{
    CRule	R;
    int		d;
    char	Delim;
    double	Lift;

    R = Alloc(1, RuleRec);

    do
    {
	switch ( ReadProp(&Delim) )
	{
	    case ERRORP:
		return 0;

	    case CONDSP:
		sscanf(PropVal, "\"%d\"", &R->Size);
		break;

	    case COVERP:
		sscanf(PropVal, "\"%f\"", &R->Cover);
		break;

	    case OKP:
		sscanf(PropVal, "\"%f\"", &R->Correct);
		break;

	    case LIFTP:
		sscanf(PropVal, "\"%f\"", &Lift);
		R->Prior = (R->Correct + 1) / ((R->Cover + 2) * Lift);
		break;

	    case CLASSP:
		Unquoted = RemoveQuotes(PropVal);
		R->Rhs = Which(Unquoted, ClassName, 1, MaxClass);
		if ( ! R->Rhs ) Error(MODELFILE, E_MFCLASS, Unquoted);
		break;
	}
    }
    while ( Delim == ' ' );

    R->Lhs = Alloc(R->Size+1, Condition);
    ForEach(d, 1, R->Size)
    {
	R->Lhs[d] = InCondition();
    }

    R->Vote = int(1000 * (R->Correct + 1.0) / (R->Cover + 2.0) + 0.5);

    return R;
}
*/

/*
Condition InCondition()
{
    Condition	C;
    char	Delim;
    int		X;
    double	XD;

    C = Alloc(1, CondRec);

    do
    {
	switch ( ReadProp(&Delim) )
	{
	    case ERRORP:
		return 0;

	    case TYPEP:
		sscanf(PropVal, "\"%d\"", &X); C->NodeType = X;
		break;

	    case ATTP:
		Unquoted = RemoveQuotes(PropVal);
		C->Tested = Which(Unquoted, AttName, 1, MaxAtt);
		if ( ! C->Tested || Exclude(C->Tested) )
		{
		    Error(MODELFILE, E_MFATT, Unquoted);
		}
		break;

	    case CUTP:
		sscanf(PropVal, "\"%lf\"", &XD);	C->Cut = XD;
		break;

	    case RESULTP:
		C->TestValue = ( PropVal[1] == '<' ? 2 : 3 );
		break;

	    case VALP:
		if ( Continuous(C->Tested) )
		{
		    C->TestValue = 1;
		}
		else
		{
		    Unquoted = RemoveQuotes(PropVal);
		    C->TestValue = Which(Unquoted,
					 AttValName[C->Tested],
					 1, MaxAttVal[C->Tested]);
		    if ( ! C->TestValue ) Error(MODELFILE, E_MFATTVAL, Unquoted);
		}
		break;

	    case ELTSP:
		C->Subset = MakeSubset(C->Tested);
		C->TestValue = 1;
		break;
	}
    }
    while ( Delim == ' ' );

    return C;
}
*/


/*************************************************************************/
/*									 */
/*	ASCII reading utilities						 */
/*									 */
/*************************************************************************/

/*
int ReadProp(char *Delim)
{
    int		c, i;
    char	*p;
    Boolean	Quote=false;

    for ( p = PropName ; (c = fgetc(TRf)) != '=' ;  )
    {
	if ( p - PropName >= 19 || c == EOF )
	{
	    Error(MODELFILE, E_MFEOF, "");
	    PropName[0] = PropVal[0] = *Delim = '\00';
	    return 0;
	}
	*p++ = c;
    }
    *p = '\00';

    for ( p = PropVal ; ((c = fgetc(TRf)) != ' ' && c != '\n') || Quote ; )
    {
	if ( c == EOF )
	{
	    Error(MODELFILE, E_MFEOF, "");
	    PropName[0] = PropVal[0] = '\00';
	    return 0;
	}

	if ( (i = p - PropVal) >= PropValSize )
	{
	    Realloc(PropVal, (PropValSize += 10000) + 3, char);
	    p = PropVal + i;
	}

	*p++ = c;
	if ( c == '\\' )
	{
	    *p++ = fgetc(TRf);
	}
	else
	if ( c == '"' )
	{
	    Quote = ! Quote;
	}
    }
    *p = '\00';
    *Delim = c;

    return Which(PropName, Prop, 1, PROPS);
}
*/

String RemoveQuotes(String S)
/*     ------------  */
{
    char	*p, *Start;

    p = Start = S;
    
    for ( S++ ; *S != '"' ; S++ )
    {
	if ( *S == '\\' ) S++;
	*p++ = *S;
	*S = '-';
    }
    *p = '\00';

    return Start;
}


/*
Set MakeSubset(Attribute Att)
{
    int		Bytes, b;
    char	*p;
    Set		S;

    Bytes = (MaxAttVal[Att]>>3) + 1;
    S = AllocZero(Bytes, unsigned char);

    for ( p = PropVal ; *p ; )
    {
	p = RemoveQuotes(p);
	b = Which(p, AttValName[Att], 1, MaxAttVal[Att]);
	if ( ! b ) Error(MODELFILE, E_MFATTVAL, p);
	SetBit(b, S);

	for ( p += strlen(p) ; *p != '"' ; p++ )
	    ;
	p++;
	if ( *p == ',' ) p++;
    }

    return S;
}

*/

/*************************************************************************/
/*									 */
/*	Recover attribute values read with "discrete N"			 */
/*									 */
/*************************************************************************/


void BinRecoverDiscreteNames()
/*   -----------------------  */
{
    Attribute	Att;
    DiscrValue	v;
    int		Length;

    ForEach(Att, 1, MaxAtt)
    {
	if ( ! StatBit(Att, DISCRETE) ) continue;

	StreamIn((char *) &MaxAttVal[Att], sizeof(int));

	/*  Insert "N/A"  */

	AttValName[Att][1] = CopyString("N/A");
	MaxAttVal[Att]++;

	ForEach(v, 2, MaxAttVal[Att])
	{
	    StreamIn((char *) &Length, sizeof(int));

	    AttValName[Att][v] = Alloc(Length, char);
	    StreamIn(AttValName[Att][v], Length);
	}

	/*  Invisible name for undefined values  */

	AttValName[Att][MaxAttVal[Att]+1] = "<other>";
    }
}



/*************************************************************************/
/*									 */
/*	Retrieve tree from saved characters				 */
/*									 */
/*************************************************************************/

/*
Tree BinInTree()
{
    Tree	T;
    DiscrValue	v, vv;
    int		Bytes;
    double	XFl;
    Set		S;

    T = (Tree) AllocZero(1, TreeRec);

    StreamIn((char *) &T->NodeType, sizeof(BranchType));
    StreamIn((char *) &T->Leaf, sizeof(ClassNo));
    StreamIn((char *) &T->Items, sizeof(ItemCount));
    StreamIn((char *) &T->Errors, sizeof(ItemCount));

    T->ClassDist = AllocZero(MaxClass+1, ItemCount);
    StreamIn((char *) T->ClassDist, (MaxClass + 1) * sizeof(ItemCount));

    if ( T->NodeType )
    {
	StreamIn((char *) &T->Tested, sizeof(Attribute));
	StreamIn((char *) &T->Forks, sizeof(int));
	T->Forks++;	// for N/A 

	switch ( T->NodeType )
	{
	    case BrDiscr:
		break;

	    case BrThresh:
		StreamIn((char *) &XFl, sizeof(double));	T->Cut = XFl;
		StreamIn((char *) &XFl, sizeof(double));	T->Lower = XFl;
		StreamIn((char *) &XFl, sizeof(double));	T->Upper = XFl;
		StreamIn((char *) &XFl, sizeof(double));	T->Mid = XFl;
		break;

	    case BrSubset:
		T->Subset = (Set *) AllocZero(T->Forks+1, Set);

		Bytes = ((MaxAttVal[T->Tested] - 1) >> 3) + 1;
		S = AllocZero(Bytes, unsigned char);
		T->Subset[1] = AllocZero(Bytes, unsigned char);
		SetBit(1, T->Subset[1]);

		ForEach(v, 2, T->Forks)
		{
		    T->Subset[v] = AllocZero(Bytes, unsigned char);
		    StreamIn((char *) S, Bytes);
		    ForEach(vv, 1, MaxAttVal[T->Tested]-1)
		    {
			if ( In(vv, S) ) SetBit(vv+1, T->Subset[v]);
		    }
		}

		free(S);
	}

	T->Branch = AllocZero(T->Forks+1, Tree);

	//  Allow for N/A branch  

	T->Branch[1] = Leaf(T->ClassDist, T->Leaf, 0.0, 0.0);

	ForEach(v, 2, T->Forks)
	{
	    T->Branch[v] = BinInTree();
	}
    }

    return T;
}
*/


/*************************************************************************/
/*								  	 */
/*	Recover a ruleset						 */
/*								  	 */
/*************************************************************************/

/*
CRuleSet BinInRules()
{
    int		ri, d, Bytes, Dummy;
    CRuleSet	RS;
    CRule	R;
    Condition	C;
    double	XFl;
    Set		S;
    DiscrValue	vv;

    RS = Alloc(1, RuleSetRec);

    StreamIn((char *) &RS->SNRules, sizeof(RuleNo));
    StreamIn((char *) &RS->SDefault, sizeof(ClassNo));

    RS->SRule = Alloc(RS->SNRules+1, CRule);

    ForEach(ri, 1, RS->SNRules)
    {
	R = RS->SRule[ri] = Alloc(1, RuleRec);

	StreamIn((char *) &R->RNo, sizeof(int));
	StreamIn((char *) &R->TNo, sizeof(int));
	StreamIn((char *) &R->Size, sizeof(int));

	R->Lhs = Alloc(R->Size+1, Condition);
	ForEach(d, 1, R->Size)
	{
	    C = R->Lhs[d] = Alloc(1, CondRec);

	    StreamIn((char *) &C->NodeType, sizeof(BranchType));
	    StreamIn((char *) &C->Tested, sizeof(Attribute));
	    StreamIn((char *) &Dummy, sizeof(int));
	    StreamIn((char *) &XFl, sizeof(double));	C->Cut = XFl;
	    if ( C->NodeType == BrSubset )
	    {
		Bytes = ((MaxAttVal[C->Tested] - 1) >> 3) + 1;
		S = AllocZero(Bytes, unsigned char);

		C->Subset = AllocZero(Bytes, unsigned char);
		StreamIn((char *) S, Bytes);
		ForEach(vv, 1, MaxAttVal[C->Tested]-1)
		{
		    if ( In(vv, S) ) SetBit(vv+1, C->Subset);
		}

		free(S);
	    }
	    StreamIn((char *) &R->Lhs[d]->TestValue, sizeof(int));
	    R->Lhs[d]->TestValue++;	// to allow for N/A 
	}
	StreamIn((char *) &R->Rhs, sizeof(ClassNo));
	StreamIn((char *) &R->Cover, sizeof(ItemCount));
	StreamIn((char *) &R->Correct, sizeof(ItemCount));
	StreamIn((char *) &R->Prior, sizeof(double));

	if ( R->Correct < 1 )
	{
	    //  Prior to Release 1.11  

	    R->Correct = (R->Cover + 2) * (1 - R->Correct) - 1;
	    memcpy(&R->Vote, &R->Prior, sizeof(int));
	    R->Prior = 1E38;
	}
	else
	{
	    R->Vote = int(1000 * (R->Correct + 1.0) / (R->Cover + 2.0) + 0.5);
	}
    }

    return RS;
}
*/


/*************************************************************************/
/*								  	 */
/*	Character stream read for binary routines			 */
/*								  	 */
/*************************************************************************/


void StreamIn(String S, int n)
/*   --------  */
{
    while ( n-- ) *S++ = getc(TRf);
}



/*************************************************************************/
/*									 */
/*	Construct a leaf in a given node				 */
/*									 */
/*************************************************************************/

/*
Tree Leaf(ItemCount *Freq, ClassNo NodeClass, ItemCount Items, ItemCount Errors)
{
    Tree	Node;

    Node = AllocZero(1, TreeRec);

    Node->ClassDist = Alloc(MaxClass+1, ItemCount);
    memcpy(Node->ClassDist, Freq, (MaxClass+1) * sizeof(ItemCount));

    Node->NodeType	= 0;
    Node->Leaf		= NodeClass;
    Node->Items		= Items;
    Node->Errors	= Errors;

    return Node;
}
*/

/*************************************************************************/
/*									 */
/*	Read variable misclassification costs				 */
/*									 */
/*************************************************************************/


void GetMCosts(FILE *Cf)
/*   ---------  */
{
    ClassNo	Pred, Real, p, r;
    char	Name[1000];
    double	Val;

    LineNo = 1;

    /*  Read entries from cost file  */

    while ( ReadName(Cf, Name, 1000, ':') )
    {
	if ( ! (Pred = Which(Name, ClassName, 1, MaxClass)) )
	{
	    Error(BADCOSTCLASS, Name, "");
	}

	if ( ! ReadName(Cf, Name, 1000, ':') ||
	     ! (Real = Which(Name, ClassName, 1, MaxClass)) )
	{
	    Error(BADCOSTCLASS, Name, "");
	}

	if ( ! ReadName(Cf, Name, 1000, ':') ||
	     sscanf(Name, "%f", &Val) != 1 || Val < 0 )
	{
	    Error(BADCOST, "", "");
	    Val = 1;
	}

	if ( Pred > 0 && Real > 0 && Pred != Real && Val != 1 )
	{
	    /*  Have a non-trivial cost entry  */

	    if ( ! MCost )
	    {
		/*  Set up cost matrices  */

		MCost = Alloc(MaxClass+1, double *);
		ForEach(p, 1, MaxClass)
		{
		    MCost[p] = Alloc(MaxClass+1, double);
		    ForEach(r, 1, MaxClass)
		    {
			MCost[p][r] = ( p == r ? 0.0 : 1.0 );
		    }
		}
	    }

	    MCost[Pred][Real] = Val;
	}
    }
    fclose(Cf);
}



/*************************************************************************/
/*                                                              	 */
/*	Categorize a case description using the given decision tree	 */
/*                                                              	 */
/*************************************************************************/

/*
ClassNo TreeClassify(Description CaseDesc, Tree DecisionTree)
{
    ClassNo	c;

    ForEach(c, 0, MaxClass)
    {
	ClassSum[c] = 0;
    }

    FindLeaf(CaseDesc, DecisionTree, Nil, 1.0);

    return SelectClass(1, (Boolean)(MCost != Nil));
}
*/


/*************************************************************************/
/*                                                              	 */
/*	Classify a case description using the given subtree by		 */
/*	adjusting the value ClassSum for each class			 */
/*                                                              	 */
/*************************************************************************/


/*
void FollowAllBranches(Description CaseDesc, Tree T, double Fraction)
{
    DiscrValue	v;

    ForEach(v, 1, T->Forks)
    {
	FindLeaf(CaseDesc, T->Branch[v], T,
		 (Fraction * T->Branch[v]->Items) / T->Items);
    }
}

*/

/*************************************************************************/
/*                                                              	 */
/*	Classify a case description using the given subtree by		 */
/*	adjusting the value ClassSum for each class			 */
/*                                                              	 */
/*************************************************************************/
/*

void FindLeaf(Description CaseDesc, Tree T, Tree PT, double Fraction)
{
    DiscrValue	v, Dv;
    ClassNo	c;
    double	NewFrac, BrWt[4];


    switch ( T->NodeType )
    {
	case 0:  // leaf 

	  LeafUpdate:

	    //  Use parent node if effectively no cases at this node  

	    if ( T->Items < Epsilon )
	    {
		T = PT;
	    }

	    //  Update from all classes.  Adjust probabilities by the Laplace correction  

	    ForEach(c, 1, MaxClass)
	    {
		if ( T->ClassDist[c] )
		{
		    ClassSum[c] += Fraction * (T->ClassDist[c] + 1) /
				   (T->Items + MaxClass);
		}
	    }

	    return;

	case BrDiscr:  // test of discrete attribute 

	    Dv = DVal(CaseDesc, T->Tested);	// > MaxAttVal if unknown 

	    if ( Dv <= T->Forks )	//  Make sure not new discrete value  
	    {
		FindLeaf(CaseDesc, T->Branch[Dv], T, Fraction);
	    }
	    else
	    {
		FollowAllBranches(CaseDesc, T, Fraction);
	    }

	    return;

	case BrThresh:  // test of continuous attribute //

	    if ( Unknown(CaseDesc, T->Tested) )
	    {
		FollowAllBranches(CaseDesc, T, Fraction);
	    }
	    else
	    if ( NotApplic(CaseDesc, T->Tested) )
	    {
		FindLeaf(CaseDesc, T->Branch[1], T, Fraction);
	    }
	    else
	    {
		//  Find weights for <= and > branches, interpolating if   probabilistic thresholds are used  

		BrWt[2] = Interpolate(T, CVal(CaseDesc, T->Tested));
		BrWt[3] = 1 - BrWt[2];

		ForEach(v, 2, 3)
		{
		    if ( (NewFrac = Fraction * BrWt[v]) >= 0.01 )
		    {
			FindLeaf(CaseDesc, T->Branch[v], T, NewFrac);
		    }
		}
	    }

	    return;

	case BrSubset:  // subset test on discrete attribute  

	    Dv = DVal(CaseDesc, T->Tested);	// > MaxAttVal if unknown 

	    if ( Dv <= MaxAttVal[T->Tested] )
	    {
		ForEach(v, 1, T->Forks)
		{
		    if ( In(Dv, T->Subset[v]) )
		    {
			FindLeaf(CaseDesc, T->Branch[v], T, Fraction);

			return;
		    }
		}

		// Value not found in any subset -- treat as leaf  

		goto LeafUpdate;
	    }
	    else
	    {
		FollowAllBranches(CaseDesc, T, Fraction);
	    }
    }
}
*/


/*************************************************************************/
/*                                                              	 */
/*	Categorize a case description using a ruleset			 */
/*                                                              	 */
/*************************************************************************/

/*
ClassNo RuleClassify(Description CaseDesc, CRuleSet RS)
{
    ClassNo	c;
    double	TotWeight=0;
    int		r;
    CRule	R;

    ForEach(c, 1, MaxClass)
    {
	ClassSum[c] = 0;
    }

    ForEach(r, 1, RS->SNRules)
    {
	R = RS->SRule[r];

	if ( Matches(R, CaseDesc) )
	{
	    ClassSum[R->Rhs] += R->Vote;
	    TotWeight += 1000.0;
	}
    }

    //  Normalise ClassSum and add default if required  

    if ( ! TotWeight )
    {
	ClassSum[RS->SDefault] = 0.5;
    }
    else
    {
	ForEach(c, 1, MaxClass)
	{
	    ClassSum[c] /= TotWeight;
	}
    }

    return SelectClass(RS->SDefault, (Boolean)(MCost != Nil));
}

*/

/*************************************************************************/
/*									 */
/*	Determine whether the given case description satisfies the	 */
/*	given condition							 */
/*									 */
/*************************************************************************/

/*
Boolean Satisfies(Description CaseDesc, Condition OneCond)
{
    DiscrValue	v;
    DiscrValue	Outcome;
    Attribute	Att;

    Att = OneCond->Tested;


    //  Determine the outcome of this test on this item  

    switch ( OneCond->NodeType )
    {
	case BrDiscr:  // test of discrete attribute 

	    v = XDVal(CaseDesc, Att);
	    Outcome = ( v == 0 ? -1 : v );
	    break;

	case BrThresh:  // test of continuous attribute 

	    Outcome = ( Unknown(CaseDesc, Att) ? -1 :
			NotApplic(CaseDesc, Att) ? 1 :
			CVal(CaseDesc, Att) <= OneCond->Cut ? 2 : 3 );
	    break;

	case BrSubset:  // subset test on discrete attribute  

	    v = XDVal(CaseDesc, Att);
	    Outcome = ( v <= MaxAttVal[Att] && In(v, OneCond->Subset) ?
			OneCond->TestValue : 0 );
    }

    return ( Outcome == OneCond->TestValue );
}

*/

/*************************************************************************/
/*									 */
/*	Determine whether a case satisfies all conditions of a rule	 */
/*									 */
/*************************************************************************/

/*
Boolean Matches(CRule R, Description Case)
{
    int d;

    ForEach(d, 1, R->Size)
    {
	if ( ! Satisfies(Case, R->Lhs[d]) )
	{
	    return false;
	}
    }

    return true;
}
*/


/*************************************************************************/
/*									 */
/*	Classify a case using boosted tree or rule sequence		 */
/*									 */
/*************************************************************************/

/*
ClassNo BoostClassify(Description CaseDesc, int MaxTrial, Boolean UseRules)
{
    ClassNo	c, Best;
    int		t;
    double	Total=0;

    ForEach(c, 0, MaxClass)
    {
	Vote[c] = 0;
    }

    ForEach(t, 0, MaxTrial)
    {
	Best = ( UseRules ? RuleClassify(CaseDesc, RuleSet[t]) :
			    TreeClassify(CaseDesc, Pruned[t]) );
	Vote[Best] += Confidence;
	Total += Confidence;
    }

    //  Copy votes into ClassSum  

    ForEach(c, 0, MaxClass)
    {
	ClassSum[c] = Vote[c] / Total;
    }

    return (ClassNo)(SelectClass((ClassNo)KRandom() * MaxClass + 1, (Boolean)(MCost != Nil)));
}
*/


/*************************************************************************/
/*									 */
/*	Select the best class to return.  Take misclassification costs	 */
/*	into account if they are defined.				 */
/*									 */
/*************************************************************************/

/*
ClassNo SelectClass(ClassNo Default, Boolean UseCosts)
{
    ClassNo	c, cc, BestClass;
    double	ExpCost, BestCost=1E38;

    BestClass = Default;
    ForEach(c, 1, MaxClass)
    {
	if ( UseCosts )
	{
	    ExpCost = 0;
	    ForEach(cc, 1, MaxClass)
	    {
		if ( cc == c ) continue;
		ExpCost += ClassSum[cc] * MCost[c][cc];
	    }

	    if ( ExpCost < BestCost )
	    {
		BestClass = c;
		BestCost  = ExpCost;
	    }
	}
	else
	{
	    if ( ClassSum[c] > ClassSum[BestClass] ) BestClass = c;
	}
    }

    Confidence = ClassSum[BestClass];

    return BestClass;
}

*/

/*************************************************************************/
/*								   	 */
/*	General classification routine					 */
/*								   	 */
/*************************************************************************/

/*
ClassNo Classify(Description Case)
{
    return ( TRIALS > 1 ? BoostClassify(Case, TRIALS-1, RULES) :
	     RULES ?	  RuleClassify(Case, RuleSet[0]) :
			  TreeClassify(Case, Pruned[0]) );
}

*/

/*************************************************************************/
/*								   	 */
/*	Interpolate a single value between Lower, Cut and Upper		 */
/*								   	 */
/*************************************************************************/

/*
double Interpolate(Tree T, ContValue Val)
{
    return ( Val <= T->Lower ? 1.0 :
	     Val >= T->Upper ? 0.0 :
	     Val <= T->Mid ?
		1 - 0.5 * (Val - T->Lower) / (T->Mid - T->Lower + 1E-6) :
		0.5 - 0.5 * (Val - T->Mid) / (T->Upper - T->Mid + 1E-6) );
}

*/

/*************************************************************************/
/*									 */
/*	Open file with given extension for read/write			 */
/*									 */
/*************************************************************************/


FILE *GetFile(String Extension, String RW)
/*    --------  */
{
    strcpy(Fn, FileStem);
    strcat(Fn, Extension);
    return fopen(Fn, RW);
}



/*************************************************************************/
/*									 */
/*	Check whether file is open.  If it is not, open it and		 */
/*	read/write discrete names					 */
/*									 */
/*************************************************************************/

/*
void CheckFile(String Extension, Boolean Write)
{
    static char	*LastExt="";

    if ( ! TRf || strcmp(LastExt, Extension) )
    {
	LastExt = Extension;

	if ( TRf )
	{
	    fprintf(TRf, "\n");
	    fclose(TRf);
	}

	ReadFilePrefix(Extension);
    }
}
*/


/*************************************************************************/
/*									 */
/*	Specialised form of the getopt() utility			 */
/*									 */
/*************************************************************************/
/*
String	OptArg, Option;


char ProcessOption(int Argc, char *Argv[], char *Options)
{
    int		i;
    static int	OptNo=1;

    if ( OptNo >= Argc ) return '\00';

    if ( *(Option = Argv[OptNo++]) != '-' ) return '?';

    for ( i = 0 ; Options[i] ; i++ )
    {
	if ( Options[i] == Option[1] )
	{
	    OptArg = ( Options[i+1] != '+' ? Nil :
		       Option[2] ? Option+2 :
		       OptNo < Argc ? Argv[OptNo++] : "0" );
	    return Option[1];
	}
    }

    return '?';
}
*/


/*************************************************************************/
/*									 */
/*	Protected memory allocation routines				 */
/*									 */
/*************************************************************************/



void *Pmalloc(unsigned Bytes)
/*    -------  */
{
    void *p=Nil;

    if ( ! Bytes || (p = (void *) malloc(Bytes)) )
    {
	return p;
    }

    Error(NOMEM, "", "");


    return Nil;
}



void *Prealloc(void *Present, unsigned Bytes)
/*    --------  */
{
    void *p=Nil;

    if ( ! Bytes ) return Nil;

    if ( ! Present ) return Pmalloc(Bytes);

    if ( (p = (void *) realloc(Present, Bytes)) )
    {
	return p;
    }

    Error(NOMEM, "", "");

//#ifdef WIN32
    return Nil;
//#endif
}



void *Pcalloc(unsigned Number, unsigned Size)
/*    -------  */
{
    void *p=Nil;

    if ( ! Number || (p = (void *) calloc(Number, Size)) )
    {
	return p;
    }

    Error(NOMEM, "", "");

//#ifdef WIN32
    return Nil;
//#endif
}



/*************************************************************************/
/*									 */
/*	Generate uniform random numbers					 */
/*									 */
/*************************************************************************/


#define	Modify(F,S)	if ( (F -= S) < 0 ) F += 1.0

int	KRFp=0, KRSp=0;

double KRandom()
/*     -------  */
{
    static double	URD[55];
    double		V1, V2;
    int			i, j;

    /*  Initialisation  */

    if ( KRFp == KRSp )
    {
	KRFp = 0;
	KRSp = 31;

	V1 = 1.0;
	V2 = 0.314159285;

	ForEach(i, 1, 55)
	{
	    URD[ j = (i * 21) % 55 ] = V1;
	    V1 = V2 - V1;
	    if ( V1 < 0 ) V1 += 1.0;
	    V2 = URD[j];
	}

	ForEach(j, 0, 5)
	{
	    ForEach(i, 0, 54)
	    {
		Modify(URD[i], URD[(i+30) % 55]);
	    }
	}
    }

    KRFp = (KRFp + 1) % 55;
    KRSp = (KRSp + 1) % 55;
    Modify(URD[KRFp], URD[KRSp]);

    return URD[KRFp];
}



void ResetKR(int KRInit)
/*   -------  */
{
    KRFp = KRSp = 0;

    while ( KRInit-- )
    {
	KRandom();
    }
}



/*************************************************************************/
/*									 */
/*	Error messages							 */
/*									 */
/*************************************************************************/


void Error(int ErrNo, String S1, String S2)
/*   -----  */
{
    Boolean	Quit=false;

//#ifdef WIN32
    char	Msg[200];

    if ( ErrNo == NOMEM )
    {
	   fprintf(stderr,"Cannot allocate sufficient memory. Fatal Error") ;
	Goodbye(1);
    }
    else
    if ( ErrNo == MODELFILE )
    {
	sprintf(Msg, "File %s is incompatible with .names file\n(%s `%s')",
		     Fn, S1, S2);
	fprintf(stderr, Msg) ;
	ErrMsgs++;
	return;
    }
//#endif

    if ( ! Of ) return;

    if ( ErrNo == NOFILE || ErrNo == NOMEM || ErrNo == MODELFILE )
    {
	fprintf(Of, "\n*** ");
    }
    else
    {
	fprintf(Of, TX_Line(LineNo, Fn));
    }

    switch ( ErrNo )
    {
	case NOFILE:
	    fprintf(Of, E_NOFILE(Fn, S2));
	    Quit = true;
	    break;

	case BADATTNAME:
	    fprintf(Of, E_BADATTNAME, S1);
	    break;

	case EOFINATT:
	    fprintf(Of, E_EOFINATT, S1);
	    break;

	case SINGLEATTVAL:
	    fprintf(Of, E_SINGLEATTVAL(S1, S2));
	    break;

	case DUPATTNAME:
	    fprintf(Of, E_DUPATTNAME, S1);
	    break;

	case BADATTVAL:
	    fprintf(Of, E_BADATTVAL(S2, S1));
	    break;

	case BADCLASS:
	    fprintf(Of, E_BADCLASS, S2);
	    break;

	case BADCOSTCLASS:
	    fprintf(Of, E_BADCOSTCLASS, S1);
	    Quit = true;
	    break;

	case BADCOST:
	    fprintf(Of, E_BADCOST, S1);
	    Quit = true;
	    break;

	case NOMEM:
	    fprintf(Of, E_NOMEM);
	    Quit = true;
	    break;

	case TOOMANYVALS:
	    fprintf(Of, E_TOOMANYVALS(S1, (int) S2));
	    Quit = true;
	    break;

	case BADDISCRETE:
	    fprintf(Of, E_BADDISCRETE, S1);
	    Quit = true;
	    break;

	case NOTARGET:
	    fprintf(Of, E_NOTARGET, S1);
	    Quit = true;
	    break;

	case BADTARGET:
	    fprintf(Of, E_BADTARGET, S1);
	    Quit = true;
	    break;

	case LONGNAME:
	    fprintf(Of, E_LONGNAME);
	    Quit = true;
	    break;

	case HITEOF:
	    fprintf(Of, E_HITEOF);
	    break;

	case MISSNAME:
	    fprintf(Of, E_MISSNAME, S2);
	    break;

	case BADTSTMP:
	    fprintf(Of, E_BADTSTMP(S2, S1));
	    break;

	case BADDATE:
	    fprintf(Of, E_BADDATE(S2, S1));
	    break;

	case BADTIME:
	    fprintf(Of, E_BADTIME(S2, S1));
	    break;

	case UNKNOWNATT:
	    fprintf(Of, E_UNKNOWNATT, S1);
	    break;

	case BADDEF1:
	    fprintf(Of, E_BADDEF1(AttName[MaxAtt], S1, S2));
	    break;

	case BADDEF2:
	    fprintf(Of, E_BADDEF2(AttName[MaxAtt], S1, S2));
	    break;

	case SAMEATT:
	    fprintf(Of, E_SAMEATT(AttName[MaxAtt], S1));
	    break;

	case BADDEF3:
	    fprintf(Of, E_BADDEF3, AttName[MaxAtt]);
	    break;

	case MODELFILE:
	    fprintf(Of, EX_MODELFILE(Fn));
	    fprintf(Of, "    (%s `%s')\n", S1, S2);
	    Quit = true;
	    break;
    }

    if ( ++ErrMsgs > 10 )
    {
	fprintf(Of, T_ErrorLimit);
	Quit = true;
    }

    if ( Quit )
    {
	Goodbye(1);
    }
}



/*************************************************************************/
/*									 */
/*	Determine precision of doubleing value				 */
/*									 */
/*************************************************************************/


int Denominator(ContValue Val)
/*  -----------  */
{
    double	RoundErr, Accuracy;
    int		Mult;

    Accuracy = fabs(Val) * 1E-6;	/* approximate */
    Val = modf(Val, &RoundErr);

    for ( Mult = 100000 ; Mult >= 1 ; Mult /= 10 )
    {
	RoundErr = fabs(rint(Val * Mult) / Mult - Val);
	if ( RoundErr > 2 * Accuracy )
	{
	    return Mult * 10;
	}
    }

    return 1;
}



/*************************************************************************/
/*									 */
/*	Routines to process dates (Algorithm due to Gauss?,		 */
/*	times, and timestamps						 */
/*									 */
/*************************************************************************/


int GetInt(String S, int N)
/*  ------  */
{
    int	Result=0;

    while ( N-- )
    {
	if ( ! isdigit(*S) ) return 0;

	Result = Result * 10 + (*S++ - '0');
    }

    return Result;
}



int DateToDay(String DS)	/*  Day 1 is 0000/03/01  */
/*  ---------  */
{
    int Year, Month, Day;

    if ( strlen(DS) != 10 ) return 0;

    Year  = GetInt(DS, 4);
    Month = GetInt(DS+5, 2);
    Day   = GetInt(DS+8, 2);

    if ( ! ( DS[4] == '/' && DS[7] == '/' || DS[4] == '-' && DS[7] == '-' ) ||
	 Year < 0 || Month < 1 || Day < 1 ||
	 Month > 12 ||
	 Day > 31 ||
	 Day > 30 &&
	    ( Month == 4 || Month == 6 || Month == 9 || Month == 11 ) ||
	 Month == 2 &&
	    ( Day > 29 ||
	      Day > 28 && ( Year % 4 != 0 ||
			    Year % 100 == 0 && Year % 400 != 0 ) ) )
    {
	return 0;
    }

    if ( (Month -= 2) <= 0 )
    {
	Month += 12;
	Year -= 1;
    }

    return Year * 365 + Year / 4 - Year / 100 + Year / 400
	   + 367 * Month / 12
	   + Day - 30;
}



int TimeToSecs(String TS)
/*  ----------  */
{
    int Hour, Mins, Secs;

    if ( strlen(TS) != 8 ) return -1;

    Hour = GetInt(TS, 2);
    Mins = GetInt(TS+3, 2);
    Secs = GetInt(TS+6, 2);

    if ( TS[2] != ':' || TS[5] != ':' ||
	 Hour >= 24 || Mins >= 60 || Secs >= 60 )
    {
	return -1;
    }

    return Hour * 3600 + Mins * 60 + Secs;
}



void SetTSBase(int y)
/*   ---------  */
{
    y -= 15;
    TSBase = y * 365 + y / 4 - y / 100 + y / 400 + (367 * 4) / 12 + 1 - 30;
}



int TStampToMins(String TS)
/*  ------------  */
{
    int		Day, Sec, i, j=11;

    /*  Check for reasonable length and space between date and time  */

    if ( strlen(TS) < 19 || ! Space(TS[10]) ) return (1 << 30);

    /*  Read date part  */

    TS[10] = '\00';
    Day = DateToDay(TS);
    TS[10] = ' ';

    /*  Skip one or more spaces  */

    for ( i = 11 ; TS[i] && Space(TS[i]) ; i++ )
	;

    /*  Read time part  */

    Sec = TimeToSecs(TS+i);

    /*  Return a long time in the future if there is an error  */

    return ( Day < 1 || Sec < 0 ? (1 << 30) :
	     (Day - TSBase) * 1440 + (Sec + 30) / 60 );
}



/*************************************************************************/
/*									 */
/*	Free case description space					 */
/*									 */
/*************************************************************************/


void FreeLastCase(Description DVec)
/*   ------------  */
{
    Attribute   Att;

    ForEach(Att, 1, MaxAtt)
    {
	if ( Exclude(Att) )
	{
	    FreeUnlessNil(SVal(DVec,Att));
	}
    }
    // free(DVec);

    free(&DVec[-1]);
}



/*************************************************************************/
/*									 */
/*	Deallocate the space used to perform classification		 */
/*									 */
/*************************************************************************/


void FreeGlobals()
/*   -----------  */
{
    /*  Free memory allocated for classifier  */
/*
    if ( RULES )
    {
	ForEach(Trial, 0, TRIALS-1)
	{
	     FreeRules(RuleSet[Trial]);
	}
	free(RuleSet);
    }
    else
    {
	ForEach(Trial, 0, TRIALS-1)
	{
	     FreeTree(Pruned[Trial]);
	}
	free(Pruned);
    }
*/
    /*  Free memory allocated for cost matrix  */

    if ( MCost )
    {
        FreeVector((void **) MCost, 1, MaxClass);
    }

    /*  Free memory for names etc  */

    FreeNames();

//    free(ClassSum);
//    free(Vote);

//    FreeUnlessNil(PropVal);
}



/*************************************************************************/
/*									 */
/*	Free up all space allocated by GetNames()			 */
/*									 */
/*************************************************************************/


void FreeNames()
/*   ---------  */
{
    Attribute a, t;

    ForEach(a, 1, MaxAtt)
    {
	if ( a != ClassAtt && Discrete(a) )
	{
	    FreeVector((void **) AttValName[a], 1, MaxAttVal[a]);
	}
    }
    FreeUnlessNil(AttValName);				AttValName = Nil;
    FreeUnlessNil(MaxAttVal);				MaxAttVal = Nil;
    FreeVector((void **) AttName, 1, MaxAtt);		AttName = Nil;
    FreeVector((void **) ClassName, 1, MaxClass);	ClassName = Nil;

    FreeUnlessNil(SpecialStatus);			SpecialStatus = Nil;

    /*  Definitions (if any)  */

    if ( AttDef )
    {
	ForEach(a, 1, MaxAtt)
	{
	    if ( AttDef[a] )
	    {
		for ( t = 0 ; DefOp(AttDef[a][t]) != OP_END ; t++ )
		{
		    if ( DefOp(AttDef[a][t]) == OP_STR )
		    {
			free(DefSVal(AttDef[a][t]));
		    }
		}

		free(AttDef[a]);
	    }
	}
	free(AttDef);					AttDef = Nil;
    }
}



/*************************************************************************/
/*									 */
/*	Free up space taken up by tree Node				 */
/*									 */
/*************************************************************************/

/*
void FreeTree(Tree T)
{
    DiscrValue v;

    if ( ! T ) return;

    if ( T->NodeType )
    {
	ForEach(v, 1, T->Forks)
	{
	    FreeTree(T->Branch[v]);
	}

	free(T->Branch);

	if ( T->NodeType == BrSubset )
	{
	    FreeVector((void **) T->Subset, 1, T->Forks);
	}

    }

    free(T->ClassDist);
    free(T);
}
*/


/*************************************************************************/
/*									 */
/*	Deallocate the space used to store rules			 */
/*									 */
/*************************************************************************/

/*
void FreeRule(CRule R)
{
    int	d;

    ForEach(d, 1, R->Size)
    {
	if ( R->Lhs[d]->NodeType == BrSubset )
	{
	    FreeUnlessNil(R->Lhs[d]->Subset);
	}
	FreeUnlessNil(R->Lhs[d]);
    }
    FreeUnlessNil(R->Lhs);
    FreeUnlessNil(R);
}
*/

/*
void FreeRules(CRuleSet RS)
{
    int	ri;

    ForEach(ri, 1, RS->SNRules)
    {
	FreeRule(RS->SRule[ri]);
    }
    free(RS->SRule);
    free(RS);
}
*/


void FreeVector(void **V, int First, int Last)
/*   ----------  */
{
    if ( V )
    {
	while ( First <= Last )
	{
	    FreeUnlessNil(V[First]);
	    First++;
	}

	free(V);
    }
}



void GetData(FILE *Df, Boolean Train)
/*   -------  */
{
    ItemNo	ItemSpace;
    Description	DVec;

    LineNo = 1;

    if ( Train || ! Item )
    {
	MaxItem =  ItemSpace = 0;
	Item = Alloc(1, Description);	/* for error reporting */
    }
    else
    {
	ItemSpace = MaxItem + 1;
	MaxItem++;
    }


    while ( (DVec = GetDescription(Df, Train)) )
    {

	/*  Make sure there is room for another item  */

	if ( MaxItem >= ItemSpace )
	{
	    ItemSpace += Inc;
	    Realloc(Item, ItemSpace+1, Description);
	}

	Item[MaxItem] = DVec;

	MaxItem++;
    }

    fclose(Df);
    MaxItem--;

}



/*************************************************************************/
/*									 */
/*	Free case description space					 */
/*									 */
/*************************************************************************/


void FreeCases(Description *Case, ItemNo MaxCase)
/*   --------  */
{
    ItemNo      i;

    /*  Release any strings holding ignored attribute values  */

    ForEach(i, 0, MaxCase)
    {
	FreeCase(Case[i]);
    }

    free(Case);
}



void FreeCase(Description DVec)
/*   --------  */
{
    Attribute	Att;

    ForEach(Att, 1, MaxAtt)
    {
	if ( Exclude(Att) )
	{
	    FreeUnlessNil(SVal(DVec, Att));
	}
    }

    free(DVec);
}
