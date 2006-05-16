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


#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>


/*************************************************************************/
/*									 */
/*		Constants, macros etc.					 */
/*									 */
/*************************************************************************/


#define	 SEE5
#define	 PUBLIC

#define	 Nil	   0		/* null pointer */
#define	 false	   0
#define	 true	   1
#define	 None	   -1 
#define	 Epsilon   1E-4

#define  EXCLUDE   1		/* special attribute status: do not use */
#define  SKIP	   2		/* do not use in classifiers */
#define  DISCRETE  4		/* ditto: collect values as data read */
#define  ORDERED   8		/* ditto: ordered discrete values */
#define  DATEVAL   16		/* ditto: YYYY/MM/DD or YYYY-MM-DD */
#define  STIMEVAL  32		/* ditto: HH:MM:SS */
#define	 TSTMPVAL  64		/* date time */

				/* unknown and N/A values are represented by
				   unlikely doubleing-point numbers
				   (octal 01600000000 and 01) */
#define	 UNKNOWN   01600000000	/* 1.5777218104420236e-30 */
#define	 NA	   01		/* 1.4012984643248171e-45 */

#define	 BrDiscr   1
#define	 BrThresh  2
#define	 BrSubset  3

#define  Alloc(N,T)		(T *) Pmalloc((N)*sizeof(T))
#define  AllocZero(N,T)		(T *) Pcalloc(N, sizeof(T))
#define  Realloc(V,N,T)		V = (T *) Prealloc(V, (N)*sizeof(T))

#define	 Bit(b)			(1 << (b))
#define	 In(b,s)		((s[(b) >> 3]) & Bit((b) & 07))
#define	 SetBit(b,s)		(s[(b) >> 3] |= Bit((b) & 07))

#define	 ForEach(v,f,l)		for(v=f ; v<=l ; ++v) 

#define	 StatBit(a,b)		(SpecialStatus[a]&(b))
#define	 Exclude(a)		StatBit(a,EXCLUDE)
#define	 Skip(a)		StatBit(a,EXCLUDE|SKIP)
#define  Discrete(a)		(MaxAttVal[a] || StatBit(a,DISCRETE))
#define  Continuous(a)		(! MaxAttVal[a] && ! StatBit(a,DISCRETE))
#define	 Ordered(a)		StatBit(a,ORDERED)
#define	 DateVal(a)		StatBit(a,DATEVAL)
#define	 TimeVal(a)		StatBit(a,STIMEVAL)
#define	 TStampVal(a)		StatBit(a,TSTMPVAL)

#define  Space(s)		(s==' ' || s=='\n' || s=='\r' || s=='\t')
#define  SkipComment		while ( ( c = InChar(f) ) != '\n' && c != EOF )

#define	 FreeUnlessNil(p)	if((p)!=Nil) free(p)

#define  rint(x)		((x)<0 ? (double)((int)((x)-0.5)) :\
					 (double)((int)((x)+0.5)))

#define	 P1(x)		(rint((x)*10) / 10)
#define	 Of		stdout
#define	 Goodbye(x)	exit(x)
#define	 CharWidth(s)	((int) strlen(s))


#define	 NOFILE		 0
#define	 BADATTNAME	 1
#define	 EOFINATT	 2
#define	 SINGLEATTVAL	 3
#define	 BADATTVAL	 4
#define	 BADCLASS	 5
#define	 BADCOSTCLASS	 6
#define	 BADCOST	 7
#define	 NOMEM		 8
#define	 TOOMANYVALS	 9
#define	 BADDISCRETE	10
#define	 NOTARGET	11
#define	 BADTARGET	12
#define	 LONGNAME	13
#define	 HITEOF		14
#define	 MISSNAME	15
#define	 BADDATE	16
#define	 BADTIME	17
#define	 BADTSTMP	24
#define	 DUPATTNAME	18
#define	 UNKNOWNATT	19
#define	 BADDEF1	20
#define	 BADDEF2	21
#define	 BADDEF3	22
#define	 SAMEATT	23
#define	 MODELFILE	30


/*************************************************************************/
/*									 */
/*		Type definitions					 */
/*									 */
/*************************************************************************/


typedef  unsigned char	Boolean, BranchType, *Set;
typedef	 char		*String;

typedef  int	ItemNo;			/* data item number */
typedef  double	ItemCount;		/* count of (partial) items */

typedef  int	ClassNo,		/* class number, 1..MaxClass */
		DiscrValue,		/* discrete attribute value (0 = ?) */
		Attribute;		/* attribute number, 1..MaxAtt */

typedef	 double	ContValue;	/* continuous attribute value */
#define	 PREC	14		/* precision */


typedef  union	 _def_val
	 {
	    String	_s_val;		/* att val for comparison */
	    ContValue	_n_val;		/* number for arith */
	 }
	 DefVal;

typedef  struct  _def_elt
	 {
	    short	_op_code;	/* type of element */
	    DefVal	_operand;	/* string or numeric value */
	 }
	 DefElt, *Definition;

typedef  struct  _elt_rec
	 {
	    int		Fi,		/* index of first char of element */
			Li;		/* last ditto */
	    char	Type;		/* 'B', 'S', or 'N' */
	 }
	 EltRec;

#define	 DefOp(DE)	DE._op_code
#define	 DefSVal(DE)	DE._operand._s_val
#define	 DefNVal(DE)	DE._operand._n_val

#define	 OP_ATT			 0	/* opcodes */
#define	 OP_NUM			 1
#define	 OP_STR			 2
#define	 OP_MISS		 3
#define	 OP_AND			10
#define	 OP_OR			11
#define	 OP_EQ			20
#define	 OP_NE			21
#define	 OP_GT			22
#define	 OP_GE			23
#define	 OP_LT			24
#define	 OP_LE			25
#define	 OP_SEQ			26
#define	 OP_SNE			27
#define	 OP_PLUS		30
#define	 OP_MINUS		31
#define	 OP_UMINUS		32
#define	 OP_MULT		33
#define	 OP_DIV			34
#define	 OP_MOD			35
#define	 OP_POW			36
#define	 OP_SIN			40
#define	 OP_COS			41
#define	 OP_TAN			42
#define	 OP_LOG			43
#define	 OP_EXP			44
#define	 OP_INT			45
#define	 OP_END			99


typedef  union  _attribute_value
	 {
	    DiscrValue	_discr_val;
	    ContValue	_cont_val;
	    String	_string_val;
	 }
	 AttValue, *Description;

#define  CVal(Case,Attribute)   Case[Attribute]._cont_val
#define  DVal(Case,Attribute)   Case[Attribute]._discr_val
#define  XDVal(Case,Att)	(Case[Att]._discr_val & 077777777)
#define  SVal(Case,Attribute)   Case[Attribute]._string_val
#define  Class(Case)		(*Case)._discr_val
#define  Weight(Case)		(*(Case-1))._cont_val

#define	 Unknown(Case,Att)	(DVal(Case,Att)==UNKNOWN)
#define	 UnknownVal(AV)		(AV._discr_val==UNKNOWN)
#define	 NotApplic(Case,Att)	(DVal(Case,Att)==NA)
#define	 NotApplicVal(AV)	(AV._discr_val==NA)


typedef  struct _treerec	*Tree;
typedef  struct _treerec
	 {
	    BranchType	NodeType;
	    ClassNo	Leaf;		/* best class at this node */
	    ItemCount	Items,		/* no of items at this node */
			*ClassDist,	/* class distribution of items */
	    		Errors;		/* no of errors at this node */
	    Attribute	Tested; 	/* attribute referenced in test */
	    int		Forks;		/* number of branches at this node */
	    ContValue	Cut,		/* threshold for continuous attribute */
		  	Lower,		/* lower limit of soft threshold */
		  	Upper,		/* upper limit ditto */
			Mid;		/* 50% point */
	    Set         *Subset;	/* subsets of discrete values  */
	    Tree	*Branch;	/* Branch[x] = subtree for outcome x */
	 }
	 TreeRec;


typedef  int	RuleNo;			/* rule number */

typedef  struct _condrec
	 {
	    BranchType	NodeType;	/* test type (see tree nodes) */
	    Attribute	Tested;		/* attribute tested */
	    int		Forks;		/* possible branches */
	    ContValue	Cut;		/* threshold (if relevant) */
	    Set		Subset;		/* subset (if relevant) */
	    int		TestValue;	/* specified outcome of test */
	 }
	 CondRec, *Condition;


typedef  struct _rulerec
	 {
	    RuleNo	RNo;		/* rule number */
	    int		TNo,		/* trial number */
	    		Size;		/* number of conditions */
	    Condition	*Lhs;		/* conditions themselves */
	    ClassNo	Rhs;		/* class given by rule */
	    ItemCount	Cover,		/* number of cases covered by rule */
			Correct;	/* number on which correct */
	    double	Prior;		/* prior probability of RHS */
	    int		Vote;		/* unit = 0.001 */
	 }
	 RuleRec, *CRule;


typedef struct _rulesetrec
	 {
	    RuleNo	SNRules;	/* number of rules */
	    CRule	*SRule;		/* rules */
	    ClassNo	SDefault;	/* default class for this ruleset */
	 }
	 RuleSetRec, *CRuleSet;


/*************************************************************************/
/*									 */
/*		Function prototypes					 */
/*									 */
/*************************************************************************/

Boolean	    ReadName(FILE *f, String s, int n, char ColonOpt);
void	    GetNames(FILE *Nf);
void	    ExplicitAtt(FILE *Nf);
int	    Which(String Val, String *List, int First, int Last);
String	    CopyString(String S);
int	    InChar(FILE *f);

Description GetDescription(FILE *Df, Boolean Train);

void	    ImplicitAtt(FILE *Nf);
void	    ReadDefinition(FILE *f);
void	    Append(char c);
Boolean	    Expression();
Boolean	    Conjunct();
Boolean	    SExpression();
Boolean	    AExpression();
Boolean	    Term();
Boolean	    Factor();
Boolean	    Primary();
Boolean	    Atom();
Boolean	    Find(String S);
int	    FindOne(String *Alt);
Attribute   FindAttName();
void	    DefSyntaxError(String Msg);
void	    DefSemanticsError(int Fi, String Msg, int OpCode);
void	    Dump(char OpCode, ContValue F, String S, int Fi);
void	    DumpOp(char OpCode, int Fi);
Boolean	    UpdateTStack(char OpCode, ContValue F, String S, int Fi);
AttValue    EvaluateDef(Definition D, Description Case);

/*
void	    ReadFilePrefix(String Extension);
void	    ReadHeader();
Tree	    GetTree(String Extension);
Tree	    InTree();
CRuleSet    GetRules(String Extension);
CRuleSet    InRules();
CRule	    InRule();
Condition   InCondition();
int	    ReadProp(char *Delim);
String	    RemoveQuotes(String S);
Set	    MakeSubset(Attribute Att);
void	    BinRecoverDiscreteNames();
Tree	    BinInTree();
CRuleSet    BinInRules();
*/
void	    StreamIn(String S, int n);
// Tree	    Leaf(ItemCount *Freq, ClassNo NodeClass, ItemCount Items, ItemCount Errors);

void	    GetMCosts(FILE *f);
/*
ClassNo	    TreeClassify(Description CaseDesc, Tree DecisionTree);
void	    FollowAllBranches(Description CaseDesc, Tree T, double Fraction);
void	    FindLeaf(Description CaseDesc, Tree T, Tree PT, double Wt);
ClassNo	    RuleClassify(Description CaseDesc, CRuleSet RS);
Boolean	    Satisfies(Description CaseDesc, Condition OneCond);
Boolean	    Matches(CRule R, Description Case);
ClassNo	    BoostClassify(Description CaseDesc, int MaxTrial, Boolean UseRules);
ClassNo	    SelectClass(ClassNo Default, Boolean UseCosts);
ClassNo	    Classify(Description CaseDesc);
double	    Interpolate(Tree T, ContValue Val);

FILE *	    GetFile(String Extension, String RW);
void	    CheckFile(String Extension, Boolean Write);

char	    ProcessOption(int Argc, char *Argv[], char *Options);
*/
void	    *Pmalloc(unsigned Bytes);
void	    *Prealloc(void *Present, unsigned Bytes);
void	    *Pcalloc(unsigned Number, unsigned Size);
double	    KRandom();
void	    Error(int ErrNo, String S1, String S2);
int	    Denominator(ContValue Val);
int	    GetInt(String S, int N);
int	    DateToDay(String DS);
int	    TimeToSecs(String TS);
void	    SetTSBase(int y);
int	    TStampToMins(String TS);

void	    FreeGlobals();
void	    FreeCosts();
void	    FreeNames();
/*
void	    FreeTree(Tree T);
void	    FreeRules(CRuleSet RS);
*/
void	    FreeLastCase(Description DVec);
void	    FreeVector(void **V, int First, int Last);

void GetData(FILE *Df, Boolean Train) ;
void FreeCases(Description *Case, ItemNo MaxCase) ;
void FreeCase(Description DVec) ;

/*************************************************************************/
/*									 */
/*		Text strings						 */
/*									 */
/*************************************************************************/


#define	 TX_Line(l,f)		"\n*** line %d of `%s': ", l, f
#define	 E_NOFILE(f,e)		"cannot open file %s%s\n", f, e
#define	 E_BADATTNAME		"`:' or `:=' expected after attribute name"\
					" `%s'\n"
#define	 E_EOFINATT		"unexpected eof while reading attribute `%s'\n"
#define	 E_SINGLEATTVAL(a,v)	"attribute `%s' has only one value `%s'\n",\
					a, v
#define	 E_DUPATTNAME		"multiple attributes with name `%s'\n"
#define	 E_BADATTVAL(v,a)	"bad value of `%s' for attribute `%s'\n", v, a
#define	 E_BADCLASS		"bad class value `%s'l\n"
#define	 E_BADCOSTCLASS		"bad class `%s'\n"
#define	 E_BADCOST		"bad cost value `%s'\n"
#define	 E_NOMEM		"unable to allocate sufficient memory\n"
#define	 E_TOOMANYVALS(a,n)	"too many values for attribute `%s'"\
					" (max %d)\n", a, n
#define	 E_BADDISCRETE		"bad number of discrete values for attribute"\
					" `%s'\n"
#define	 E_NOTARGET		"target attribute `%s' not found\n"
#define	 E_BADTARGET		"target attribute `%s' must be specified by"\
					" a list of discrete values\n"
#define	 E_LONGNAME		"overlength name: check data file formats\n"
#define	 E_HITEOF		"unexpected end of file\n"
#define	 E_MISSNAME		"missing name or value before `%s'\n"
#define	 E_BADTSTMP(d,a)	"bad timestamp `%s' for attribute `%s'\n", d, a
#define	 E_BADDATE(d,a)		"bad date `%s' for attribute `%s'\n", d, a
#define	 E_BADTIME(d,a)		"bad time `%s' for attribute `%s'\n", d, a
#define	 E_UNKNOWNATT		"unknown attribute name `%s'\n"
#define	 E_BADDEF1(a,s,x)	"in definition of attribute `%s':\n"\
					"\tat `%.12s': expect %s\n", a, s, x
#define	 E_BADDEF2(a,s,x)	"in definition of attribute `%s':\n"\
					"\t`%s': %s\n", a, s, x
#define	 E_SAMEATT(a,b)		"attribute `%s' is identical to attribute"\
					" `%s'\n", a, b
#define	 E_BADDEF3		"cannot define target attribute `%s'\n"
#define	 EX_MODELFILE(f)	"file %s incompatible with .names file\n", f
#define	 E_MFATT		"undefined or excluded attribute"
#define	 E_MFATTVAL		"undefined attribute value"
#define	 E_MFCLASS		"undefined class"
#define	 E_MFEOF		"unexpected eof"
#define	 T_ErrorLimit		"Error limit exceeded\n"
