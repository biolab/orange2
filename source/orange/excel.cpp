#ifdef _MSC_VER

#include "table.hpp"
#include "stringvars.hpp"

#include "filegen.hpp"
#include <list>

#include <direct.h>
#include <ole2.h>

class TExcelDomain : public TDomain
{ public:
    TExcelDomain(PVariable, TVarList);
    ~TExcelDomain();
};


// These functions are wrapped into a class for easier implementation of clean-up (especially at exceptions).
class TExcelReader {
public:
  TExcelReader();
  ~TExcelReader();

  TExampleTable *operator ()(char *filename, char *sheet, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore);

  friend class TExcelDomain;
protected:
  static list<TExcelDomain *> knownDomains;
  static TKnownVariables knownVariables;

  static void removeKnownVariable(TVariable *var);
  static void addKnownDomain(TExcelDomain *domain);

private:
  IDispatch *pXlApp;
  IDispatch *pXlBooks;
  IDispatch *pXlBook;
  IDispatch *pXlSheet;
  IDispatch *pXlUsedRange;

  SAFEARRAY *cells;
  long rowsLow, rowsHigh;
  long columnsLow, columnsHigh;
  long nExamples, nAttrs;

  VARIANT result, args[2];
  DISPPARAMS dp;
  DISPID dispidNamed;
  LPWSTR lfilename;
  LPWSTR lsheet;
  char *cellvalue;

  void Invoke(int autoType, IDispatch *pDisp, LPOLESTR ptName, int cArgs);
  void getProperty(IDispatch *pDisp, LPOLESTR ptName);

  void openFile(char *filename, char *sheet);

  void cellAsVariant(const int &row, const int &col);
  char *cellAsText(const int &row, const int &col);
  int cellType(const int &row, const int &col);

  PVariable TExcelReader::constructAttr(const int &attrNo, int &id, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore);
  PDomain constructDomain(vector<int> &specials, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore);
  TExampleTable *readExamples(PDomain domain, const vector<int> &specials);
  void readValue(const int &row, const int &col, PVariable var, TValue &value);

  void setArg(const int argno, const int arg);
};


TExcelDomain::TExcelDomain(PVariable classVar, TVarList attributes)
: TDomain(classVar, attributes)
{}


TExcelDomain::~TExcelDomain()
{ TExcelReader::knownDomains.remove(this); }


list<TExcelDomain *> TExcelReader::knownDomains;
TKnownVariables TExcelReader::knownVariables;


TExcelReader::TExcelReader()
: pXlApp(NULL),
  pXlBooks(NULL),
  pXlBook(NULL),
  pXlSheet(NULL),
  pXlUsedRange(NULL),
  dispidNamed(DISPID_PROPERTYPUT),
  lfilename(NULL),
  lsheet(NULL),
  cells(NULL),
  cellvalue(NULL)
{
  dp.rgvarg = args;
  dp.rgdispidNamedArgs = &dispidNamed;
  dp.cNamedArgs = 0;

  VariantInit(args);
  VariantInit(args+1);
  VariantInit(&result);
  CoInitialize(NULL);
}


TExcelReader::~TExcelReader()
{
  if (pXlBook) {
    setArg(0, 1);
    Invoke(DISPATCH_PROPERTYPUT, pXlBook, L"Saved", 1);
  }

  if (pXlApp)
    Invoke(DISPATCH_METHOD, pXlApp, L"Quit", 0);

  if (pXlUsedRange)
    pXlUsedRange->Release();
  if (pXlSheet)
    pXlSheet->Release();
  if (pXlBook)
    pXlBook->Release();
  if (pXlBooks)
    pXlBooks->Release();
  if (pXlApp)
    pXlApp->Release();

  if (cells)
    SafeArrayDestroy(cells);

  if (lfilename)
    free(lfilename);
  if (lsheet)
    free(lsheet);
  if (cellvalue)
    free(cellvalue);

  CoUninitialize();
}


void TExcelReader::removeKnownVariable(TVariable *var)
{ knownVariables.remove(var);
  var->destroyNotifier = NULL;
}


void TExcelReader::Invoke(int autoType, IDispatch *pDisp, LPOLESTR ptName, int cArgs)
{
  char name[32];
  WideCharToMultiByte(CP_ACP, 0, ptName, -1, name, 32, NULL, NULL);

  if(!pDisp)
    raiseError("NULL IDispatch passed to AutoWrap()");

  VariantInit(&result); // not variantClear! the previous caller got the result and is responsible for it!
  dp.cArgs = cArgs;
  dp.cNamedArgs = (autoType & DISPATCH_PROPERTYPUT) ? 1 : 0;

  DISPID dispID;

  if (FAILED(pDisp->GetIDsOfNames(IID_NULL, &ptName, 1, LOCALE_USER_DEFAULT, &dispID)))
    raiseError("IDispatch::GetIDsOfNames(\"%s\") failed", name);
  
  if (FAILED(pDisp->Invoke(dispID, IID_NULL, LOCALE_SYSTEM_DEFAULT, autoType, &dp, &result, NULL, NULL)))
    if (strcmp(name, "Open"))
      raiseError("IDispatch::Invoke(\"%s\") failed", name);
    else
      raiseError("File not found (or cannot be opened)");
}


void TExcelReader::getProperty(IDispatch *pDisp, LPOLESTR ptName)
{ Invoke(DISPATCH_PROPERTYGET, pDisp, ptName, 0); }


void TExcelReader::setArg(const int argno, const int arg)
{ args[argno].vt = VT_I4;
  args[argno].lVal = arg;
}


void TExcelReader::openFile(char *filename, char *sheet)
{
  // Get the full path name and the sheet name (if given)
  char fnamebuf[1024], *foo;
  long fulllen;
  if (*sheet) {
    *sheet = 0;
    fulllen = GetFullPathName(filename, 1024, fnamebuf, &foo);

    const int slen = 2+2*strlen(sheet+1);
    lsheet = (LPWSTR)malloc(slen);
    MultiByteToWideChar(CP_ACP, 0, sheet+1, -1, lsheet, slen);

    *sheet = '#';
  }
  else
    fulllen = GetFullPathName(filename, 1024, fnamebuf, &foo);

  if (!fulllen)
    raiseError("invalid filename or path too long");

  lfilename = (LPWSTR)malloc(fulllen*2+2);
  MultiByteToWideChar(CP_ACP, 0, fnamebuf, -1, lfilename, fulllen*2+2);

  
  // Open Excel
  CLSID clsid;
  HRESULT hr = CLSIDFromProgID(L"Excel.Application", &clsid);

  if(FAILED(hr))
    raiseError("CLSIDFromProgID() failed");

  hr = CoCreateInstance(clsid, NULL, CLSCTX_LOCAL_SERVER, IID_IDispatch, (void **)&pXlApp);
  if(FAILED(hr))
    raiseError("Excel not registered properly");

/*  // Make it visible (i.e. app.visible = 1)
  setArg(0, 1);
  Invoke(DISPATCH_PROPERTYPUT, pXlApp, L"Visible", 1);
*/

  // Open the worksheet and get the range
  getProperty(pXlApp, L"Workbooks");
  pXlBooks = result.pdispVal;

  args->vt = VT_BSTR;
  args->bstrVal = SysAllocString(lfilename);
  Invoke(DISPATCH_PROPERTYGET, pXlBooks, L"Open", 1);
  pXlBook = result.pdispVal;
  VariantClear(args);

  if (lsheet) {
    args->vt = VT_BSTR;
    args->bstrVal = SysAllocString(lsheet);
    Invoke(DISPATCH_PROPERTYGET, pXlBook, L"Sheets", 1);
    VariantClear(args);
  }
  else
    getProperty(pXlApp, L"ActiveSheet");

  pXlSheet = result.pdispVal;

  getProperty(pXlSheet, L"UsedRange");
  pXlUsedRange = result.pdispVal;

  getProperty(pXlUsedRange, L"Value");
  cells = result.parray;

  SafeArrayGetLBound(cells, 1, &rowsLow);
  SafeArrayGetUBound(cells, 1, &rowsHigh);
  nExamples = rowsHigh - rowsLow; // no -1 -- these are inclusive limits!

  SafeArrayGetLBound(cells, 2, &columnsLow);
  SafeArrayGetUBound(cells, 2, &columnsHigh);
  nAttrs = columnsHigh - columnsLow + 1;
}


// row = example number (1..nExamples), or 0 for attribute row
// col = 0..nAttrs-1
void TExcelReader::cellAsVariant(const int &row, const int &col)
{
  VariantInit(&result);
  long pos[] = {rowsLow + row, columnsLow + col};
  SafeArrayGetElement(cells, pos, &result);
}


char *TExcelReader::cellAsText(const int &row, const int &col)
{
  if (cellvalue) {
    mldelete cellvalue;
    cellvalue = NULL;
  }

  cellAsVariant(row, col);

  if (result.vt & VT_BSTR) {
    const int blen = SysStringLen(result.bstrVal)+1;
    cellvalue = mlnew char[blen];
    const int res = WideCharToMultiByte(CP_ACP, 0, result.bstrVal, -1, cellvalue, blen, NULL, NULL);
    VariantClear(&result);
    if (!res)
      raiseError("invalid cell value");
  }

  else if (result.vt & VT_R8) {
    cellvalue = mlnew char[32];
    sprintf(cellvalue, "%8.6f", result.dblVal);
  }

  else
    raiseError("invalid cell value");

  return cellvalue;    
}


int TExcelReader::cellType(const int &row, const int &col) // 0 cannot be continuous, 1 can be continuous, 2 can even be coded discrete
{ cellAsVariant(row, col);

  if (result.vt & VT_BSTR) {
    char buf[32];
    const int res = WideCharToMultiByte(CP_ACP, 0, result.bstrVal, -1, buf, 32, NULL, NULL);
    VariantClear(&result);
    if (!res)
      return false;

    float f;
    int ssr = sscanf(buf, "%f", &f);
    return (ssr && (ssr!=EOF));
  }

  return (result.vt & VT_R8) ? 1 : 0;
}


// id is 0 for normal, > 0 for meta (meta id), -1 for class
PVariable TExcelReader::constructAttr(const int &attrNo, int &id, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
{ char *name = cellAsText(0, attrNo);
  char special = 0;

  int type = - 1;
  char *cptr = name;
  if (*cptr && (cptr[1]=='#')) {
    if (*cptr == 'i')
      return PVariable();
    else if ((*cptr == 'm') || (*cptr == 'c'))
      special = *cptr;
    else if (*cptr == 'D')
      type = TValue::INTVAR;
    else if (*cptr == 'C')
      type = TValue::FLOATVAR;
    else if (*cptr == 'S')
      type = TValue::OTHERVAR;
    else
      raiseError("unrecognized flags in attribute name '%s'", cptr);

    *cptr += 2;
  }

  else if (*cptr && cptr[1] && (cptr[2]=='#')) {
    if (*cptr == 'i')
      return PVariable();
    else if ((*cptr == 'm') || (*cptr == 'c'))
      special = *cptr;
    else
      raiseError("unrecognized flags in attribute name '%s'", cptr);

    cptr++;
    if (*cptr == 'D')
      type = TValue::INTVAR;
    else if (*cptr == 'C')
      type = TValue::FLOATVAR;
    else if (*cptr == 'S')
      type = TValue::OTHERVAR;
    else
      raiseError("unrecognized flags in attribute name '%s'", cptr);

    *cptr += 2; // we have already increased cptr once
  }

  PVariable var = makeVariable<TExcelDomain>(cptr, type, sourceVars, sourceDomain,
                                             dontCheckStored ? NULL : &knownVariables,
                                             dontCheckStored ? NULL : &knownDomains,
                                             id, type>=0, dontStore ? NULL : removeKnownVariable);
  if (!var) {
    char minCellType = 2; // 0 cannot be continuous, 1 can be continuous, 2 can even be coded discrete
    for (int row = 1; row<=nExamples; row++) {
      const char tct = cellType(row, attrNo);
      if (!tct)
        return mlnew TEnumVariable(cptr);
      if (tct < minCellType)
        minCellType = tct;
    }

    var = makeVariable<TExcelDomain>(cptr, minCellType==1 ? TValue::FLOATVAR : TValue::INTVAR,
                                     sourceVars, sourceDomain,
                                     dontCheckStored ? NULL : &knownVariables,
                                     dontCheckStored ? NULL : &knownDomains,
                                     id, false, dontStore ? NULL : removeKnownVariable);
  }

  if (special == 'm') {
    if (!id)
      id = getMetaID();
  }
  else if (special == 'c') 
    id = -1;
  else
    id = 0;

  return var;
}


// specials: 0 = normal, -1 = class, -2 = ignore, >0 = meta id
PDomain TExcelReader::constructDomain(vector<int> &specials, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
{
  TVarList attributes;
  TMetaVector metas;
  PVariable classVar;

  for (int i = 0; i < nAttrs; i++) {
    int id;
    PVariable var = constructAttr(i, id, sourceVars, sourceDomain, dontCheckStored, dontStore);
    if (!var)
      specials.push_back(-2);
    else {
      specials.push_back(id);
      if (id > 0)
        metas.push_back(TMetaDescriptor(id, var));
      else if (id == -1)
        if (classVar)
          raiseError("Multiple class variables ('%s' and '%s')", classVar->name.c_str(), var->name.c_str());
        else
          classVar = var;
      else
        attributes.push_back(var);
    }
  }

  TExcelDomain *domain = mlnew TExcelDomain(classVar, attributes);
  domain->metas = metas;

  TExcelDomain *sdomain = sourceDomain.AS(TExcelDomain);
  return (sdomain && sameDomains(domain, sdomain)) ? sourceDomain : PDomain(domain);
}


void TExcelReader::readValue(const int &row, const int &col, PVariable var, TValue &value)
{ 
  if (cellvalue) {
    mldelete cellvalue;
    cellvalue = NULL;
  }

  cellAsVariant(row, col);

  if ((result.vt & VT_BSTR) != 0) {
    const int blen = SysStringLen(result.bstrVal)+1;
    cellvalue = mlnew char[blen];
    const int res = WideCharToMultiByte(CP_ACP, 0, result.bstrVal, -1, cellvalue, blen, NULL, NULL);
    VariantClear(&result);
    if (!res)
      raiseError("invalid cell value");

    var->str2val_add(cellvalue, value);
  }

  else if ((result.vt & VT_R8) != 0) 
    if (var->varType == TValue::FLOATVAR)
      value = TValue(float(result.dblVal));
    else {
      cellvalue = mlnew char[32];
      sprintf(cellvalue, "%8.6f", result.dblVal);
      var->str2val_add(cellvalue, value);
    }
}


TExampleTable *TExcelReader::readExamples(PDomain domain, const vector<int> &specials)
{ TExampleTable *table = mlnew TExampleTable(domain);
  PVariable &classVar = domain->classVar;
  try {
    for (int row = 1; row <= nExamples; row++) {
      TExample example(domain);
      vector<int>::const_iterator speci(specials.begin());
      TVarList::const_iterator vari(domain->attributes->begin());
      TMetaVector::const_iterator meti(domain->metas.begin());
      TExample::iterator exi(example.begin());
      for (int col = 0; col < nAttrs ; col++, speci++)
        if (!*speci)
          readValue(row, col, *(vari++), *(exi++));
        else if (*speci == -1) {
          TValue value;
          readValue(row, col, classVar, value);
          example.setClass(value);
        }
        else if (*speci>0) {
          TValue value;
          readValue(row, col, (*meti).variable, value);
          example.meta.setValue((*meti).id, value);
          meti++;
        }

      table->addExample(example);
    }
  }
  catch (...) {
    mldelete table;
    throw;
  }
  return table;
}


TExampleTable *TExcelReader::operator ()(char *filename, char *sheet, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
{ openFile(filename, sheet);
  
  vector<int> specials;
  PDomain domain = constructDomain(specials, sourceVars, sourceDomain, dontCheckStored, dontStore);
  return readExamples(domain, specials);
}

TExampleTable *readExcelFile(char *filename, char *sheet, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
{ return TExcelReader()(filename, sheet, sourceVars, sourceDomain, dontCheckStored, dontStore); }

// import orange; t = orange.ExampleTable(r"D:\ai\Domene\Imp\imp\merged2.xls")

#endif