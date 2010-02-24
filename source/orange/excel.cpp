/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifdef INCLUDE_EXCEL

#include "table.hpp"
#include "stringvars.hpp"
#include "domaindepot.hpp"

#include "filegen.hpp"
#include <list>

#include <direct.h>
#include <ole2.h>

// These functions are wrapped into a class for easier implementation of clean-up (especially at exceptions).
class TExcelReader {
public:
  TExcelReader();
  ~TExcelReader();

  TExampleTable *operator ()(char *filename, char *sheet, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore);

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
  char celcol[3];

  void Invoke(int autoType, IDispatch *pDisp, LPOLESTR ptName, int cArgs);
  void getProperty(IDispatch *pDisp, LPOLESTR ptName);

  void openFile(char *filename, char *sheet);

  void cellAsVariant(const int &row, const int &col);
  char *cellAsText(const int &row, const int &col);
  int cellType(const int &row, const int &col);

  PDomain constructDomain(vector<int> &specials, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore);
  TExampleTable *readExamples(PDomain domain, const vector<int> &specials);
  void readValue(const int &row, const int &col, PVariable var, TValue &value);

  char *column2Chars(const int &);

  static void destroyNotifier(TDomain *domain);

  void setArg(const int argno, const int arg);
};



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


char *TExcelReader::column2Chars(const int &col)
{ if (col<=26) {
    celcol[0] = col+'A';
    celcol[1] = 0;
  }
  else {
    celcol[0] = col/26 + 'A';
    celcol[1] = col%26 + 'A';
    celcol[2] = 0;
  }

  return celcol;
}

// row = example number (1..nExamples), or 0 for attribute row
// col = 0..nAttrs-1
void TExcelReader::cellAsVariant(const int &row, const int &col)
{
  VariantInit(&result);
  long pos[] = {rowsLow + row, columnsLow + col};
  SafeArrayGetElement(cells, pos, &result);
  if (result.vt == VT_ERROR)
    raiseError("invalid value in cell %s%i", column2Chars(col), row+1);
}


char *TExcelReader::cellAsText(const int &row, const int &col)
{
  if (cellvalue) {
    mldelete cellvalue;
    cellvalue = NULL;
  }

  cellAsVariant(row, col);

  int prevvt = result.vt;
  if (   (VariantChangeType(&result, &result, 0, VT_BSTR) != S_OK)
      || (result.vt != VT_BSTR))
    raiseError("cannot convert the cell %s%i content into a string", column2Chars(col), row+1);

  const int blen = SysStringLen(result.bstrVal)+1;
  cellvalue = mlnew char[blen];
  const int res = WideCharToMultiByte(CP_ACP, 0, result.bstrVal, -1, cellvalue, blen, NULL, NULL);
  VariantClear(&result);
  if (!res)
    raiseError("invalid value in cell %s%i", column2Chars(col), row+1);

  return cellvalue;    
}


int TExcelReader::cellType(const int &row, const int &col) // 0 cannot be continuous, 1 can be continuous, 2 can even be coded discrete
{ cellAsVariant(row, col);

  if (result.vt == VT_R8) {
    float t = float(result.dblVal);
    if (floor(t) != t)
      return 1;
  }

  cellAsText(row, col);

  if (   !*cellvalue
      || !cellvalue[1] && (*cellvalue>='0') && (*cellvalue<='9'))
    return 2;

  float f;
  int ssr = sscanf(cellvalue, "%f", &f);
  return (ssr && (ssr!=(char)EOF)) ? 1 : 0;
}


// specials: 0 = normal, -1 = class, 1 = ignore, <-1 = meta id
PDomain TExcelReader::constructDomain(vector<int> &specials, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore)
{
  TDomainDepot::TAttributeDescriptions attributeDescriptions;
  TDomainDepot::TAttributeDescriptions metas;
  TDomainDepot::TAttributeDescription classDescription("", -1);

  for (int attrNo = 0; attrNo < nAttrs; attrNo++) {
    TDomainDepot::TAttributeDescription *attributeDescription;

    char *name = cellAsText(0, attrNo);
    char special = 0;

    int type = - 1;
    char *cptr = name;
    if (*cptr && (cptr[1]=='#')) {
      if (*cptr == 'i') {
        specials.push_back(1);
        continue;
      }

      else if ((*cptr == 'm') || (*cptr == 'c'))
        special = *cptr;

      else if (*cptr == 'D')
        type = TValue::INTVAR;
      else if (*cptr == 'C')
        type = TValue::FLOATVAR;
      else if (*cptr == 'S')
        type = STRINGVAR;
      else
        raiseError("unrecognized flags in attribute name '%s'", cptr);

      cptr += 2;
    }

    else if (*cptr && cptr[1] && (cptr[2]=='#')) {

      if (*cptr == 'i') {
        specials.push_back(1);
        continue;
      }
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
        type = STRINGVAR;
      else
        raiseError("unrecognized flags in attribute name '%s'", cptr);

      cptr += 2; // we have already increased cptr once
    }

    switch (special) {
      case 0:
        attributeDescriptions.push_back(TDomainDepot::TAttributeDescription(cptr, type));
        attributeDescription = &attributeDescriptions.back();
        specials.push_back(0);
        break;

      case 'm':
        metas.push_back(TDomainDepot::TAttributeDescription(cptr, type));
        attributeDescription = &metas.back();
        specials.push_back(-2); // this will later be replaced with a real id
        break;

      case 'c':
        classDescription.name = cptr;
        classDescription.varType = type;
        attributeDescription = &classDescription;
        specials.push_back(-1);
        break;
    };
        
    if (type<0) {
      char minCellType = 2; // 0 cannot be continuous, 1 can be continuous, 2 can even be coded discrete
      for (int row = 1; row<=nExamples; row++) {
        const char tct = cellType(row, attrNo);
        if (!tct) {
          attributeDescription->varType = TValue::INTVAR;
          break;
        }
        if (tct < minCellType)
          minCellType = tct;
      }

      attributeDescription->varType = minCellType==1 ? TValue::FLOATVAR : TValue::INTVAR;
    }
  }

  if (classDescription.varType >= 0)
    attributeDescriptions.push_back(classDescription);

  if (sourceDomain) {
    if (!domainDepot.checkDomain(sourceDomain.AS(TDomain), &attributeDescriptions, true, NULL))
      raiseError("given domain does not match the file");
    else
      return sourceDomain;
  }

  int *metaIDs = mlnew int[metas.size()];
  PDomain newDomain = domainDepot.prepareDomain(&attributeDescriptions, classDescription.varType>=0, &metas, sourceVars, NULL, dontStore, dontCheckStored, NULL, metaIDs);

  int *mid = metaIDs;
  ITERATE(vector<int>, ii, specials)
    if (*ii == -2)
      *ii = *(mid++);

  mldelete metaIDs;

  return newDomain;
}


void TExcelReader::readValue(const int &row, const int &col, PVariable var, TValue &value)
{ 
  if (cellvalue) {
    mldelete cellvalue;
    cellvalue = NULL;
  }

  cellAsVariant(row, col);

  if ((result.vt == VT_R8) && (var->varType == TValue::FLOATVAR))
    value = TValue(float(result.dblVal));

  else {
    int prevvt = result.vt;
    if (   (VariantChangeType(&result, &result, 0, VT_BSTR) != S_OK)
        || (result.vt != VT_BSTR))
      raiseError("cannot convert content of cell %s%i", column2Chars(col), row+1);
  }


  if ((result.vt & VT_BSTR) != 0) {
    const int blen = SysStringLen(result.bstrVal)+1;
    cellvalue = mlnew char[blen];
    const int res = WideCharToMultiByte(CP_ACP, 0, result.bstrVal, -1, cellvalue, blen, NULL, NULL);
    VariantClear(&result);
    if (!res)
      raiseError("invalid value in cell %s%i", column2Chars(col), row+1);

    try {
      var->str2val_add(cellvalue, value);
    }
    catch (mlexception err) {
      raiseError("cannot convert content of cell %s%i ('%s')", column2Chars(col), row+1, cellvalue);
    }
  }

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
        else if (*speci < -1) {
          TValue value;
          readValue(row, col, (*meti).variable, value);
          example.setMeta((*meti).id, value);
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
