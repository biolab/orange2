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


#include <iostream>
#include <fstream>

#ifdef _MSC_VER
  #include <direct.h>
#else
  #include <unistd.h>
#endif

#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "table.hpp"

#include "filegen.hpp"
#include "tabdelim.hpp"
#include "c45inter.hpp"
#include "basket.hpp"

#include <string.h>


#ifdef INCLUDE_EXCEL
  TExampleTable *readExcelFile(char *filename, char *sheet, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore);
#endif

bool fileExists(const string &s) {
  FILE *f = fopen(s.c_str(), "rt");
  if (!f)
    return false;

  fclose(f);
  return true;
}


typedef enum {UNKNOWN, TXT, CSV, BASKET, TAB, TSV, C45, EXCEL} TFileFormats;

char *fileTypes[][2] = {{"Tab-delimited", "*.tab"}, {"Tab-delimited (simplified)", "*.txt"}, {"Comma-separated", "*.csv"},
                       {"C45", "*.names"}, {"Basket", "*.basket"},
                       {NULL, NULL}};
                       
TExampleGenerator *readGenerator(char *filename, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus, const char *DK, const char *DC, bool noExcOnUnknown = false, bool noCodedDiscrete = false, bool noClass = false)
{ char *ext, *hash;
  if (filename) {
    for(ext = hash = filename + strlen(filename); ext!=filename; ext--) {
      if (*ext == '.')
        break;
      else if ((*ext=='/') || (*ext=='\\') || (*ext==':')) {
        ext = NULL;
        break;
      }
      else if (!*hash && (*ext == '#'))
        hash = ext;
    }
    if (ext==filename)
      ext = NULL;
  }
  else
    ext = NULL;

  // If the extension is given, we simply determine the format and load the files
  if (ext) {
    if (!strcmp(ext, ".txt"))
      return mlnew TTabDelimExampleGenerator(filename, true, false, createNewOn, status, metaStatus, DK, DC, noCodedDiscrete, noClass);

    if (!strcmp(ext, ".csv"))
      return mlnew TTabDelimExampleGenerator(filename, true, true, createNewOn, status, metaStatus, DK, DC, noCodedDiscrete, noClass);

    if (!strcmp(ext, ".tab") || !strcmp(ext, ".tsv"))
      return mlnew TTabDelimExampleGenerator(filename, false, false, createNewOn, status, metaStatus, DK, DC);

    if (!strcmp(ext, ".basket"))
      return mlnew TBasketExampleGenerator(filename, PDomain(), createNewOn, status, metaStatus);

    if (!strcmp(ext, ".data") || !strcmp(ext, ".names") || !strcmp(ext, ".test"))
      return mlnew TC45ExampleGenerator(strcmp(ext, ".names") ? filename : string(filename, ext) + ".data",
                                                         string(filename, ext) + ".names",
                                                         createNewOn, status, metaStatus);
    #ifdef INCLUDE_EXCEL
    if ((hash-ext==4) && !strncmp(ext, ".xls", 4))
      return readExcelFile(filename, hash, knownVars, knownDomain, dontCheckStored, dontStore);
    #endif
  }

  /* If no filename is given at all, assume that the stem equals the last
     subdirectory name. Eg, the directory c:\data\monk1 is supposed to
     contain a file monk1 in one of the supported formats. */
  char *ep;
  if (!filename) {
    #ifdef _MSC_VER
      char dirName[_MAX_PATH];
      _getcwd(dirName, _MAX_PATH);
      ep = dirName + strlen(dirName);
      for(filename = ep; (*filename != '\\') && (*filename != '/'); filename--);
    #else
      char dirName[256];
      getcwd(dirName, 256);
      ep = dirName + strlen(dirName);
      for(filename = ep; *filename != '/'; filename--);
    #endif

    if ((filename == ep ) || (filename == ep-1))
      raiseError("filename not given and cannot be concluded from the working directory");

    filename++;
    hash = filename + strlen(filename);
  }

  int fileFormat = UNKNOWN;
  // CHECKFF(file extension, format name)
  #define CHECKFF(fext,ff) \
     if (fileExists(string(filename)+fext)) \
       if (fileFormat != UNKNOWN) \
         raiseError("Multiple files with stem '%s' exist; specify the complete file name", filename); \
       else \
         fileFormat = ff;
              
  CHECKFF(".txt", TXT);
  CHECKFF(".csv", CSV);
  CHECKFF(".basket", BASKET);
  CHECKFF(".tab", TAB);
  CHECKFF(".tsv", TSV);
  CHECKFF(".names", C45);

  #ifdef INCLUDE_EXCEL
    if (*hash) {
      *hash = 0;
      CHECKFF(".xls", EXCEL);
      *hash = '#';
    }
    else
      CHECKFF(".xls", EXCEL);
  #endif

  #undef CHECKFF

  if (fileFormat == UNKNOWN) {
    if (noExcOnUnknown)
      return NULL;
    else
      if (ext)
        raiseError("unknown file format for file '%s' or file not found", filename);    
      else
        raiseError("file '%s' is not found or has unknown extension", filename);
  }


  string sfilename(filename);

  switch (fileFormat) {
    case TXT: 
      return mlnew TTabDelimExampleGenerator(sfilename+".txt", true, false, createNewOn, status, metaStatus, DK, DC, noCodedDiscrete, noClass);

    case CSV:
      return mlnew TTabDelimExampleGenerator(sfilename+".csv", true, true, createNewOn, status, metaStatus, DK, DC, noCodedDiscrete, noClass);

    case TAB:
      return mlnew TTabDelimExampleGenerator(sfilename+".tab", false, false, createNewOn, status, metaStatus, DK, DC);

    case TSV:
      return mlnew TTabDelimExampleGenerator(sfilename+".tsv", false, false, createNewOn, status, metaStatus, DK, DC);

    case BASKET:
      return mlnew TBasketExampleGenerator(sfilename+".basket", PDomain(), createNewOn, status, metaStatus);

    case C45:
      return mlnew TC45ExampleGenerator(sfilename + ".data", sfilename + ".names", createNewOn, status, metaStatus);


    #ifdef INCLUDE_EXCEL
    case EXCEL:
      return readExcelFile(filename, hash, knownVars, knownDomain, dontCheckStored, dontStore);
    #endif

    default:
      if (noExcOnUnknown)
        return NULL;
      else
        raiseError("unknown file format for file '%s'", filename);
  }
  return NULL;
}


TExampleTable *readTable(char *filename, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus, const char *DK, const char *DC, bool noExcOnUnknown = false, bool noCodedDiscrete = false, bool noClass = false)
{
  TExampleGenerator *gen = readGenerator(filename, createNewOn, status, metaStatus, DK, DC, noExcOnUnknown, noCodedDiscrete, noClass);
  if (!gen)
    return NULL;
  TExampleTable *table = dynamic_cast<TExampleTable *>(gen);
  return table ? table : new TExampleTable(gen);
}
