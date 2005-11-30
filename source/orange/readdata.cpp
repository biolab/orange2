/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
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
#include "retisinter.hpp"
#include "assistant.hpp"
#include "basket.hpp"

#include <string.h>


#ifdef _MSC_VER
  TExampleTable *readExcelFile(char *filename, char *sheet, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore);
#endif

bool fileExists(const string &s) {
  FILE *f = fopen(s.c_str(), "rt");
  if (!f)
    return false;

  fclose(f);
  return true;
}


typedef enum {UNKNOWN, TXT, CSV, BASKET, TAB, TSV, C45, RETIS, ASSISTANT, EXCEL} TFileFormats;

char *fileTypes[][2] = {{"Tab-delimited", "*.tab"}, {"Tab-delimited (simplified)", "*.txt"}, {"Comma-separated", "*.csv"},
                       {"C45", "*.names"}, {"Retis", "*.rda"}, {"Assistant", "*.dat"}, {"Basket", "*.basket"},
                       {NULL, NULL}};
                       
WRAPPER(ExampleTable);

TExampleTable *readData(char *filename, PVarList knownVars, TMetaVector *knownMetas, PDomain knownDomain, bool dontCheckStored, bool dontStore, const char *DK, const char *DC, bool noExcOnUnknown = false, bool noCodedDiscrete = false, bool noClass = false)
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
    if (!strcmp(ext, ".txt")) {
      PExampleGenerator gen = mlnew TTabDelimExampleGenerator(filename, true, false, knownVars, knownMetas, knownDomain, dontCheckStored, dontStore, DK, DC, noCodedDiscrete, noClass);
      return mlnew TExampleTable(gen);
    }

    if (!strcmp(ext, ".csv")) {
      PExampleGenerator gen = mlnew TTabDelimExampleGenerator(filename, true, true, knownVars, knownMetas, knownDomain, dontCheckStored, dontStore, DK, DC, noCodedDiscrete, noClass);
      return mlnew TExampleTable(gen);
    }

    if (!strcmp(ext, ".tab") || !strcmp(ext, ".tsv")) {
      PExampleGenerator gen = mlnew TTabDelimExampleGenerator(filename, false, false, knownVars, knownMetas, knownDomain, dontCheckStored, dontStore, DK, DC);
      return mlnew TExampleTable(gen);
    }

    if (!strcmp(ext, ".basket")) {
      PExampleGenerator gen = mlnew TBasketExampleGenerator(filename, knownDomain, dontCheckStored, dontStore);
      return mlnew TExampleTable(gen);
    }

    if (!strcmp(ext, ".data") || !strcmp(ext, ".names") || !strcmp(ext, ".test")) {
      PExampleGenerator gen = mlnew TC45ExampleGenerator(strcmp(ext, ".names") ? filename : string(filename, ext) + ".data",
                                                         string(filename, ext) + ".names",
                                                         knownVars, knownDomain, dontCheckStored, dontStore);
      return mlnew TExampleTable(gen);
    }

    if (!strcmp(ext, ".rda") || !strcmp(ext, ".rdo")) {
      PExampleGenerator gen = mlnew TRetisExampleGenerator(string(filename, ext) + ".rda",
                                                           string(filename, ext) + ".rdo",
                                                           knownVars, knownDomain, dontCheckStored, dontStore);
      return mlnew TExampleTable(gen);
    }

    if (!strcmp(ext, ".dat")) {
      char *stem;
      for(stem = ext; (stem!=filename) && (*stem!=':') && (*stem!='\\'); stem--);
      if (stem!=filename)
        stem++;
      if (!strncmp(stem, "asd", 3) || ( (stem[3]!='o') && (stem[4]!='a') ))
        raiseError("invalid assistant filename (it should start with 'asdo' or 'asda')");

      stem += 3;
      PExampleGenerator gen = mlnew TAssistantExampleGenerator(string(filename, stem) + "a" + string(stem+1, ext), 
                                                               string(filename, stem) + "o" + string(stem+1, ext),
                                                               knownVars, knownDomain, dontCheckStored, dontStore);
      return mlnew TExampleTable(gen);
    }

    #ifdef _MSC_VER
    if ((hash-ext==4) && !strncmp(ext, ".xls", 4))
      return readExcelFile(filename, hash, knownVars, knownDomain, dontCheckStored, dontStore);
    #endif

    if (noExcOnUnknown)
      return NULL;
    else
      raiseError("unknown file format for file '%s'", filename);    
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
  CHECKFF(".rdo", RETIS);

  #ifdef _MSC_VER
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
      raiseError("file '%s' not found", filename);
  }


  /* Assistant is annoying: if path+stem is given, asd[ao] must be inserted in between */
  char *stem;
  #ifdef _MSC_VER
  for(stem = filename+strlen(filename); (stem != filename) && (*stem != '\\') && (*stem != ':') && (*stem != '/'); stem--);
  #else
  for(stem = filename+strlen(filename); (stem != filename) && (*stem != '/'); stem--);
  #endif
  if (stem!=filename)
    stem++;
  
  if (fileExists(string(filename, stem) + "asdo" + string(stem)+".dat"))
    if (fileFormat != UNKNOWN)
      raiseError("Multiple files with stem '%s' exist; specify the complete file name", filename);
    else
      fileFormat = ASSISTANT;


  string sfilename(filename);

  switch (fileFormat) {
    case TXT: {
      PExampleGenerator gen = mlnew TTabDelimExampleGenerator(sfilename+".txt", true, false, knownVars, knownMetas, knownDomain, dontCheckStored, dontStore, DK, DC, noCodedDiscrete, noClass);
      return mlnew TExampleTable(gen);
    }

    case CSV: {
      PExampleGenerator gen = mlnew TTabDelimExampleGenerator(sfilename+".csv", true, true, knownVars, knownMetas, knownDomain, dontCheckStored, dontStore, DK, DC, noCodedDiscrete, noClass);
      return mlnew TExampleTable(gen);
    }

    case TAB: {
      PExampleGenerator gen = mlnew TTabDelimExampleGenerator(sfilename+".tab", false, false, knownVars, knownMetas, knownDomain, dontCheckStored, dontStore, DK, DC);
      return mlnew TExampleTable(gen);
    }

    case TSV: {
      PExampleGenerator gen = mlnew TTabDelimExampleGenerator(sfilename+".tsv", false, false, knownVars, knownMetas, knownDomain, dontCheckStored, dontStore, DK, DC);
      return mlnew TExampleTable(gen);
    }

    case BASKET: {
      PExampleGenerator gen = mlnew TBasketExampleGenerator(sfilename+".basket", knownDomain, dontCheckStored, dontStore);
      return mlnew TExampleTable(gen);
    }

    case C45: {
      PExampleGenerator gen = mlnew TC45ExampleGenerator(sfilename + ".data", sfilename + ".names", knownVars, knownDomain, dontCheckStored, dontStore);
      return mlnew TExampleTable(gen);
    }

    case RETIS: {
      PExampleGenerator gen = mlnew TRetisExampleGenerator(sfilename + ".rda", sfilename + ".rdo", knownVars, knownDomain, dontCheckStored, dontStore);
      return mlnew TExampleTable(gen);
    }

    case ASSISTANT: {
      PExampleGenerator gen = mlnew TAssistantExampleGenerator(string(filename, stem) + "asda" + string(stem)+".dat",
                                                               string(filename, stem) + "asdo" + string(stem)+".dat",
                                                               knownVars, knownDomain, dontCheckStored, dontStore);
      return mlnew TExampleTable(gen);
    }

    #ifdef _MSC_VER
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

