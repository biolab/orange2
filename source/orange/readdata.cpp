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

#include <string.h>

bool fileExists(const string &s) {
  FILE *f = fopen(s.c_str(), "rt");
  if (!f)
    return false;

  fclose(f);
  return true;
}


typedef enum {UNKNOWN, TXT, TAB, C45, RETIS, ASSISTANT} TFileFormats;

WRAPPER(ExampleTable);

TExampleTable *readData(char *filename, PVarList knownVars)
{ char *ext;
  if (filename) {
    for(ext = filename + strlen(filename); ext!=filename; ext--) {
      if (*ext == '.')
        break;
      else if ((*ext=='\\') || (*ext==':')) {
        ext = NULL;
        break;
      }
    }
    if (ext==filename)
      ext = NULL;
  }
  else
    ext = NULL;

  // If the extension is given, we simply determine the format and load the files
  if (ext) {
    if (!strcmp(ext, ".txt")) {
      PDomain domain = mlnew TTabDelimDomain(filename, knownVars);
      PExampleGenerator gen = mlnew TTabDelimExampleGenerator(filename, domain);
      return mlnew TExampleTable(gen);
    }

    if (!strcmp(ext, ".tab")) {
      PDomain domain = mlnew TTabDelimDomain(filename, knownVars);
      PExampleGenerator gen = mlnew TTabDelimExampleGenerator(filename, domain);
      return mlnew TExampleTable(gen);
    }

    if (!strcmp(ext, ".data") || !strcmp(ext, ".names") || !strcmp(ext, ".test")) {
      PDomain domain = mlnew TC45Domain(string(filename, ext) + ".names", knownVars);
      PExampleGenerator gen = mlnew TC45ExampleGenerator(strcmp(ext, ".names") ? filename : string(filename, ext) + ".data", domain);
      return mlnew TExampleTable(gen);
    }

    if (!strcmp(ext, ".rda") || !strcmp(ext, ".rdo")) {
      PDomain domain = mlnew TRetisDomain(string(filename, ext) + ".rdo", knownVars);
      PExampleGenerator gen = mlnew TRetisExampleGenerator(string(filename, ext) + ".rda", domain);
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
      PDomain domain = mlnew TAssistantDomain(string(filename, stem) + "o" + string(stem+1, ext));
      PExampleGenerator gen = mlnew TAssistantExampleGenerator(string(filename, stem) + "a" + string(stem+1, ext), domain);
      return mlnew TExampleTable(gen);
    }

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
      for(filename = ep; *filename != '\\'; filename--);
    #else
      char dirName[256];
      getcwd(dirName, 256);
      ep = dirName + strlen(dirName);
      for(filename = ep; *filename != '/'; filename--);
    #endif

    if ((filename == ep ) || (filename == ep-1))
      raiseError("filename not given and cannot be concluded from the working directory");

    filename++;
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
  CHECKFF(".tab", TAB);
  CHECKFF(".names", C45);
  CHECKFF(".rdo", RETIS);

  #undef CHECKFF

  if (fileFormat == UNKNOWN)
    raiseError("file '%s' not found", filename);


  /* Assistant is annoying: if path+stem is given, asd[ao] must be inserted in between */
  char *stem;
  for(stem = filename+strlen(filename); (stem != filename) && (*stem != '\\') && (*stem != ':'); stem--);
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
      PDomain domain = mlnew TTabDelimDomain(sfilename+".txt", knownVars, true);
      PExampleGenerator gen = mlnew TTabDelimExampleGenerator(sfilename+".txt", domain);
      return mlnew TExampleTable(gen);
    }

    case TAB: {
      PDomain domain = mlnew TTabDelimDomain(sfilename+".tab", knownVars, false);
      PExampleGenerator gen = mlnew TTabDelimExampleGenerator(sfilename+".tab", domain);
      return mlnew TExampleTable(gen);
    }

    case C45: {
      PDomain domain = mlnew TC45Domain(sfilename + ".names", knownVars);
      PExampleGenerator gen = mlnew TC45ExampleGenerator(sfilename + ".data", domain);
      return mlnew TExampleTable(gen);
    }

    case RETIS: {
      PDomain domain = mlnew TRetisDomain(sfilename + ".rdo", knownVars);
      PExampleGenerator gen = mlnew TRetisExampleGenerator(sfilename + ".rda", domain);
      return mlnew TExampleTable(gen);
    }

    case ASSISTANT: {
      PDomain domain = mlnew TAssistantDomain(string(filename, stem) + "asdo" + string(stem)+".dat");
      PExampleGenerator gen = mlnew TAssistantExampleGenerator(string(filename, stem) + "asda" + string(stem)+".dat", domain);
      return mlnew TExampleTable(gen);
    }

    default:
      raiseError("unknown file format for file '%s'", filename);
  }
  return NULL;
}
