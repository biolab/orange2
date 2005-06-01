if ! ~/install-scripts/linux/compileOrange.sh &> output.log; then
  mail -s "Linux: ERROR compiling Orange" janez.demsar@fri.uni-lj.si < output.log
  mail -s "Linux: ERROR compiling Orange" tomaz.curk@fri.uni-lj.si < output.log
  mail -s "Linux: ERROR compiling Orange" jurem@flextronics.si < output.log 
  cat output.log
  echo -e "\n\nERROR compiling, see log above"
  exit 1 
fi

echo -e "\n\nStarting regression tests...\n\n" >> output.log

if ! ~/install-scripts/linux/testOrange.sh >> output.log 2>&1; then
  mail -s "Linux: ERROR regression tests (compile OK) Orange" janez.demsar@fri.uni-lj.si < output.log
  mail -s "Linux: ERROR regression tests (compile OK) Orange" tomaz.curk@fri.uni-lj.si < output.log
  mail -s "Linux: ERROR regression tests (compile ok) Orange" jurem@flextronics.si < output.log 
  cat output.log
  echo regression tests failed
else  
  mail -s "Linux: Orange compiled successfully" tomaz.curk@fri.uni-lj.si < output.log
  mail -s "Linux: Orange compiled successfully" jurem@flextronics.si < output.log 
  cat output.log
  echo compiling was successful
fi

