if ! ./compileOrange.sh &> output.log; then
  mail -s "Linux: ERROR compiling Orange" janez.demsar@fri.uni-lj.si < output.log
  mail -s "Linux: ERROR compiling Orange" tomaz.curk@fri.uni-lj.si < output.log 
  cat output.log
  echo -e "\n\nERROR compiling, see log above"
else
  mail -s "Linux: Orange compiled successfully" tomaz.curk@fri.uni-lj.si < output.log
  cat output.log
  echo compiling was successful
fi

