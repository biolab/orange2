if ! ./compileOrange.sh &> output.log; then
  mail -s "ERROR compiling Orange" janez.demsar@fri.uni-lj.si < output.log
  mail -s "ERROR compiling Orange" tomaz.curk@fri.uni-lj.si < output.log 
  cat output.log
  echo -e "\n\nERROR compiling, see log above"
else
  echo "good job" | mail -s "Orange compiled successfully" tomaz.curk@fri.uni-lj.si
  echo compiling was successful
fi

