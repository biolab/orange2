if ! ~/install-scripts/mac/macinstallation.sh $1 $2 $3 $4 $5 &> output.log; then
  mail -s "Mac: ERROR compiling Orange" janez.demsar@fri.uni-lj.si < output.log
  mail -s "Mac: ERROR compiling Orange" tomaz.curk@fri.uni-lj.si < output.log
  cat output.log
  echo -e "\n\nERROR compiling, see log above"
else
  mail -s "Mac: Orange compiled successfully" tomaz.curk@fri.uni-lj.si < output.log
  cat output.log
  echo compiling was successful
fi
