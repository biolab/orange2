if ! ./update.sh &> output.log; then
  mail -s "ERROR updating Orange RPMs" tomaz.curk@fri.uni-lj.si < output.log
  cat output.log
  echo -e "\n\nERROR updating, see log above"
else
  echo "good job" | mail -s "Orange RPM update successful" tomaz.curk@fri.uni-lj.si
  echo update was successful
fi

