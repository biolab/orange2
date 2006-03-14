# remove old logs
rm /var/www/html/orange/*.log

# copy new ones
cd /home/orange/daily/orange
cp -p *.log ../*.log /var/www/html/orange
chmod o+r /var/www/html/orange/*.log

cd /home/orange/daily/test_install/orange
rm -Rf /var/www/html/orange/tests
mkdir /var/www/html/orange/tests

for f in *-output; do
	echo copying $f
	mkdir /var/www/html/orange/tests/$f
	cp -p $f/*.txt /var/www/html/orange/tests/$f
	chmod o+r /var/www/html/orange/tests/$f
	chmod o+r /var/www/html/orange/tests/$f/*.txt
done

cp testresults.xml /var/www/html/orange/tests
chmod o+r /var/www/html/orange/tests/testresults.xml

