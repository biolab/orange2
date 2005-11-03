# remove old logs
rm /var/www/html/orange/*.log

# copy new ones
cd /home/orange/daily/orange
cp *.log ../*.log /var/www/html/orange
chmod o+r /var/www/html/orange/*.log
