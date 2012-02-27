import urllib, sys, os, md5


files = "orange", "corn", "statc", "orangeom", "orangene", "_orngCRS"

if sys.version_info[:2] == (2, 7):
	# orangeqt is build only for 2.7
	files = files + ("orangeqt",)
	
baseurl = "http://orange.biolab.si/download/binaries/%i%i/" % sys.version_info[:2]
fleurl = baseurl + "%s.pyd"

op = filter(lambda x:x[-12:].lower() in ["\\orange\\orng", "/orange/orng"], sys.path)
if not op:
	print "Orange is not found on the Python's path"
	
op = os.path.dirname(op[0])

print "Downloading to %s (for Python %i.%i)" % (op, sys.version_info[0], sys.version_info[1])
os.chdir(op)

def rep(blk_cnt, blk_size, tot_size):
	print "\rDownloading %s: %i of %i" % (fle, min(tot_size, blk_cnt*blk_size), tot_size),

repository_stamps = dict([tuple(x.split()) for x in urllib.urlopen(baseurl + "stamps_pyd.txt") if x.strip()])

for fle in files:
    if os.path.exists(fle+".pyd") and repository_stamps.get(fle+".pyd", "") == md5.md5(file(fle+".pyd", "rb").read()).hexdigest().upper():
        print "\nSkipping %s" % fle,
    else:
        print "\nDownloading %s" % fle,
        urllib.urlretrieve(fleurl % fle, fle+".temp", rep)
        if os.path.exists(fle+".pyd"):
            os.remove(fle+".pyd")
        os.rename(fle+".temp", fle+".pyd")
