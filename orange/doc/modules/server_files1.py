import orngServerFiles

# remove a file from a local repository and download it from the server
filename = "urllib2_file.py"
print orngServerFiles.listfiles("demo")
orngServerFiles.remove("demo", filename)
orngServerFiles.download("demo", filename, verbose=False)

info = orngServerFiles.info("demo", filename)
print "%s: size=%s, datetime=%s" % (filename, info["size"], info["datetime"])
