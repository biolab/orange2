import os, glob, sys, orange, time, os.path, string

descriptors = ['fname', 'inst', 'size', 'att', 'categ', 'cont', '%cont', 'class', 'values', '%major', 'date', 'description']
verbose = 1

# construct_datasets(): constructs list of data sets
def build_datasets():
  # use current directory instead
  # os.chdir("d:\webMagix\orange\download\demos")
  return glob.glob("*.tab")

def dataset_statistics(fname, trace=0):
  data = orange.ExampleTable(fname)
  s = [fname]

  # instances and size [kBytes]
  size = '%5.1f' % (os.path.getsize(fname)/1000.)
  s = s + [len(data), size]

  # attributes
  natt = len(data.domain.attributes)
  ncont=0; ndisc=0
  for a in data.domain.attributes:
    if a.varType == orange.VarTypes.Discrete: ndisc = ndisc + 1
    else: ncont = ncont + 1
  pcont = '%5.1f' % (100.0 * ncont / natt)
  s = s + [natt, ndisc, ncont, pcont]

  # class name, values, majority class
  if data.domain.classVar:
    cname = data.domain.classVar.name
    if data.domain.classVar.varType == 1:  # categorical data set
      cval = 'discrete/' + str(len(data.domain.classVar.values))
      c = [0] * len(data.domain.classVar.values)
      for e in data:
        c[int(e.getclass())] += 1
      cmaj = '%5.1f' % (100.0 * max(c) / len(data))
    else: # continuous data set
      cval = 'continuous'
      cmaj = 'n/a'
  else:
    cname = 'n/a'; cval = 'n/a'; cmaj = 'n/a'
  s = s + [cname, cval, cmaj]

  # date
  rtime = time.gmtime(os.path.getmtime(fname))
  t = time.strftime("%m/%d/%y", rtime)
  s = s + [t]

  # description
  s = s + ['-']

  # wrap up    
  if trace: print fname, s
  return s

def compute_statistics(flist, trace=0):
  global verbose
  stat = {}
  for f in flist:
    if verbose:
      print "processing %s" % (f)
    s = dataset_statistics(f, trace)
    stat[f] = s
  return stat

# obtain past descriptions (attributes) from info file
def get_past():
  past = {}
  if glob.glob("data_info.txt"):
    f = open("data_info.txt")
    for line in f:
      line = line[:-1] #remove new line at the end
      att = string.split(line, '\t')
      past[att[0]] = att
    f.close()

    import time
    t = time.strftime("%m-%d-%y_%H-%M-%S", time.localtime(time.time()))
    os.rename('data_info.txt', 'data_info_%s.txt' % t)
  return past

def get_past_desc():
  past_desc = {}
  if glob.glob("data_info.txt"):
    f = open("data_info.txt")
    for line in f:
      line = line[:-1]
      att = string.split(line, '\t')
      past_desc[att[0]] = att[-1]
    f.close()

    import time
    t = time.strftime("%m-%d-%y_%H-%M-%S", time.localtime(time.time()))
    os.rename('data_info.txt', 'data_info_%s.txt' % t)
  return past_desc

def save_info(stat):
  f = open("data_info.txt", 'w')
  s = reduce(lambda x,y: str(x)+"\t"+str(y), descriptors)
  f.write(s+'\n')
  keys = stat.keys()
  keys.sort()
  print keys
  for k in keys:
    s = reduce(lambda x,y: str(x)+"\t"+str(y),stat[k])
    f.write(s+"\n")
  f.close()

def help():
  print 'data_info.py [-help|-list|-update|-add]'
  print '  -help   prints this message'
  print '  -list   lists statistics for data files'
  print '  -update updates statistics in data_info.txt, maintains description fields'
  print '  -add    adds statistics for data files not present in data_info.txt'

def main():
  flist = build_datasets()
  if '-help' in sys.argv: help()
  elif '-list' in sys.argv:
    compute_statistics(flist, trace=1)
  elif '-add' in sys.argv:
    past = get_past()
    k = past.keys()
    fnew = filter(lambda x, k=k: not x in k, flist)
    print 'new=', fnew
    stat = compute_statistics(fnew)
    # append past statistics
    for k in past.keys():
      stat[k] = past[k]
    save_info(stat)
  elif '-update' in sys.argv or 1:
    # only description and file name is read here, where not equal to '-'
    # this is used to update new statistics.
    # this is constructed primarily if we want to change the number of
    # descriptive fields
    past_desc = get_past_desc()
    past_desc_keys = past_desc.keys()
    stat = compute_statistics(flist)
    for k in stat.keys():
      if k in past_desc_keys:
        stat[k][-1] = past_desc[k]
    save_info(stat)
  else: help()
  
main()
