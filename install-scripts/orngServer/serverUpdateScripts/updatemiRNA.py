
import urllib, re, pylab, random, os, math, locale, gzip, StringIO, orngServerFiles, orngEnviron, os, sys
from getopt import getopt
import obiTaxonomy as tax


def get_intoFiles(self):
    
    ## automatic
    address = 'ftp://mirbase.org/pub/mirbase/CURRENT/miRNA.dat.gz'
    try:
        data_webPage = gzip.GzipFile(fileobj=StringIO.StringIO(urllib.urlopen(address).read())).read()
    except IOerror:
        print "get_intoFiles Error: Check the web-address."
        stop    
        
    sections = data_webPage.split('//\n')
    sections.pop()
    
    files = []
    for s in sections:
        org = str(re.findall('ID\s*(\S*)\s*standard;',s.split('\n')[0])[0]).split('-')[0]
        f = open(self+'/%s_sections.txt' % org,'a')
        f.write(s+'//\n')
        f.close()
        
        if not('%s_miRNA.txt' % org) in files:
            files.append('%s_sections.txt' % org)
            
    return list(set(files))
    
            
        
def miRNA_info(path,object):
    
    address = path+'/%s' % object
    prefix = str(re.findall('(\S*)_sections\.txt',object)[0])
    
    try:
        data_webPage = urllib.urlopen(address).read()
    except IOError:
        print "miRNA_info Error: Check the web-address."
    
    if data_webPage == []:
        print 'Cannot read %s ' % address
    else:
        print 'I have read: %s' % address
        sections = data_webPage.split('//\n')
        sections.pop()
        print 'Sections found: ', str(len(sections))
        
            
        num_s = 0
        
        ### files to write  
        premiRNA = open(path+'/%s_premiRNA.txt' % prefix,'w')
        premiRNA.write('preID'+'\t'+'preACC'+'\t'+'preSQ'+'\t'+'matACCs'+'\t'+'pubIDs'+'\t'+'clusters'+'\t'+'web_addr'+'\n')
        premiRNA.close()
            
        matmiRNA = open(path+'/%s_matmiRNA.txt' % prefix,'w')
        matmiRNA.write('matID'+'\t'+'matACC'+'\t'+'matSQ'+'\t'+'pre_forms'+'\n')
        matmiRNA.close()
        
        dictG = {}
        dictP = {}
            
        for s in sections:
            num_s = num_s+1
            print 'section: ', num_s,
                            
            pubIDs = ''
            matIDs = ''
            matACCs = ''
            preSQ=''
            
            my_ids =[]
            my_accs=[]
            my_locs=[] # if it's [61..81] you have to take from 60 to 81.
            
            rows = s.split('\n')
                
            for r in rows:
                
                if r[0:2] == 'ID':
                    preID = str(re.findall('ID\s*(\S*)\s*standard;',r)[0])
                    print preID
                        
                elif r[0:2] == 'AC':
                    preACC = str(re.findall('AC\s*(\S*);',r)[0])
                    #print preACC
                    web_addr = 'http://www.mirbase.org/cgi-bin/mirna_entry.pl?acc=%s' % preACC
                    #print web_addr    
                elif r[0:2] == 'RX' and not(re.findall('RX\s*PUBMED;\s(\d*).',r)==[]):
                    pmid = str(re.findall('RX\s*PUBMED;\s(\d*).',r)[0])
                        
                    if pubIDs == '':
                         pubIDs = pmid
                    else:
                         pubIDs = pubIDs + ',' + pmid
                    
                            
                elif r[0:2]=='FT' and not(re.findall('FT\s*miRNA\s*(\d{1,}\.\.\d{1,})',r)==[]):
                    loc_mat = str(re.findall('FT\s*miRNA\s*(\d{1,}\.\.\d{1,})',r)[0])
                        
                    if not(loc_mat==[]):
                         my_locs.append(loc_mat)
                        
                    #print my_locs,
                
                elif r[0:2]=='FT' and not(re.findall('FT\s*/accession="(MIMAT[0-9]*)"', r)==[]):
                     mat_acc = str(re.findall('FT\s*/accession="(MIMAT[0-9]*)"', r)[0])
                        
                     if matACCs == '':
                         matACCs = mat_acc
                     else:
                         matACCs = matACCs + ',' + mat_acc
                            
                     if not(mat_acc == []):
                         my_accs.append(mat_acc)    
                        
                     #print my_accs
                        
                elif r[0:2]=='FT' and not(re.findall('FT\s*/product="(\S*)"', r)==[]):
                     mat_id = str(re.findall('FT\s*/product="(\S*)"', r)[0])
                        
                     if matIDs == '':
                         matIDs = mat_id
                     else:
                         matIDs = matIDs + ',' + mat_id     
                        
                     if not(mat_id == []):
                         my_ids.append(mat_id)
                        
                     #print my_ids
                              
                elif r[0:2]=='SQ':
            
                     preSQ_INFO = str(re.findall('SQ\s*(.*other;)', r)[0])
                     seq = 'on'
            
                elif r[0:2]=='  ' and seq == 'on':
                     piece_seq= str(re.findall('\s*([a-z\s]*)\s*\d*',r)[0]).replace(' ','')
                        
                     if preSQ == '':
                         preSQ = piece_seq
                     else:
                         preSQ = preSQ + piece_seq
                        
                     #print preSQ
                     
            ### cluster search
            clusters = ''
            try:
                mirna_page = urllib.urlopen('http://www.mirbase.org/cgi-bin/mirna_entry.pl?acc=%s' % preACC).read()
            except IOError:
                print "miRNA_info Error: Check the address for the miRNA page."
                pass
            
            clust_check = re.findall('<td class="\S*">(Clustered miRNAs)</td>',mirna_page)
                
            if clust_check != [] and str(clust_check[0]) == 'Clustered miRNAs':    
                 club = re.findall('<td><a href="/cgi-bin/mirna_entry.pl\?acc=MI\d*">(\S*?)</a></td>',mirna_page)
                 clusters = ','.join(club)
                 
                 
            if clusters == '':
                clusters = 'None'
            #print clusters,
                    
            if pubIDs == '':
                 pubIDs = 'None'
            #print pubIDs
            #print
                 
        
            premiRNA = open(path+'/%s_premiRNA.txt' % prefix,'a')
            premiRNA.write(preID+'\t'+preACC+'\t'+preSQ+'\t'+matACCs+'\t'+pubIDs+'\t'+clusters+'\t'+web_addr+'\n')
            premiRNA.close()
                
            for tup in zip(my_ids, my_accs, my_locs):
                
                [start,stop] = tup[2].split('..')
                
                if not(tup[0] in dictG):
                    dictG[tup[0]]=[]
                
                dictG[tup[0]] = [tup[1],preSQ[int(start)-1:int(stop)]]
                
                if not(tup[0] in dictP):
                    dictP[tup[0]]=[]
                
                dictP[tup[0]].append(preID)
                
        for k,v in dictG.items():                
            pre_forms = ''
            for p in dictP[k]:
                if pre_forms=='':
                      pre_forms = p
                else:
                      pre_forms = pre_forms + ',' + p
                        
            #print 'stampa: ', k, v[0], v[1], pre_forms
            matmiRNA = open(path+'/%s_matmiRNA.txt' % prefix,'a')        
            matmiRNA.write(k+'\t'+v[0]+'\t'+v[1]+'\t'+pre_forms+'\n')
            matmiRNA.close()  
            
        
        #print 'End of miRNA_info'    
        return [path+'/%s_matmiRNA.txt' % prefix, path+'/%s_premiRNA.txt' % prefix]


opt = dict(getopt(sys.argv[1:], "u:p:", ["user=", "password="])[0])

username = opt.get("-u", opt.get("--user", "username"))
password = opt.get("-p", opt.get("--password", "password"))

path = os.path.join(orngEnviron.bufferDir, "tmp_miRNA")
print 'path: ', path

serverFiles = orngServerFiles.ServerFiles(username, password)

try:
    os.mkdir(path)
except OSError:
    pass


os.system('rm %s/*_sections.txt' % path)

## automatic
address = 'ftp://mirbase.org/pub/mirbase/CURRENT/miRNA.dat.gz'
try:
    data_webPage = gzip.GzipFile(fileobj=StringIO.StringIO(urllib.urlopen(address).read())).read()
except IOerror:
    print "updatemiRNA Error: Check the web-address."
    stop 

orgs_des = dict(zip([re.findall('ID\s*(\S{3,4})-\S*\s*standard;',l)[0] for l in data_webPage.split('\n') if l[0:2]=='ID'],[re.findall('DE\s*(.*)\s\S*.*\sstem[\s|-]loop',l)[0] for l in data_webPage.split('\n') if l[0:2]=='DE']))

file_org = get_intoFiles(path)

miRNA_path = path+'/miRNA.txt'
premiRNA_path = path+'/premiRNA.txt'

total_miRNA = open(miRNA_path,'w')
total_miRNA.write('matID'+'\t'+'matACC'+'\t'+'matSQ'+'\t'+'pre_forms'+'\n')
total_miRNA.close()

total_premiRNA = open(premiRNA_path,'w')
total_premiRNA.write('preID'+'\t'+'preACC'+'\t'+'preSQ'+'\t'+'matACCs'+'\t'+'pubIDs'+'\t'+'clusters'+'\t'+'web_addr'+'\n')
total_premiRNA.close()


for fx in file_org:
    if orgs_des[fx.split('_')[0]] in [tax.name(id) for id in tax.common_taxids()]:
        
        end_files = miRNA_info(path, fx)
        
        for filename in end_files:
                        
            org = re.findall('/(\S{3,4})_\S{3}miRNA\.txt',filename)[0]
            type_file = re.findall(org+'_(\S*)miRNA\.txt',filename)[0]
            label = re.findall('/(\S{3,4}_\S{3}miRNA?)\.txt',filename)[0]
            
            #print org, type_file, label
            
            if type_file == 'mat':
                serverFiles.upload("miRNA", label, filename, title="mature miRNA for %s" % org, tags=["tag1", "tag2"])
                serverFiles.unprotect("miRNA", label)
                print 'mat uploaded'
                
                
                total_miRNA = open(miRNA_path,'a')
                content_lines = open(filename).readlines()[1:]
                for file_line in content_lines:
                    total_miRNA.write(file_line)
                    #print file_line
                total_miRNA.close()
                
            elif type_file == 'pre':
                serverFiles.upload("miRNA", label, filename, title="pre-miRNA for %s" % org, tags=["tag1", "tag2"])
                serverFiles.unprotect("miRNA", label)
                print 'pre uploaded'
                
                total_premiRNA = open(premiRNA_path,'a')
                content_lines = open(filename).readlines()[1:]
                for file_line in content_lines:
                    total_premiRNA.write(file_line)
                    #print file_line
                total_premiRNA.close()
            else:
                print 'Check the label.'


serverFiles.upload("miRNA", "miRNA.txt", miRNA_path)
serverFiles.unprotect("miRNA", "miRNA.txt")

serverFiles.upload("miRNA", "premiRNA.txt", premiRNA_path)
serverFiles.unprotect("miRNA", "premiRNA.txt")

                
            



   