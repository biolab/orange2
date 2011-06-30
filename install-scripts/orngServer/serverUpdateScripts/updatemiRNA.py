
import urllib
import re
import pylab
import random
import os
import math
import locale
import gzip
import StringIO
import sys
from getopt import getopt
import zipfile

import obiTaxonomy as tax
import orngServerFiles
import orngEnviron

def fastprint(filename,mode,what):
    
    file = open(filename,mode)
    file.write(what)
    file.close()
    

def sendMail(subject):
    
    toaddr = "rsberex@yahoo.it"
    fromaddr = "orange@fri.uni-lj.si";
    msg = "From: %s\r\nTo: %s\r\nSubject: %s" % (fromaddr, toaddr, subject)
    try:
        import smtplib
        s = smtplib.SMTP('212.235.188.18', 25)
        s.sendmail(fromaddr, toaddr, msg)
        s.quit()
    except Exception, ex:
        print "Failed to send error report due to:", ex

        
def format_checker(content):
    
    if len(re.findall('(ID.*?)ID',content.replace('\n',''))):        
        return True
    else:
        sendMail('Uncorrect format of miRBase data-file.')        
        return False

    
def get_intoFiles(path, data_webPage):
    
    sections = data_webPage.split('//\n')
    sections.pop()
    
    files = []
    os.system('rm %s/*_sections.txt' % path)
    
    for s in sections:
        org = str(re.findall('ID\s*(\S*)\s*standard;',s.split('\n')[0])[0]).split('-')[0]
        fastprint(os.path.join(path,'%s_sections.txt' % org),'a',s+'//\n')
        
        if not('%s_sections.txt' % org) in files:
            files.append('%s_sections.txt' % org)
            
    content = '\n'.join(list(set(files)))    
    fastprint(os.path.join(path,'fileList.txt'),'w',content)
            
    return os.path.join(path,'fileList.txt')
    
            
        
def miRNA_info(path,object,org_name):
    
    address = os.path.join(path,'%s' % object)
    prefix = str(re.findall('(\S*)_sections\.txt',object)[0])
    
    try:
        data_webPage = urllib.urlopen(address).read()
    except IOError:
        print "miRNA_info Error: Check the web-address."
    
    if data_webPage == []:
        sendMail('Cannot read %s ' % address)
    else:
        format_checker(data_webPage)
            
        print 'I have read: %s' % address
        sections = data_webPage.split('//\n')
        sections.pop()
        print 'Sections found: ', str(len(sections))
            
        num_s = 0
        
        ### files to write        
        fastprint(os.path.join(path,'%s_premiRNA.txt' % prefix),'w','preID'+'\t'+'preACC'+'\t'+'preSQ'+'\t'+'matACCs'+'\t'+'pubIDs'+'\t'+'clusters'+'\t'+'web_addr'+'\n')
        fastprint(os.path.join(path,'%s_matmiRNA.txt' % prefix),'w','matID'+'\t'+'matACC'+'\t'+'matSQ'+'\t'+'pre_forms'+'\t'+'targets'+'\n')
        
        dictG = {}
        dictP = {}
            
        for s in sections:
            num_s = num_s+1
            print 'section: ', num_s, '/', str(len(sections)),
                            
            pubIDs = []
            matIDs = ''
            matACCs = ''
            preSQ=[]
            
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
                    web_addr = 'http://www.mirbase.org/cgi-bin/mirna_entry.pl?acc=%s' % preACC
                        
                elif r[0:2] == 'RX' and not(re.findall('RX\s*PUBMED;\s(\d*).',r)==[]):
                    pubIDs.append(str(re.findall('RX\s*PUBMED;\s(\d*).',r)[0]))
                            
                elif r[0:2]=='FT' and not(re.findall('FT\s*miRNA\s*(\d{1,}\.\.\d{1,})',r)==[]):
                    loc_mat = str(re.findall('FT\s*miRNA\s*(\d{1,}\.\.\d{1,})',r)[0])
                        
                    if not(loc_mat==[]):
                         my_locs.append(loc_mat)
                
                elif r[0:2]=='FT' and not(re.findall('FT\s*/accession="(MIMAT[0-9]*)"', r)==[]):
                     mat_acc = str(re.findall('FT\s*/accession="(MIMAT[0-9]*)"', r)[0])
                        
                     if matACCs == '':
                         matACCs = mat_acc
                     else:
                         matACCs = matACCs + ',' + mat_acc
                            
                     if not(mat_acc == []):
                         my_accs.append(mat_acc)    
                                
                elif r[0:2]=='FT' and not(re.findall('FT\s*/product="(\S*)"', r)==[]):
                     mat_id = str(re.findall('FT\s*/product="(\S*)"', r)[0])
                        
                     if matIDs == '':
                         matIDs = mat_id
                     else:
                         matIDs = matIDs + ',' + mat_id     
                        
                     if not(mat_id == []):
                         my_ids.append(mat_id)
                                          
                elif r[0:2]=='SQ':
            
                     preSQ_INFO = str(re.findall('SQ\s*(.*other;)', r)[0])
                     seq = 'on'
            
                elif r[0:2]=='  ' and seq == 'on':
                     preSQ.append(str(re.findall('\s*([a-z\s]*)\s*\d*',r)[0]).replace(' ',''))
                     
            ### cluster search
            clusters = ''
            try:
                mirna_page = urllib.urlopen('http://www.mirbase.org/cgi-bin/mirna_entry.pl?acc=%s' % preACC).read()
            except IOError:
                print "miRNA_info Error: Check the address for the miRNA page."
                pass
            
            clust_check = re.findall('<td class="\S*">(Clustered miRNAs)</td>',mirna_page)
                
            if clust_check != [] and str(clust_check[0]) == 'Clustered miRNAs':    
                 clusters = ','.join(re.findall('<td><a href="/cgi-bin/mirna_entry.pl\?acc=MI\d*">(\S*?)</a></td>',mirna_page))
                      
            if clusters == '':
                clusters = 'None'
            
            ### before printing:       
            if pubIDs == []:
                 pubIDs = 'None'
            else:
                pubIDs = ','.join(pubIDs)
            
            preSQ = ''.join(preSQ)
            
            fastprint(os.path.join(path,'%s_premiRNA.txt' % prefix),'a',preID+'\t'+preACC+'\t'+preSQ+'\t'+matACCs+'\t'+pubIDs+'\t'+clusters+'\t'+web_addr+'\n')
                
            for tup in zip(my_ids, my_accs, my_locs):
                
                [start,stop] = tup[2].split('..')
                
                if not(tup[0] in dictG):
                    dictG[tup[0]]=[]
                
                dictG[tup[0]] = [tup[1],preSQ[int(start)-1:int(stop)]]
                
                if not(tup[0] in dictP):
                    dictP[tup[0]]=[]
                
                dictP[tup[0]].append(preID)
                
        for k,v in dictG.items():                
            pre_forms = ','.join(dictP[k]) 
            
            ### targets
            targets = 'None'
            if k in TargetScanLib:
                targets = ','.join(TargetScanLib[k])
           
            fastprint(os.path.join(path,'%s_matmiRNA.txt' % prefix),'a',k+'\t'+v[0]+'\t'+v[1]+'\t'+pre_forms+'\t'+targets+'\n')
        
            
        return [os.path.join(path,'%s_matmiRNA.txt' % prefix), os.path.join(path,'%s_premiRNA.txt' % prefix)]



##############################################################################################################################################################
##############################################################################################################################################################

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


org_taxo = [tax.name(id) for id in tax.common_taxids()]

### targets library from TargetScan
try:
    tarscan_url = 'http://www.targetscan.org//vert_50//vert_50_data_download/Conserved_Site_Context_Scores.txt.zip'
    
    zf = zipfile.ZipFile(StringIO.StringIO(urllib.urlopen(tarscan_url).read()))
    arch = zf.read(zf.namelist()[0]).split('\n')[1:]
    arch.pop()
    mirnas = [a.split('\t')[3] for a in arch]
    gene_ids = [a.split('\t')[1] for a in arch]
    
    TargetScanLib = {}
    for m,t in zip(mirnas,gene_ids):
        if not(m in TargetScanLib):
            TargetScanLib[m] = []
        if not(t in TargetScanLib[m]):           
            TargetScanLib[m].append(t)
except IOError:
    sendMail('Targets not found on: %s' % tarscan_url)    

### miRNA library form miRBase
print "\nBuilding miRNA library..."
address = 'ftp://mirbase.org/pub/mirbase/CURRENT/miRNA.dat.gz'
flag = 1
try:
    data_webPage = gzip.GzipFile(fileobj=StringIO.StringIO(urllib.urlopen(address).read())).read()    
except IOError:
    flag = 0
    sendMail('Database file of miRNAs not found on: %s' % address)
     
        
if flag:
    orgs_des = dict(zip([re.findall('ID\s*(\S{3,4})-\S*\s*standard;',l)[0] for l in data_webPage.split('\n') if l[0:2]=='ID'],[re.findall('DE\s*(.*)\s\S*.*\sstem[\s|-]loop',l)[0] for l in data_webPage.split('\n') if l[0:2]=='DE']))
    
    file_org = get_intoFiles(path,data_webPage)
    
    miRNA_path = os.path.join(path,'miRNA.txt')
    print 'miRNA file path: %s' % miRNA_path
    premiRNA_path = os.path.join(path,'premiRNA.txt')
    print 'pre-miRNA file path: %s' % premiRNA_path
    
    fastprint(miRNA_path,'w','matID'+'\t'+'matACC'+'\t'+'matSQ'+'\t'+'pre_forms'+'\t'+'targets'+'\n')
    fastprint(premiRNA_path,'w','preID'+'\t'+'preACC'+'\t'+'preSQ'+'\t'+'matACCs'+'\t'+'pubIDs'+'\t'+'clusters'+'\t'+'web_addr'+'\n')
    
    for fx in [l.rstrip() for l in open(file_org).readlines()]:
        if orgs_des[fx.split('_')[0]] in org_taxo:
            
            end_files = miRNA_info(path, fx,orgs_des[fx.split('_')[0]])
            
            for filename in end_files:
                print "Now reading %s..." % filename            
                org = re.findall('/(\S{3,4})_\S{3}miRNA\.txt',filename)[0]
                type_file = re.findall(org+'_(\S*)miRNA\.txt',filename)[0]
                label = re.findall('/(\S{3,4}_\S{3}miRNA?)\.txt',filename)[0]
                
                if type_file == 'mat':
                    serverFiles.upload("miRNA", label, filename, title="miRNA: %s mature form" % org, tags=["tag1", "tag2"])
                    serverFiles.unprotect("miRNA", label)
                    print '%s mat uploaded' % org
                    
                    for file_line in open(filename).readlines()[1:]:
                        fastprint(miRNA_path,'a',file_line)                 
                    
                elif type_file == 'pre':
                    serverFiles.upload("miRNA", label, filename, title="miRNA: %s pre-form" % org, tags=["tag1", "tag2"])
                    serverFiles.unprotect("miRNA", label)
                    print '%s pre uploaded' % org
                    
                    for file_line in open(filename).readlines()[1:]:
                        fastprint(premiRNA_path,'a',file_line)
                        
                else:
                    print 'Check the label.'
    
    serverFiles.upload("miRNA", "miRNA.txt", miRNA_path)
    serverFiles.unprotect("miRNA", "miRNA.txt")
    print '\nmiRNA.txt uploaded'
    
    serverFiles.upload("miRNA", "premiRNA.txt", premiRNA_path)
    serverFiles.unprotect("miRNA", "premiRNA.txt")
    print 'premiRNA.txt uploaded\n'
else:
    print "Check the address of miRNA file on %s" % address

                
            



   