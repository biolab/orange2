import obiKEGG, obiGenomicsUpdate

u = obiKEGG.Update()
pkg = obiKEGG.PKGManager(u ,serverFiles=orngServerFiles.ServerFiles("username", "password"),
                         domain="kegg")

for item in u.GetDownloadable() + u.GetUpdatable():
    pkg.MakePKG(*item)