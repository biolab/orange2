import obiGO, obiGenomicsUpdate

u = obiGO.Update()
pkg = obiGO.PKGManager(u ,serverFiles=orngServerFiles.ServerFiles("username", "password"),
                         domain="go")

for item in u.GetDownloadable() + u.GetUpdatable():
    if item != (obiGO.Update.UpdateAnnotation, ('goa_uniprot',)):
        pkg.MakePKG(*item)

    
    