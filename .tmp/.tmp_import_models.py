from importlib import import_module
modules=['menipy.models.context','menipy.models.config','menipy.models.frame','menipy.models.geometry','menipy.models.fit','menipy.models.result']
for m in modules:
    try:
        mod=import_module(m)
        print(m+' OK', [a for a in dir(mod) if a[0].isalpha()][:12])
    except Exception as e:
        print(m+' FAIL', e)
