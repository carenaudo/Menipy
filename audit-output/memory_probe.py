"""Measure native Windows working set for video decoding in a fresh process."""
import ctypes
from ctypes import wintypes
import json
import sys
import time
from isolation import isolate, OUT
count=int(sys.argv[1])
state=isolate(f'memory-{count}')
import cv2
import numpy as np
from menipy.common.sequence_acquisition import load_video

class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
       ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
        'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=wintypes.HANDLE
psapi=ctypes.WinDLL('psapi'); psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
def memory():
    c=Counters(); c.cb=ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb): raise ctypes.WinError()
    return {'working_set_mib':c.WorkingSetSize/2**20,'process_peak_working_set_mib':c.PeakWorkingSetSize/2**20}
path=state/'synthetic.avi'
writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30.0,(640,480))
if not writer.isOpened(): raise RuntimeError('MJPG encoder unavailable')
base=np.full((480,640,3),230,dtype=np.uint8)
cv2.circle(base,(320,300),100,(20,20,20),-1)
for i in range(count): writer.write(base)
writer.release()
before=memory(); start=time.perf_counter()
frames,metadata=load_video(path)
elapsed=time.perf_counter()-start
result={'frames':len(frames),'shape':[480,640,3],'fps':metadata.fps,'elapsed_ms':elapsed*1000,
 'retained_pixel_mib':sum(f.image.nbytes for f in frames)/2**20,'before':before,'after':memory(),
 'encoded_bytes':path.stat().st_size,'note':'Synthetic MJPG; peak is process lifetime, not just decoder allocations.'}
(OUT/f'memory-{count}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
