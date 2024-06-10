from enum import IntEnum, unique
from typing import Dict, Optional
from datetime import datetime

@unique
class RegionEnum(IntEnum):
    Unknown = 1
    Bilibili = 1 << 1
    Taiwan = 1 << 2
    Japan = 1 << 3
    All = 1 << 4

class QueryRequestContext:
    def __init__(self, region: RegionEnum = RegionEnum.Unknown):
        self.time = datetime.now()
        self.region = region
    
gs_qqid2request: Dict[int, QueryRequestContext] = {}
gs_seconds_to_wait = 60
_gs_last_clean_time = datetime.now()

def _ClearOldRequests():
    current_time = datetime.now()
    if (current_time - _gs_last_clean_time).seconds < 599:
        return
    to_delete = [qqid for qqid, context in gs_qqid2request.items() 
                 if (current_time - context.time).seconds >= gs_seconds_to_wait]
    for qqid in to_delete:
        gs_qqid2request.pop(qqid, None)

def GetRequest(qqid: int) -> Optional[QueryRequestContext]:
    _ClearOldRequests()
    
    if qqid not in gs_qqid2request:
        return None
    req = gs_qqid2request[qqid]
    if (datetime.now() - req.time).seconds >= gs_seconds_to_wait:
        return None
    return req

def PopRequest(qqid: int) -> Optional[QueryRequestContext]:
    req = GetRequest(qqid)
    gs_qqid2request.pop(qqid, None)
    return req
