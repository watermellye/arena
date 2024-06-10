from hoshino import Service
from hoshino.typing import *

from .. import chara
from .. import _pcr_data
from .qq_context_requests import *

sv_help = f'''
查询请发送"bjjc/rjjc/tjjc+防守队伍"，无需+号。可以分开发送。
防守队伍可以是五个角色的昵称，也可以是截图。
截图支持局部图片和全局图片，支持多队查询，支持未定队伍查询，支持近似查询。
截图与bjjc/rjjc/tjjc可以分开发送。
源码：https://github.com/watermellye/arena
'''.strip()

sv = Service('pcr-arena', help_=sv_help, bundle='pcr查询')

gs_prefix_all = ('怎么拆', '怎么解', '怎么打', '如何拆', '如何解', '如何打', 'jjc查询')
gs_prefix_bilibili = tuple(["bjjc"] + ['b' + x for x in gs_prefix_all] + ['B' + x for x in gs_prefix_all])
gs_prefix_taiwan = tuple(["tjjc"] + ['t' + x for x in gs_prefix_all] + ['T' + x for x in gs_prefix_all])
gs_prefix_japan = tuple(["rjjc"] + ['r' + x for x in gs_prefix_all] + ['R' + x for x in gs_prefix_all])

def IsEmptyMessage(message: Message) -> bool:
    return all([x.type == 'text' and x.data['text'].strip() == '' for x in message])


@sv.on_prefix(gs_prefix_all)
async def QueryAllInterface(bot: HoshinoBot, ev: CQEvent):
    if IsEmptyMessage(ev.message):
        await bot.send(ev, sv.help)
    else:
        await QueryArenaInterface(bot, ev, ev.message, RegionEnum.All)
        await bot.send(ev, f'请使用 bjjc/rjjc/tjjc 以过滤查询的服务器。')


@sv.on_prefix(gs_prefix_bilibili)
async def QueryBilibiliInterface(bot: HoshinoBot, ev: CQEvent):
    if IsEmptyMessage(ev.message):
        await bot.send(ev, f'已收到查作业（B服）请求，请在 {gs_seconds_to_wait} 秒内发送防守队伍截图')
        gs_qqid2request[ev.user_id] = QueryRequestContext(RegionEnum.Bilibili)
    else:
        await QueryArenaInterface(bot, ev, ev.message, RegionEnum.Bilibili)


@sv.on_prefix(gs_prefix_taiwan)
async def QueryTaiwanInterface(bot: HoshinoBot, ev: CQEvent):
    if IsEmptyMessage(ev.message):
        await bot.send(ev, f'已收到查作业（台服）请求，请在 {gs_seconds_to_wait} 秒内发送防守队伍截图')
        gs_qqid2request[ev.user_id] = QueryRequestContext(RegionEnum.Taiwan)
    else:
        await QueryArenaInterface(bot, ev, ev.message, RegionEnum.Taiwan)


@sv.on_prefix(gs_prefix_japan)
async def QueryJapanInterface(bot: HoshinoBot, ev: CQEvent):
    if IsEmptyMessage(ev.message):
        await bot.send(ev, f'已收到查作业（日服）请求，请在 {gs_seconds_to_wait} 秒内发送防守队伍截图')
        gs_qqid2request[ev.user_id] = QueryRequestContext(RegionEnum.Japan)
    else:
        await QueryArenaInterface(bot, ev, ev.message, RegionEnum.Japan)


def GetImageUrlFromMessage(message: Message) -> Optional[str]:
    images = [x.data for x in message if x.type == 'image']
    url = images[0].get("url", None) if len(images) else None
    if url is not None:
        url = url.replace("&amp;", "&").split(",file_size=")[0] # temp
    return url


async def QueryArenaInterface(bot: HoshinoBot, ev: CQEvent, msg: Message, region: RegionEnum):
    image_url = GetImageUrlFromMessage(msg)
    if image_url:
        await QueryArenaImageAsync(image_url, region, bot, ev)
    else:
        await QueryArenaTextAsync(msg.extract_plain_text().strip(), region, bot, ev)
    
@sv.on_message('group')
async def QueryArenaGroupMessageContextInterface(bot: HoshinoBot, ev: CQEvent):
    await QueryArenaMessageContextInterface(bot, ev)

@sv.on_message('private')
async def QueryArenaPrivateMessageContextInterface(bot: HoshinoBot, ev: CQEvent):
    await QueryArenaMessageContextInterface(bot, ev)
    
async def QueryArenaMessageContextInterface(bot: HoshinoBot, ev: CQEvent):
    image_url = GetImageUrlFromMessage(ev.message)
    if not image_url:
        return

    req = PopRequest(ev.user_id)
    if req is None:
        return
    
    await QueryArenaImageAsync(image_url, req.region, bot, ev)

async def QueryArenaImageAsync(image_url: str, region: RegionEnum, bot: HoshinoBot, ev: CQEvent) -> None:
    from .old_main import _QueryArenaImageAsync
    await _QueryArenaImageAsync(image_url, Region2Int(region), bot, ev)

async def QueryArenaTextAsync(text: str, region: RegionEnum, bot: HoshinoBot, ev: CQEvent) -> None:
    from .old_main import _QueryArenaTextAsync
    await _QueryArenaTextAsync(text, Region2Int(region), bot, ev)
    
def Region2Int(region: RegionEnum) -> int:
    if region == RegionEnum.All:
        return 1
    if region == RegionEnum.Bilibili:
        return 2
    if region == RegionEnum.Taiwan:
        return 3
    if region == RegionEnum.Japan:
        return 4
    return -1