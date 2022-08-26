import asyncio
import base64
import os
import time
from collections import defaultdict

from hoshino import aiorequests, config, util

from .. import chara
from . import sv

try:
    import ujson as json
except:
    import json

from os.path import dirname, join, exists
from os import remove
from asyncio import Lock

querylock = Lock()

logger = sv.logger
"""
Database for arena likes & dislikes
DB is a dict like: { 'md5_id': {'like': set(qq), 'dislike': set(qq)} }
"""
DB_PATH = os.path.expanduser("~/.hoshino/arena_db.json")
DB = {}
try:
    with open(DB_PATH, encoding="utf8") as f:
        DB = json.load(f)
    for k in DB:
        DB[k] = {
            "like": set(DB[k].get("like", set())),
            "dislike": set(DB[k].get("dislike", set())),
        }
except FileNotFoundError:
    logger.warning(f"arena_db.json not found, will create when needed.")


def dump_db():
    """
    Dump the arena databese.
    json do not accept set object, this function will help to convert.
    """
    j = {}
    for k in DB:
        j[k] = {
            "like": list(DB[k].get("like", set())),
            "dislike": list(DB[k].get("dislike", set())),
        }
    with open(DB_PATH, "w", encoding="utf8") as f:
        json.dump(j, f, ensure_ascii=False)


def get_likes(id_):
    return DB.get(id_, {}).get("like", set())


def add_like(id_, uid):
    e = DB.get(id_, {})
    l = e.get("like", set())
    k = e.get("dislike", set())
    l.add(uid)
    k.discard(uid)
    e["like"] = l
    e["dislike"] = k
    DB[id_] = e


def get_dislikes(id_):
    return DB.get(id_, {}).get("dislike", set())


def add_dislike(id_, uid):
    e = DB.get(id_, {})
    l = e.get("like", set())
    k = e.get("dislike", set())
    l.discard(uid)
    k.add(uid)
    e["like"] = l
    e["dislike"] = k
    DB[id_] = e


_last_query_time = 0
quick_key_dic = {}  # {quick_key: true_id}


def refresh_quick_key_dic():
    global _last_query_time
    now = time.time()
    if now - _last_query_time > 300:
        quick_key_dic.clear()
    _last_query_time = now


def gen_quick_key(true_id: str, user_id: int) -> str:
    qkey = int(true_id[-6:], 16)
    while qkey in quick_key_dic and quick_key_dic[qkey] != true_id:
        qkey = (qkey + 1) & 0xFFFFFF
    quick_key_dic[qkey] = true_id
    mask = user_id & 0xFFFFFF
    qkey ^= mask
    return base64.b32encode(qkey.to_bytes(3, "little")).decode()[:5]


def get_true_id(quick_key: str, user_id: int) -> str:
    mask = user_id & 0xFFFFFF
    if not isinstance(quick_key, str) or len(quick_key) != 5:
        return None
    qkey = (quick_key + "===").encode()
    qkey = int.from_bytes(base64.b32decode(qkey, casefold=True, map01=b"I"), "little")
    qkey ^= mask
    return quick_key_dic.get(qkey, None)


def __get_auth_key():
    return config.priconne.arena.AUTH_KEY


async def do_query(id_list, user_id, region=1, raw=0, try_cnt=1):
    defen = id_list
    key = ''.join([str(x) for x in sorted(defen)]) + str(region)
    if try_cnt <= 1:
        print()
    if try_cnt != -1:
        logger.info(f'查询阵容：{key} try_cnt={try_cnt}')
    else:
        logger.info(f'查询阵容：{key} 仅使用缓存')
    value = int(time.time())

    curpath = dirname(__file__)
    bufferpath = join(curpath, 'buffer/buffer.json')

    buffer = {}
    with open(bufferpath, 'r', encoding="utf-8") as fp:
        buffer = json.load(fp)

    if (value - buffer.get(key, 0) < 3600 * 24) and (exists(join(curpath, f'buffer/{key}.json'))):  # 24h内查询过 直接返回
        logger.info(f'    存在本服({region})近缓存，直接使用')
        with open(join(curpath, f'buffer/{key}.json'), 'r', encoding="utf-8") as fp:
            result = json.load(fp)
    else:
        degrade_result = None
        if try_cnt <= 1:
            if exists(join(curpath, f'buffer/{key}.json')):
                logger.info(f'    存在本服({region})远缓存，作为降级备用')
                with open(join(curpath, f'buffer/{key}.json'), 'r', encoding="utf-8") as fp:
                    degrade_result = json.load(fp)
            else:
                logger.info(f'    不存在本服({region})缓存，查找它服缓存')
                query_seq = {
                    1: [2, 4, 3],  # 全服查询顺序为[B,日,台]
                    2: [1, 3, 4],  # B服查询顺序为[全,台,日]
                    3: [1, 2, 4],  # 台服查询顺序为[全,B,日]
                    4: [1, 3, 2]  # 日服查询顺序为[全,台,B]
                }

                query_seq = query_seq.get(region, [])
                for other_region in query_seq:
                    other_key = ''.join([str(x) for x in sorted(defen)]) + str(other_region)
                    if exists(join(curpath, f'buffer/{other_key}.json')):
                        logger.info(f'        存在它服({other_region})缓存，作为降级备用')
                        with open(join(curpath, f'buffer/{other_key}.json'), 'r', encoding="utf-8") as fp:
                            degrade_result = json.load(fp)
                        break
                else:
                    logger.info(f'        不存在它服缓存')
        if try_cnt == -1:
            result = degrade_result if degrade_result else []
        else:
            id_list_query = [x * 100 + 1 for x in id_list]
            header = {
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36",
                "authorization": __get_auth_key(),
            }
            payload = {
                "_sign": "a",
                "def": id_list_query,
                "nonce": "a",
                "page": 1,
                "sort": 1,
                "ts": int(time.time()),
                "region": region,
            }

            query_again = False
            should_sleep = False
            if querylock.locked():
                should_sleep = True  # 旨在不要连续调用api
            async with querylock:
                if should_sleep:
                    await asyncio.sleep(3)
                res = None
                try:
                    resp = await aiorequests.post(
                        "https://api.pcrdfans.com/x/v1/search",
                        headers=header,
                        json=payload,
                        timeout=5,
                    )
                    res = await resp.json()
                    logger.info("    服务器有返回")
                    if res["code"]:
                        logger.info(f'        服务器报错：返回值{res["code"]}')
                        raise Exception()
                    result = res["data"]["result"]
                except:
                    if degrade_result:
                        logger.info("    查询失败，使用缓存")
                        result = degrade_result
                    else:
                        if try_cnt < 2:
                            logger.info("    查询失败，再次查询")
                            query_again = True
                        else:
                            logger.info("    查询失败，返回None")
                            return None
                else:
                    logger.info(f'    查询成功，共有{len(result)}条结果')
                    if len(result):
                        logger.info("        保存结果至缓存库")
                        buffer[key] = value

                        with open(bufferpath, 'w', encoding="utf-8") as fp:
                            json.dump(buffer, fp, ensure_ascii=False, indent=4)

                        homeworkpath = join(curpath, f'buffer/{key}.json')
                        with open(homeworkpath, 'w', encoding="utf-8") as fp:
                            json.dump(result, fp, ensure_ascii=False, indent=4)
                    else:
                        if degrade_result:
                            logger.info(f'    使用缓存')
                            result = degrade_result
                        else:
                            if try_cnt < 2 and region != 1:
                                logger.info(f'    尝试查询全服结果')
                                region = 1
                                query_again = True
                            else:
                                logger.info(f'    返回[]')
                                return []
            if query_again:
                await asyncio.sleep(1)
                return await do_query(id_list, user_id, region, raw, try_cnt + 1)
    ret = []
    for entry in result:
        eid = entry["id"]
        likes = get_likes(eid)
        dislikes = get_dislikes(eid)
        ret.append({
            "qkey": gen_quick_key(eid, user_id),
            "atk": [chara.fromid(c["id"] // 100, c["star"], c["equip"]) for c in entry["atk"]],
            "def": [chara.fromid(c["id"] // 100, c["star"], c["equip"]) for c in entry["def"]],
            "up": entry["up"],
            "down": entry["down"],
            "my_up": len(likes),
            "my_down": len(dislikes),
            "user_like": 1 if user_id in likes else -1 if user_id in dislikes else 0,
        })
    logger.info(f'    共有{len(ret)}条结果')
    return ret


async def do_like(qkey, user_id, action):
    true_id = get_true_id(qkey, user_id)
    if true_id is None:
        raise KeyError
    add_like(true_id, user_id) if action > 0 else add_dislike(true_id, user_id)
    dump_db()
    # TODO: upload to website
