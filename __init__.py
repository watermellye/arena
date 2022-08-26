import base64
from random import random
import re
import time
import asyncio
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageColor

import hoshino
from hoshino import Service, R
from hoshino.typing import *
from hoshino.util import FreqLimiter, concat_pic, pic2b64, silence

from .. import chara
from .. import _pcr_data
from .record import update_dic
from os.path import dirname, join, exists
from os import remove
import numpy as np
import json
from io import BytesIO
import requests
import copy

import cv2

sv_help = '''
[怎么拆] 接防守队角色名 查询竞技场解法
[点赞] 接作业id 评价作业
[点踩] 接作业id 评价作业
'''.strip()
sv = Service('pcr-arena', help_=sv_help, bundle='pcr查询')

from . import arena

lmt = FreqLimiter(5)

aliases = ('怎么拆', '怎么解', '怎么打', '如何拆', '如何解', '如何打', 'jjc查询')
aliases_b = tuple('b' + a for a in aliases) + tuple('B' + a for a in aliases)
aliases_b = list(aliases_b)
aliases_b.append('bjjc')
aliases_b = tuple(aliases_b)

aliases_tw = tuple('台' + a for a in aliases)
aliases_tw = list(aliases_tw)
aliases_tw.append('台jjc')
aliases_tw.append('tjjc')
aliases_tw = tuple(aliases_tw)

aliases_jp = tuple('日' + a for a in aliases)
aliases_jp = list(aliases_jp)
aliases_jp.append('日jjc')
aliases_jp.append('rjjc')
aliases_jp = tuple(aliases_jp)

try:
    thumb_up_i = R.img('priconne/gadget/thumb-up-i.png').open().resize((16, 16), Image.LANCZOS)
    thumb_up_a = R.img('priconne/gadget/thumb-up-a.png').open().resize((16, 16), Image.LANCZOS)
    thumb_down_i = R.img('priconne/gadget/thumb-down-i.png').open().resize((16, 16), Image.LANCZOS)
    thumb_down_a = R.img('priconne/gadget/thumb-down-a.png').open().resize((16, 16), Image.LANCZOS)
except Exception as e:
    sv.logger.exception(e)


# 全服=1 b服=2 台服=3 日服=4
@sv.on_prefix(aliases)
async def arena_query(bot, ev):
    await _arena_query(bot, ev, region=1)


@sv.on_prefix(aliases_b)
async def arena_query_b(bot, ev):
    await _arena_query(bot, ev, region=2)


@sv.on_prefix(aliases_tw)
async def arena_query_tw(bot, ev):
    await _arena_query(bot, ev, region=3)


@sv.on_prefix(aliases_jp)
async def arena_query_jp(bot, ev):
    await _arena_query(bot, ev, region=4)


@sv.on_prefix(('testjjc'))
async def arena_query_test(bot, ev):
    await _arena_query(bot, ev, region=-20)


async def render_atk_def_teams(entries, border_pix=5):
    '''
    entries = [ {'atk': [int], 'up': int, 'down': int } ]
    '''
    n = len(entries)
    icon_size = 64
    im = Image.new('RGBA', (5 * icon_size + 100, n * (icon_size + border_pix) - border_pix), (255, 255, 255, 255))
    font = ImageFont.truetype('msyh.ttc', 16)
    draw = ImageDraw.Draw(im)
    for i, e in enumerate(entries):
        if len(e) == 0:
            continue

        y1 = i * (icon_size + border_pix)
        y2 = y1 + icon_size
        ee = True
        if e == "placeholder":
            ee = False
            e = {'atk': [chara.fromid(9000) for _ in range(5)]}

        for j, c in enumerate(e['atk']):
            x1 = j * icon_size
            x2 = x1 + icon_size
            try:
                icon = await c.render_icon(icon_size)  # 如使用旧版hoshino（不返回结果），请去掉await
                im.paste(icon, (x1, y1, x2, y2), icon)
            except:
                icon = c.render_icon(icon_size)
                im.paste(icon, (x1, y1, x2, y2), icon)

        if ee:
            #thumb_up = thumb_up_a if e['user_like'] > 0 else thumb_up_i
            thumb_up = thumb_up_a
            #thumb_down = thumb_down_a if e['user_like'] < 0 else thumb_down_i
            thumb_down = thumb_down_a
            x1 = 5 * icon_size + 10
            x2 = x1 + 16
            im.paste(thumb_up, (x1, y1 + 12, x2, y1 + 28), thumb_up)
            im.paste(thumb_down, (x1, y1 + 39, x2, y1 + 55), thumb_down)
            #draw.text((x1, y1), e['qkey'], (0, 0, 0, 255), font)
            draw.text((x1 + 25, y1 + 10), f"{e['up']}", (0, 0, 0, 255), font)
            #draw.text((x1+25, y1+35), f"{e['down']}+{e['my_down']}" if e['my_down'] else f"{e['down']}", (0, 0, 0, 255), font)
            draw.text((x1 + 25, y1 + 35), f"{e['down']}", (0, 0, 0, 255), font)
    return im


async def getBox(img):
    return await getPos(img)


curpath = dirname(__file__)

dataDir = join(curpath, 'dic.npy')
if not exists(dataDir):
    update_dic()
data = np.load(dataDir, allow_pickle=True).item()
data_processed = None


async def cut_image(image, hash_size=16):
    '''
    将图像缩小成(16+1)*16并转化成灰度图
    :param image: PIL.Image
    :return list[int]
    '''

    image1 = image.resize((hash_size + 1, hash_size), Image.ANTIALIAS).convert('L')
    pixel = list(image1.getdata())
    return pixel


async def trans_hash(lists):
    '''
    比较列表中相邻元素大小
    :param lists: list[int]
    :return list[bool]
    '''
    return [1 if lists[index - 1] > val else 0 for index, val in enumerate(lists)][1:]


async def difference_value(image_lists):
    # 获得图像差异值并获得指纹
    assert len(image_lists) == 17 * 16, "size error"
    m, n = 0, 17
    hash_list = []
    for i in range(0, 16):
        slc = slice(m, n)
        image_slc = image_lists[slc]
        hash_list.append(await trans_hash(image_slc))
        m += 17
        n += 17
    return hash_list


async def get_hash_arr(image):
    return np.array(await difference_value(await cut_image(image)))


async def calc_distance_arr(arr1, arr2):
    return sum(sum(abs(arr1 - arr2)))


async def calc_distance_img(image1, image2):
    return await calc_distance_arr(await get_hash_arr(image2) - await get_hash_arr(image1))


async def process_data():
    global data, data_processed
    data_processed = {}
    for uid in data:
        data_processed[uid] = await get_hash_arr(Image.fromarray(data[uid][25:96, 8:97, :]))


async def cutting(img, mode):
    '''
    :param img: 传入的待识别的原图片 PIL格式
    :param mode: mode=1：返回图片中最大的长方形区域，以及该区域的定位点 [PIL.Image, [x, y, w, h]] ; mode=2: 返回两个列表，列表中每个元素为正方形区域在原图的[x, y, w, h]。第一个列表为聚类结果，第二个列表为排除结果。若无正方形区域，返回[], []。
    '''
    im_grey = img.convert('L')
    totArea = (im_grey.size)[0] * (im_grey.size)[1]
    im_grey = im_grey.point(lambda x: 255 if x > 210 else 0)  # 没有用自带的二值化。考虑修改，使用更合适的函数。
    thresh = np.array(im_grey)

    # cv2.findContours. opencv3版本会返回3个值，opencv2和4只返回后两个值
    #img2, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 获取轮廓
    img2 = thresh

    lis = []
    icon = []  # 每个元素为：[边长，[矩阵左上点坐标，矩阵宽高]]
    # icon_area = {}  # key为边长 value为个数
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # 计算contour包围的像素点个数
        lis.append(area)
        if area > 500:
            if mode == 2:
                x, y, w, h = cv2.boundingRect(contours[i])  # 获取contour的aabb包围盒
                if w / h > 0.95 and w / h < 1.05:  # 近似正方形
                    are = (w + h) // 2
                    areaRatio = are * are / totArea * 100  # 获取该范围占整个输入图像的占比
                    # print(f"{areaRatio:2f}%")
                    if areaRatio >= 0.5:  # 过滤占比小于0.5%的正方形（可能为文字）
                        icon.append([are, [x, y, w, h]])
                        # icon_area[are] = icon_area.setdefault(are, 0) + 1
    # print()
    if mode == 1:
        i = lis.index(max(lis))
        x, y, w, h = cv2.boundingRect(contours[i])
        #cv2.rectangle(img2, (x, y), (x + w, y + h), (153, 153, 0), 5)
        img3 = img2[y + 2:y + h - 2, x + 2:x + w - 2]
        img4 = Image.fromarray(img3)
        return img4, [x, y, w, h]
    if mode == 2:
        if len(icon) == 0:
            return [], []

        # 对边长作简易聚类分析
        kinds = {}  # 边长: [[],[],[],...,[]]
        for i in icon:
            sidelen = i[0]
            category = -1
            for kind in kinds:
                ratio = sidelen / kind
                if ratio > 0.9 and ratio < 1.1:  # 将边长相差10%以内作为一类
                    category = kind
                    kinds[kind].append(i[1])
                    break
            if category == -1:
                kinds[sidelen] = [i[1]]

        def clusterWeight(x):
            sidelen = x[0]
            val = len(x[1])
            if val == 5:  # 该类有5个元素，优先返回。第二关键字为边长（从大到小）。
                return 5000000 + sidelen
            if val % 5 == 0:  # 该类元素个数为5的倍数，次优先返回。
                return 1000000 + sidelen
            return val * 10000 + sidelen

        kinds = sorted(kinds.items(), key=clusterWeight, reverse=True)
        kind = kinds[0]  # (边长, [[],[]])
        if len(kind[1]) % 5 == 0:  # 存在某一类，其元素个数为5的倍数，返回该类
            otherborder = []
            for x in range(1, len(kinds)):
                otherborder += kinds[x][1]
            return kind[1], otherborder
        else:
            return [x[1] for x in icon], []  # 否则返回所有找到的正方形区域，由后续程序进一步判断


async def cut(img, border):
    '''
    :param img: 待裁剪图片 PIL.Image
    :param border: 裁剪范围 [x, y, w, h]
    :return 裁剪后图像 PIL.Image
    '''
    x, y, w, h = border
    img = np.array(img)
    img = img[y + 2:y + h - 2, x + 2:x + w - 2]
    img = Image.fromarray(img)
    return img


async def getPos(img: Image):
    '''
    :param img: 待识别图片 PIL.Image
    :return 识别出的阵容的坐标, 识别出的阵容角色字符串 [[第1队第1个角色(int), 1-2, ..., 1-5], [2-1, ..., 2-5], [3-1, ..., 3-5]], str；若识别失败，返回[], ""
    '''
    img = img.convert("RGBA")
    im_grey = img

    actual_img = img
    actual_x = 0
    actual_y = 0

    nowcolor = 0
    outpImg = Image.new(mode="RGBA", size=img.size, color=ImageColor.getrgb(f'rgb({nowcolor},{nowcolor},{nowcolor})'))
    outpImgText = Image.new("RGBA", img.size, (0, 0, 0, 0))
    outpImgTextDraw = ImageDraw.Draw(outpImgText)

    cnt = 0
    while cnt <= 6:
        bo = False
        cnt += 1
        border, otherborder = await cutting(im_grey, 2)  # 获取当前图像中的正方形区域
        if len(border) == 0:  # 如果莫得 说明需要翻转图像的黑白
            bo = True
        else:
            # print(f'cnt={cnt} border={border}') # test
            # 将正方形区域按照行列分组
            def highlight(rec, color="red"):
                x, y, w, h = rec
                cropped = img.crop([x + 2, y + 2, x + w - 2, y + h - 2])
                outpImgText.paste(cropped, (actual_x + x + 2, actual_y + y + 2, actual_x + x + w - 2, actual_y + y + h - 2))
                outpImgTextDraw.rectangle((actual_x + x, actual_y + y, actual_x + x + w, actual_y + y + h), fill=None, outline=color, width=4)

            for i in otherborder:
                highlight(i, 'blue')
            if len(border) < 4:  # 由5改为4 是为了日后准备加入残缺队伍查询（通过无api版本）
                for i in border:
                    highlight(i, 'blue')
            else:
                recs = border
                recs = set([tuple(rec) for rec in recs])

                def split_last_col_recs(recs):
                    recs = sorted(recs, key=lambda x: x[0], reverse=True)
                    last_col_recs = [rec for rec in recs if abs(rec[0] - recs[0][0]) < recs[0][2] / 2]
                    # last_col_recs = []
                    # for rec in recs:
                    #     if abs(rec[0] - recs[0][0]) < recs[0][2] / 2:
                    #         last_col_recs.append(rec)
                    last_col_recs = sorted(last_col_recs, key=lambda x: x[1])
                    return list(set(recs) - set(last_col_recs)), last_col_recs

                recs, last_col_recs = split_last_col_recs(recs)  # 先找出最右侧的一列有几行
                row_cnt = len(last_col_recs)

                arr = [[None for __ in range(5)] for _ in range(row_cnt)]
                arr_id = [[] for _ in range(row_cnt)]
                arr_id_6 = [[0 for __ in range(5)] for _ in range(row_cnt)]
                for index, rec in enumerate(last_col_recs):
                    highlight(rec)
                    arr[index][0] = rec
                    x, y, w, h = rec
                    cropped = img.crop([x + 2, y + 2, x + w - 2, y + h - 2])
                    uid_6, unit_id, unit_name, similarity = await getUnit(cropped)
                    arr_id[index].append(unit_id)  # 认为最后一列必须要有角色
                    arr_id_6[index][0] = uid_6

                for col_index in range(1, 5):  # 从右往左一列一列掰，最多拿五列
                    recs, last_col_recs = split_last_col_recs(recs)
                    if len(last_col_recs) == 0:
                        break
                    for rec in last_col_recs:
                        # 看看rec能不能被识别出来
                        x, y, w, h = rec
                        cropped = img.crop([x + 2, y + 2, x + w - 2, y + h - 2])
                        uid_6, unit_id, unit_name, similarity = await getUnit(cropped)
                        if unit_id == 0:
                            continue
                        highlight(rec)

                        most_near_row = 0
                        for row_index in range(1, len(arr)):
                            if abs(arr[row_index][0][1] - rec[1]) < abs(arr[most_near_row][0][1] - rec[1]):
                                most_near_row = row_index
                        if arr[most_near_row][col_index] is None or abs(arr[most_near_row][0][1] - arr[most_near_row][col_index][1]) > abs(arr[most_near_row][0][1] - rec[1]):
                            arr[most_near_row][col_index] = rec
                            arr_id[most_near_row].append(unit_id)
                            arr_id_6[most_near_row][col_index] = uid_6

                for rec in recs:
                    highlight(rec, "green")

                # 创建一个 rowcnt行 5列 的画布，行间及四周留16px空隙，每行中的每列分为上下两个头像：截出来的和通过识别的id render出来的（均为64*64)。头像间隙0px。
                icon_size = 64
                compare_img = Image.new("RGBA", (icon_size * 5 + 16 * 2, icon_size * 2 * row_cnt + 16 * (row_cnt + 1)), (255, 255, 255, 255))

                for row_index in range(row_cnt):
                    for col_index in range(5):
                        if arr[row_index][4 - col_index] is None:
                            continue
                        pos_x = 16 + icon_size * col_index
                        pos_y = 16 * (row_index + 1) + icon_size * 2 * row_index
                        x, y, w, h = arr[row_index][4 - col_index]
                        cropped = img.crop([x + 2, y + 2, x + w - 2, y + h - 2]).resize((64, 64), Image.ANTIALIAS)
                        compare_img.paste(cropped, (pos_x, pos_y), cropped)  # 要不要加cropped
                        c = chara.fromid(arr_id_6[row_index][4 - col_index] // 100, arr_id_6[row_index][4 - col_index] % 100 // 10)
                        icon = await c.render_icon(icon_size)
                        compare_img.paste(icon, (pos_x, pos_y + 64), icon)

                def outp_b64(outp_img):
                    buf = BytesIO()
                    outp_img.save(buf, format='PNG')
                    base64_str = f'base64://{base64.b64encode(buf.getvalue()).decode()}'
                    return f'[CQ:image,file={base64_str}]'

                outpImg = Image.blend(outpImg, actual_img, 0.2)
                outpImg.alpha_composite(outpImgText)

                return arr_id, f'{outp_b64(outpImg)}识别阵容为：{outp_b64(compare_img)}'

        try:
            im_grey, border = await cutting(im_grey, 1)  # 获取图片中最大的长方形区域
        except:
            return [], ""
        if cnt == 1 or bo:  # 如果是第一次裁剪，或者识别不到正方形区域，反色
            im_grey = im_grey.point(lambda x: 0 if x > 128 else 255)
        img = await cut(img, border)  # 将原始img也裁剪，和im_grey同步
        actual_x += border[0]
        actual_y += border[1]

        nowcolor = (nowcolor + 60) % 300
        outpImg.paste(ImageColor.getrgb(f'rgb({nowcolor},{nowcolor},{nowcolor})'), (actual_x, actual_y, actual_x + border[2], actual_y + border[3]))

        # outpImg.show()  # test

    return [], ""


async def getUnit(img2):
    img2 = img2.convert("RGB").resize((128, 128), Image.ANTIALIAS)
    img3 = np.array(img2)
    img4 = img3[25:96, 8:97, :]
    img4 = Image.fromarray(img4)
    dic = {}
    global data_processed
    if data_processed == None:
        await process_data()

    img4_arr = await get_hash_arr(img4)
    for uid in data_processed:
        dic[uid] = await calc_distance_arr(data_processed[uid], img4_arr)

    lis = list(sorted(dic.items(), key=lambda x: abs(x[1])))

    similarity = int(lis[0][1])
    if similarity > 90:  # 没一个相似的
        return 0, 0, "Unknown", 100 - similarity
    uid_6 = int(lis[0][0])
    uid = uid_6 // 100
    try:
        return uid_6, uid, chara.fromid(uid).name, 100 - similarity
    except:
        return uid_6, uid, "Unknown", 100 - similarity


async def get_pic(address):
    return requests.get(address, timeout=20).content


async def _arena_query(bot, ev: CQEvent, region: int):
    arena.refresh_quick_key_dic()
    uid = ev.user_id

    if not lmt.check(uid):
        await bot.finish(ev, '您查询得过于频繁，请稍等片刻', at_sender=True)

    # 处理输入数据
    defen = ""
    ret = re.match(r"\[CQ:image,file=(.*),url=(.*)\]", str(ev.message))
    if ret:
        # await bot.send(ev, "recognizing")
        image = Image.open(BytesIO(await get_pic(ret.group(2))))
        boxDict, s = await getBox(image)

        if boxDict == []:
            await bot.finish(ev, "未识别到角色！")

        try:
            await bot.send(ev, s)
        except:
            pass

        if region == -20:
            return

        lis = []  # [[[第1队第1解],[第1队第2解]], [[第2队第1解]], []]
        if len(boxDict) == 1:
            await __arena_query(bot, ev, region, boxDict[0])
            return
        if len(boxDict) > 3:
            await bot.finish(ev, "请截图pjjc详细对战记录（对战履历详情）（含敌我双方2或3队阵容）")
        tot = 0
        lmt.start_cd(uid)
        for i in boxDict:
            li = []
            res = await __arena_query(bot, ev, region, i, 1)
            # print(res)
            if res == []:
                lis.append([])
            else:
                tot += 1
                for num, squad in enumerate(res):
                    soutp = ""
                    squads = []  # [int int int int int 评价 string 原始阵容]
                    for nam in squad["atk"]:
                        # print(nam)
                        soutp += nam.name + " "
                        squads.append(nam.id)
                    #squads.append(int(squad["up"]) - int(squad["down"]))
                    # squads.append(num)
                    squads.append(int(squad["up"]) * 10 / (int(squad["down"] + int(squad["up"]) + 1)) + random() / 100)
                    squads.append(soutp[:-1])
                    squads.append(copy.deepcopy(squad))
                    li.append(copy.deepcopy(squads))
                lis.append(copy.deepcopy(li))
            await asyncio.sleep(2)
        # print(lis)
        if tot == 0:
            await bot.finish(ev, "均未查询到解法！")
        if tot == 1:
            for num, i in enumerate(lis):
                if len(i) > 0:
                    await bot.send(ev, f"仅第{num+1}队查询到解法！")
                    await __arena_query(bot, ev, region, boxDict[num], only_use_cache=True)  # 历史遗留性质的偷懒，好在使用了缓存
            return
        le = len(lis)
        outp = []
        outp_priority = []
        outp_img = []
        cnt = 0
        if le == 3:
            s1 = lis[0]
            s2 = lis[1]
            s3 = lis[2]
            for x in s1:
                for y in s2:
                    for z in s3:
                        temp = x[:-3] + y[:-3] + z[:-3]
                        if len(temp) == len(set(temp)):
                            cnt += 1
                            if cnt <= 8:
                                outp_img.append([x[-1], y[-1], z[-1]])
                                outp.append(f"第{1}队：{x[-2]}\n第{2}队：{y[-2]}\n第{3}队：{z[-2]}\n")
                                outp_priority.append(-(x[-3] + y[-3] + z[-3]))
                                #outp += f"优先级：{x[-2]+y[-2]+z[-2]:03.1f}\n第{1}队：{x[-1]}\n第{2}队：{y[-1]}\n第{3}队：{z[-1]}\n"

        if outp != []:
            outp_priority, outp, outp_img = zip(*sorted(zip(outp_priority, outp, outp_img)))
            outp_render = []
            for i in outp_img:
                outp_render += i
                outp_render.append([])
            # for i in range(len(outp_priority)):
            #     print(f'优先级={outp_priority[i]:03.1f}\n阵容=\n{outp[i]}\n原始数据={outp_img[i]}')
            #outp = "三队无冲配队：\n" + '\n'.join(outp)
            # await bot.finish(ev, outp)
            teams = await render_atk_def_teams(outp_render[:-1])
            teams = pic2b64(teams)
            teams = MessageSegment.image(teams)
            await bot.finish(ev, str(teams))

        for i in range(le - 1):
            for j in range(i + 1, le):
                s1 = lis[i]
                s2 = lis[j]
                for x in s1:
                    for y in s2:
                        if not (set(x[:-3]) & set(y[:-3])):
                            cnt += 1
                            if cnt < 8:
                                if le == 2:
                                    outp_img_space = ["placeholder", "placeholder"]
                                else:
                                    outp_img_space = ["placeholder", "placeholder", "placeholder"]
                                outp_img_space[i] = x[-1]
                                outp_img_space[j] = y[-1]
                                outp_img.append(copy.deepcopy(outp_img_space))
                                outp.append(f"第{i+1}队：{x[-2]}\n第{j+1}队：{y[-2]}\n")
                                outp_priority.append(-(x[-3] + y[-3]))
                                #outp += f"优先级：{x[-2]+y[-2]:03.1f}\n第{i+1}队：{x[-1]}\n第{j+1}队：{y[-1]}\n"
        if outp != []:
            outp_priority, outp, outp_img = zip(*sorted(zip(outp_priority, outp, outp_img)))
            outp_render = []
            for i in outp_img:
                outp_render += i
                outp_render.append([])
            # for i in range(len(outp_priority)):
            #     print(f'优先级={outp_priority[i]:03.1f}\n阵容=\n{outp[i]}\n原始数据={outp_img[i]}')
            #outp = "三队无冲配队：\n" + '\n'.join(outp)
            # await bot.finish(ev, outp)
            teams = await render_atk_def_teams(outp_render[:-1])
            teams = pic2b64(teams)
            teams = MessageSegment.image(teams)
            await bot.finish(ev, str(teams))

        outp = "不存在无冲配队！"
        for num, i in enumerate(lis):
            if i != []:
                outp += f"\n第{num+1}队的解法为：\n"
                for j in i:
                    outp += j[-2] + "\n"
        await bot.finish(ev, outp.strip())

    else:
        await __arena_query(bot, ev, region)


def remove_buffer(uid: str):
    curpath = dirname(__file__)
    bufferpath = join(curpath, 'buffer/buffer.json')

    buffer = {}
    with open(bufferpath, 'r', encoding="utf-8") as fp:
        buffer = json.load(fp)

    try:
        remove(join(curpath, f'buffer/{uid}.json'))
    except:
        pass

    if uid in buffer:
        buffer.pop(uid)
        with open(bufferpath, 'w', encoding="utf-8") as fp:
            json.dump(buffer, fp, ensure_ascii=False, indent=4)


async def __arena_query(bot, ev: CQEvent, region: int, defen="", raw=0, only_use_cache=False):
    uid = ev.user_id
    unknown = ""
    if defen == "":
        defen = ev.message.extract_plain_text()
        defen = re.sub(r'[?？，,_]', '', defen)
        defen, unknown = chara.roster.parse_team(defen)

    if unknown:
        _, name, score = chara.guess_id(unknown)
        if score < 70 and not defen:
            return  # 忽略无关对话
        msg = f'无法识别"{unknown}"' if score < 70 else f'无法识别"{unknown}" 您说的有{score}%可能是{name}'
        await bot.finish(ev, msg)
    if not defen:
        await bot.finish(ev, '查询请发送"b/r/tjjc+防守队伍"，无需+号', at_sender=True)
    if len(defen) > 5:
        await bot.finish(ev, '编队不能多于5名角色', at_sender=True)
    if len(defen) < 5:
        await bot.finish(ev, '由于数据库限制，少于5名角色的检索条件请移步pcrdfans.com进行查询', at_sender=True)
    if len(defen) != len(set(defen)):
        await bot.finish(ev, '编队中含重复角色', at_sender=True)
    if any(chara.is_npc(i) for i in defen):
        await bot.finish(ev, '编队中含未实装角色', at_sender=True)
    if 1004 in defen:
        await bot.send(ev, '\n⚠️您正在查询普通版炸弹人\n※万圣版可用万圣炸弹人/瓜炸等别称', at_sender=True)

    key = ''.join([str(x) for x in sorted(defen)]) + str(region)
    # 执行查询
    lmt.start_cd(uid)
    res = await arena.do_query(defen, uid, region, raw, -1 if only_use_cache else 1)

    # 处理查询结果
    if res is None:
        remove_buffer(key)
        if not raw:
            await bot.finish(ev, '数据库未返回数据，请再次尝试查询或前往pcrdfans.com', at_sender=True)
        else:
            return []
    if not len(res):
        remove_buffer(key)
        if not raw:
            await bot.finish(ev, '抱歉没有查询到解法\n作业上传请前往pcrdfans.com', at_sender=True)
        else:
            return []

    res = res[:min(10, len(res))]  # 限制显示数量，截断结果
    if raw:
        return res
    # print(res)

    # 发送回复
    sv.logger.info('Arena generating picture...')
    teams = await render_atk_def_teams(res)
    teams = pic2b64(teams)
    teams = MessageSegment.image(teams)
    sv.logger.info('Arena picture ready!')
    # 纯文字版
    # atk_team = '\n'.join(map(lambda entry: ' '.join(map(lambda x: f"{x.name}{x.star if x.star else ''}{'专' if x.equip else ''}" , entry['atk'])) , res))

    # details = [" ".join([
    #     f"赞{e['up']}+{e['my_up']}" if e['my_up'] else f"赞{e['up']}",
    #     f"踩{e['down']}+{e['my_down']}" if e['my_down'] else f"踩{e['down']}",
    #     e['qkey'],
    #     "你赞过" if e['user_like'] > 0 else "你踩过" if e['user_like'] < 0 else ""
    # ]) for e in res]

    defen = [chara.fromid(x).name for x in defen]
    defen = f"防守方【{' '.join(defen)}】"
    # at = str(MessageSegment.at(ev.user_id))

    msg = [
        defen,
        str(teams),
        # '作业评价：',
        # *details,
        # '※发送"点赞/点踩"可进行评价'
    ]
    if region == 1:
        msg.append('※使用"b怎么拆"或"台怎么拆"可按服过滤')
    # msg.append('https://www.pcrdfans.com/battle')

    sv.logger.debug('Arena sending result...')
    await bot.send(ev, '\n'.join(msg))
    sv.logger.debug('Arena result sent!')


# @sv.on_prefix('点赞')
async def arena_like(bot, ev):
    await _arena_feedback(bot, ev, 1)


# @sv.on_prefix('点踩')
async def arena_dislike(bot, ev):
    await _arena_feedback(bot, ev, -1)


rex_qkey = re.compile(r'^[0-9a-zA-Z]{5}$')


async def _arena_feedback(bot, ev: CQEvent, action: int):
    action_tip = '赞' if action > 0 else '踩'
    qkey = ev.message.extract_plain_text().strip()
    if not qkey:
        await bot.finish(ev, f'请发送"点{action_tip}+作业id"，如"点{action_tip}ABCDE"，不分大小写', at_sender=True)
    if not rex_qkey.match(qkey):
        await bot.finish(ev, f'您要点{action_tip}的作业id不合法', at_sender=True)
    try:
        await arena.do_like(qkey, ev.user_id, action)
    except KeyError:
        await bot.finish(ev, '无法找到作业id！您只能评价您最近查询过的作业', at_sender=True)
    await bot.send(ev, '感谢您的反馈！', at_sender=True)


@sv.on_fullmatch('竞技场更新卡池')
async def _update_dic(bot, ev):
    try:
        await bot.send(ev, f'{update_dic()}')
        global data
        data = np.load(dataDir, allow_pickle=True).item()
        await process_data()
    except Exception as e:
        await bot.send(ev, f'Error: {e}')


@sv.scheduled_job('cron', hour='3', minute='21')
async def _update_dic_cron():
    try:
        update_dic()
        global data
        data = np.load(dataDir, allow_pickle=True).item()
        await process_data()
    except Exception as e:
        pass


@sv.on_command('arena-upload', aliases=('上传作业', '作业上传', '上傳作業', '作業上傳'))
async def upload(ss: CommandSession):
    atk_team = ss.get('atk_team', prompt='请输入进攻队+5个表示星级的数字+5个表示专武的0/1 无需空格')
    def_team = ss.get('def_team', prompt='请输入防守队+5个表示星级的数字+5个表示专武的0/1 无需空格')
    if 'pic' not in ss.state:
        ss.state['pic'] = MessageSegment.image(pic2b64(concat_pic([
            chara.gen_team_pic(atk_team),
            chara.gen_team_pic(def_team),
        ])))
    confirm = ss.get('confirm', prompt=f'{ss.state["pic"]}\n{MessageSegment.at(ss.event.user_id)}确认上传？\n> 确认\n> 取消')
    # TODO: upload
    await ss.send('假装上传成功了...')


@upload.args_parser
async def _(ss: CommandSession):
    if ss.is_first_run:
        await ss.send('我将帮您上传作业至pcrdfans，作业将注明您的昵称及qq。您可以随时发送"算了"或"取消"终止上传。')
        await asyncio.sleep(0.5)
        return
    arg = ss.current_arg_text.strip()
    if arg == '算了' or arg == '取消':
        await ss.finish('已取消上传')

    if ss.current_key.endswith('_team'):
        if len(arg) < 15:
            return
        team, star, equip = arg[:-10], arg[-10:-5], arg[-5:]
        if not re.fullmatch(r'[1-6]{5}', star):
            await ss.pause('请依次输入5个数字表示星级，顺序与队伍相同')
        if not re.fullmatch(r'[01]{5}', equip):
            await ss.pause('请依次输入5个0/1表示专武，顺序与队伍相同')
        star = [int(s) for s in star]
        equip = [int(s) for s in equip]
        team, unknown = chara.roster.parse_team(team)
        if unknown:
            _, name, score = chara.guess_id(unknown)
            await ss.pause(f'无法识别"{unknown}"' if score < 70 else f'无法识别"{unknown}" 您说的有{score}%可能是{name}')
        if len(team) != 5:
            await ss.pause('队伍必须由5个角色组成')
        ss.state[ss.current_key] = [chara.fromid(team[i], star[i], equip[i]) for i in range(5)]
    elif ss.current_key == 'confirm':
        if arg == '确认' or arg == '確認':
            ss.state[ss.current_key] = True
    else:
        raise ValueError
    return
