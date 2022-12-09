import numpy as np
from PIL import Image
import os
from os.path import dirname, join
import json
from hoshino import R
from .. import chara
import re


def update_dic():
    nowpath = os.path.abspath(R.get('img/priconne/unit/').path)
    dic = {}
    icon_list = []
    for file in os.listdir(nowpath):
        try:
            ret = re.match(r"^icon_unit_(\d{6}).png$", file)
            icon_id = int(ret.group(1))
            if (1000 <= icon_id // 100 < 1300) or (1700 < icon_id // 100 < 1900):  # is_npc的判定是(1000,1900)
                icon_list.append([file, icon_id])  # 但此处我们需要识别问号（pjjc用），因此设置为[1000, 1900)
        except:
            continue
    msg = [f'共检测到{len(icon_list)}个pcr头像']
    cnt_success = 0
    for file, icon_id in icon_list:
        try:
            img = Image.open(os.path.join(nowpath, file))
            img = img.convert("RGB")
            img = img.resize((128, 128))
            dic[icon_id] = np.array(img)
            cnt_success += 1
        except Exception as e:
            msg.append(f'Warning. 头像{icon_id}加入识别库失败：{e}')
    np.save(os.path.join(os.path.dirname(__file__), "dic"), dic)
    if cnt_success:
        msg.append(f'Succeed. 更新成功。共收录{cnt_success}个头像进入识别库')
    return '\n'.join(msg)


def update_record():
    curpath = dirname(__file__)
    bufferpath = join(curpath, 'buffer/')

    buffer_region_cnt = [None, {}, {}, {}, {}]  # 全服=1 b服=2 台服=3 日服=4
    tot_file_cnt = len(os.listdir(bufferpath))
    for index, filename in enumerate(os.listdir(bufferpath)):  # 我为什么不用buffer.json 我是猪鼻
        # if index % 100 == 0:
        #     print(f'{index:5d}/{tot_file_cnt}')  # test

        if len(filename) != 26:
            continue

        try:
            region = int(filename[-6])
            if region == 1:  # 按理说全服查询可能出现任何角色，应该归入4
                region = 2  # 但本bot 绝大多数的全服查询实际均为国服，因此归入国服。少数的其它服查询在频率排序后会被滤过。
            if region not in [1, 2, 3, 4]:
                continue
        except:
            continue

        try:
            filepath = join(bufferpath, filename)
            with open(filepath, "r", encoding="utf-8") as fp:
                records = json.load(fp)

            for record in records:
                if "atk" in record:
                    unit_id_list = tuple([(unit.get("id", 100001) + unit.get("star", 3) * 10) for unit in record["atk"]])
                    buffer_region_cnt[region][unit_id_list] = 1 + buffer_region_cnt[region].get(unit_id_list, 0)
        except:
            continue

    best_atk_records_item = sorted(buffer_region_cnt[2].items(), key=lambda x: x[1], reverse=True)[:200]
    best_atk_records = [x[0] for x in best_atk_records_item]
    with open(join(bufferpath, "best_atk_records.json"), "w", encoding="utf-8") as fp:
        json.dump(best_atk_records, fp, ensure_ascii=False, indent=4)

    return f'从{tot_file_cnt}个文件中搜索到{len(buffer_region_cnt[2])}个进攻阵容（不计日台服查询）\n已缓存最频繁使用的{len(best_atk_records)}个阵容'
