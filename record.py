import numpy as np
from PIL import Image
import os
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
            chara_obj = chara.fromid(icon_id // 100)
            if chara_obj.is_npc:
                continue
            icon_list.append([file, icon_id])
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
