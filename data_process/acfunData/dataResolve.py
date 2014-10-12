#coding=utf-8
import json
import re
__author__ = 'nkssai'


def json_tranform(str, id, name, split_time = 60, radio = 200):
    ret = {}
    ret['id'] = id
    ret['name'] = name
    ret['ci'] = {}
    try:
        ci_list = json.loads(str)
        sum = 0
        for ci in ci_list:
            try:
                time = float(ci['c'].split(',')[0])
                time_tick = int(time / split_time)
                commit = ci['m']
                commit = re.sub(ur'[^\u4e00-\u9fa5 a-z A-Z]','',ci['m'])
                if time_tick in ret['ci'].keys():
                    ret['ci'][time_tick] += (" " + commit)
                    sum += len(commit)
                else:
                    ret['ci'][time_tick] = commit
                    sum += len(commit)
            except:
                continue
    except:
        ret['sum'] = -1
        return ret

    ret['sum'] = sum / len(ret['ci'].keys())

    return ret


