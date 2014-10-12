from acfunData import dataResolve as dr
import conf
import os
import json
import logging

__author__ = 'nkssai'


def main():
    logging.debug("start")
    all_video = os.listdir(conf.input_dir)
    if len(all_video) == 0:
        logging.warn("empty input dir")
        return


    tmp_dir = make_tmp_dir()
    print tmp_dir
    json_f = open(tmp_dir + "/all_json.txt", 'w')

    #for every video
    for video in all_video:
        id = video
        dir_path = conf.input_dir + video
        if os.path.isfile(dir_path):
            continue
        inter_dir = os.listdir(dir_path)

        if len(inter_dir) != 1:
            logging.debug("dir " + video + " is empty")
            continue
        name = inter_dir[0]



        f_list = os.listdir(dir_path + "/" + name)
        if len(f_list) < 2:
            logging.debug(id + " is incompleted")
            continue

        json_path = dir_path + "/" + name + "/"

        # for json
        ret = None
        for f in f_list:
            l = f.split('.')
            if l[-1] == "json":
                ret = json_resolve(json_path + f, name, id)
                if ret != None:
                    json_f.write(ret + "\n")

                break

        if ret == None:
            logging.debug("dir " + name + " has no json")

    json_f.close()

def json_resolve(path, name, id):
    #print (id + " " + name + " " + path)
    s = open(path).readline()
    ret = dr.json_tranform(s, id, name)
    if ret['sum'] > conf.file_radio:
        return json.dumps(ret)
    else:
        logging.debug(id + " " + name + " is too small")
        return None


def make_tmp_dir():
    tmp_dir = conf.output_dir + "/tmp"
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    return tmp_dir

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename = "tmp.log",
                    filemode = "w",
                    )

    main()

