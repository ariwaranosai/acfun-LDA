from acfunData import dataResolve as dr
import codecs
import conf
import os,sys
import json
import logging
import jieba

__author__ = 'nkssai'


def dir2onefile():
    print "merging file"
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
    return tmp_dir

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


def word_list(path = conf.output_dir + "/tmp/"):
    jieba.initialize()
    jieba.load_userdict("./user_dict.txt")
    print "cutting words"
    dict = {}
    f = open(path+"/all_json.txt", "r")
    new_f = codecs.open(path + "/new_json.txt", "w")
    i = 0
    for line in f:
        if (i %100) == 0:
            sys.stderr.write(str(i) + "\n")
        i += 1
        json_obj = json.loads(line)
        danmu = json_obj['ci']
        json_obj['ci'] = {}
        for k in danmu.keys():
            words_list = danmu[k]
            word = jieba.cut(words_list)
            word = list(word)
            json_obj['ci'][k] = word
            for w in word:
                if w in dict.keys():
                    dict[w] += 1
                else:
                    dict[w] = 1
        new_f.write(json.dumps(json_obj) + u"\n")
    f.close()
    new_f.close()

    out = codecs.open(path + "/words.txt", "wb", "utf-8")
    for k in dict.keys():
        out.write(k)
        out.write(" ")
        out.write(unicode(dict[k]))
        out.write("\n")

    out.close()


def word2id(path = conf.output_dir + "/tmp/dict.txt"):
    #load dict
    dict = {}
    print "load dict"
    with codecs.open(path, "r", "utf-8") as dict_f:
        i = 0
        for line in dict_f:
            word = line.split(" ")[0]
            dict[unicode(word)] = i
            i += 1

    print "word 2 id"
    out = open(conf.output_dir + "/tmp/id_json.txt", "w")
    with open(conf.output_dir + "/tmp/new_json.txt", "r") as new_json_f:
        for line in new_json_f:
            json_obj = json.loads(line.strip("\n"))
            for k in json_obj['ci'].keys():
                l = json_obj['ci'][k]
                json_obj['ci'][k] = []
                for w in l:
                    if w in dict.keys():
                        json_obj['ci'][k].append(dict[w])
            out.write(json.dumps(json_obj) + "\n")

    out.close()







def main():
    path = dir2onefile()
    word_list(path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Error"
        sys.exit(0)
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename = "tmp.log",
                    filemode = "w",
                    )


    if sys.argv[1] == '0':
        main()
    elif sys.argv[1] == '1':
        dir2onefile()
    elif sys.argv[1] == '2':
        word_list()
    elif sys.argv[1] == '3':
        word2id()
    else:
        print "Error"
