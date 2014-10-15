def topic_id2word(p):
    f = open(p + "/model.vPhi")
    wfile = open(p + "/model.vPhi.new", "w")
    words = load(p + "/dict.txt");
    for line in f:
        dict = {}
        i = 0
        for w in line.strip().split(" "):
            try:
                tmp = float(w)
            except:
                print w
            dict[words[i]] = tmp
            i += 1

        order_list = sorted(dict.iteritems(), key = lambda a:a[1], reverse = True)

        s = ""
        for key in order_list:
            s = s + key[0] + " " + str(key[1]) + " "

        wfile.write(s + "\n")



def load(path):
    dict = {}
    f = open(path)
    i = 0
    for line in f:
        l = line.split()
        dict[i] = l[0]
        i+=1
    return dict


topic_id2word(".")
