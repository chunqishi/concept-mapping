import sys, os, traceback
import json
import codecs
import collections
import re
import hashlib
from sklearn import svm
# from sklearn.metrics import jaccard_similarity_score
import cPickle as pickle
import time

import logging
logging.basicConfig(filename='logsim.txt',level=logging.DEBUG)
current_milli_time = lambda: int(round(time.time()))
current_time_tag = lambda: time.strftime("%Y%m%d_%H%M%S")

def uniq_arr(arr=[]):
    """Function: uniq_arr
    ----------------------
    Brief: filter duplicated from the array
    Args:
        arr: The arr

    Returns:
        The de-duplicated array
    """
    mp = {}
    for a in arr:
        mp[a] = 0
    return mp.keys()

def uniq_arrs(arr1=[], arr2=[]):
    """Function: uniq_arrs
    ----------------------
    Brief: filter duplicated from two arrays
    Args:
        arr1: One array
        arr2: The other array
    Returns:
        The de-duplicated arrays
    """
    mp = {}
    l = min(len(arr1), len(arr2))
    new_arr1 = []
    new_arr2 = []
    for i in range(l):
        s = arr1[i]
        cnt = mp.get(s, 0)
        if cnt == 0:
            new_arr1.append(arr1[i])
            new_arr2.append(arr2[i])
        mp[s] = cnt + 1
    return new_arr1, new_arr2

def str2long(s):
    """Function: str2long
    ----------------------
    Brief: give string an integer id
    Args:
        s: The string

    Returns:
        The integer id
    """
    # hash_object = hashlib.md5(s.encode('ascii', 'ignore'))
    # l = long(hash_object.hexdigest(), 16) % (10 ** 8)
    l = abs(hash(s)) % (10 ** 8)
    # print l
    return l


def first_whitespace_index(s):
    for idx, c in enumerate(s):
        if c.isspace():
            return idx
    return len(s)


def str_remove_between(s, bgn, end):
    line = re.sub(r"" + re.escape(bgn) + "[^"+re.escape(bgn)+re.escape(end)+"]+" + re.escape(end), "", s)
    return line


def line_process(line):
    """Function: line_process
    ----------------------
    Brief: parse data and category
    Args:
        line: The input line
    Returns:
        data: The data array
        target: The category array
    """
    line = line.strip()
    idx = first_whitespace_index(line)
    target = line[:idx]
    data = line[idx:]
    target = str_remove_between(target, "[","]")
    target = str_remove_between(target, "{","}")
    data = data.strip()
    target = target.strip()
    return data, target


def file_process(f, data=[], target=[], umap={}):
    """Function: file_process
    ----------------------
    Brief: parse the data and category
    Args:
        f: The data file
        data: The data array
        target: The category array
        umap: The unique mapping
    Returns:
        data: The data array
        target: The category array
    """
    inputstream = codecs.open(f, encoding='utf-8')
    lines = inputstream.readlines()
    for line in lines:
        if not line.isspace():
            line = line.strip()
            id = str2long(line)
            cnt = umap.get(id, 0)
            if cnt == 0:
                linedata, linetarget = line_process(line)
                data.append(linedata)
                target.append(linetarget)
            else:
                umap[id] = cnt + 1
    return data, target


def directory_process(d, data=[], target=[], numlim=50):
    """Function: directory_process
    ----------------------
    Brief: parse files in the directory
    Args:
        d: The directory
        data: The data array
        target: The category array
    Returns:
        data: The data array
        target: The category array
        svmdata: The data integer array
        svmtarget: The category integer array
    """
    begin_time = current_milli_time()
    from os import listdir
    from os.path import isfile, join
    files = [ f for f in listdir(d) if isfile(join(d,f)) ]
    logging.info("read files from " + d + ":" + str(files))
    umap = {}
    if numlim > 0:
        files = files[:numlim]
    for f in files:
        logging.info("now processing file: " + f + " ...")
        file_process(join(d,f), data, target, umap)
    logging.info("read data (%d) " % len(data))
    # now we have the data and target.
    # we need to deduplicate and then generate id for each string.
    udata, utarget = uniq_arrs(data, target)
    logging.info("unique data length (%d) " % len(udata))
    logging.info("unique data (%s) " % str(udata))
    logging.info("unique target (%s) " % str(utarget))
    svmdata = []
    svmtarget = []
    for i in range(len(udata)):
        svmdata.append([str2long(udata[i])])
        svmtarget.append(str2long(utarget[i]))
    logging.info("directory_process time used: %d s" % (current_milli_time() - begin_time))
    return udata, utarget, svmdata, svmtarget

def rf_train(svmdata, svmtarget):
    from sklearn.ensemble import ExtraTreesClassifier
    begin_time = current_milli_time()
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
    logging.info("training RF Model: %s ..." % str(clf))
    logging.info("data length:%d" % len(svmdata))
    logging.info("target length:%d" % len(svmtarget))
    logging.info("data:" + str(svmdata))
    logging.info("target:" + str(svmtarget))
    clf.fit(svmdata, svmtarget)
    logging.info("train time used: %d s" % (current_milli_time() - begin_time))
    return clf


def svm_train(svmdata, svmtarget):
    """Function: train
    ----------------------
    Brief: train the classifier
    Args:
        d: The directory
        data: The data array
        target: The category array
    Returns:
        data: The data array
        target: The category array
        svmdata: The data integer array
        svmtarget: The category integer array
    """
    begin_time = current_milli_time()
    # http://scikit-learn.org/stable/modules/svm.html
    clf = svm.SVC()
    logging.info("training SVM Model: %s ..." % str(clf))
    logging.info("data length:%d" % len(svmdata))
    logging.info("target length:%d" % len(svmtarget))
    logging.info("data:" + str(svmdata))
    logging.info("target:" + str(svmtarget))
    clf.fit(svmdata, svmtarget)
    logging.info("train time used: %d s" % (current_milli_time() - begin_time))
    return clf


def dir_training(dir_train, dir_test, train=rf_train):
    logging.info("train data from \"%s\", Test data from \"%s\" ." % (dir_train, dir_test))
    train_data, train_target, train_svmdata, train_svmtarget = directory_process(dir_train, [], [])
    omtrained = train(train_svmdata, train_svmtarget)
    train_target_map = {}
    for i in range(len(train_svmtarget)):
        train_target_map[train_svmtarget[i]] = train_target[i]
    test_data, test_target, test_svmdata, test_svmtarget = directory_process(dir_test, [], [])
    logging.info("begin prediction ...")
    begin_time = current_milli_time()
    predict = omtrained.predict(test_svmdata)
    logging.info("predict time used: %d s" % (current_milli_time() - begin_time))
    logging.info("prediction = %s..." % str(predict))
    uniq_map = {}
    for i in range(len(predict)):
        key = test_target[i] + "--->" + train_target_map.get(predict[i])
        cnt = uniq_map.get(key, 0)
        uniq_map[key] = cnt + 1
    logging.info("prediction (unique) = %s..." % str(uniq_map))
    for i in uniq_map.keys():
        print i.encode('utf-8'),  uniq_map.get(i)
    return uniq_map

    
def dir_sim(dir_train, dir_test):
    logging.info("One data from \"%s\", the other data from \"%s\" ." % (dir_train, dir_test))
    one_data, one_target, one_svmdata, one_svmtarget = directory_process(dir_train, [], [])
    other_data, other_target, other_svmdata, other_svmtarget = directory_process(dir_test, [], [])
    onelen = len(one_data)
    otherlen = len(other_data)
    one_map = {}
    other_map = {}    
    for i in range(onelen):
        one_set = one_map.get(one_target[i], [])    
        one_set.append(one_svmdata[i][0])
        one_map[one_target[i]] = one_set
    for i in range(otherlen):
        other_set = other_map.get(other_target[i], [])    
        other_set.append(other_svmdata[i][0])
        other_map[other_target[i]] = other_set    
    logging.info("Two set map length: %s and %s"%(str(len(one_map)), str(len(other_map))))
    simmap = {}
    for one in one_map.keys():
        one_val = one_map.get(one)
        for other in other_map.keys():
                other_val = other_map.get(other)    
                jaccsim = jaccard_similarity(one_val, other_val)
                if jaccsim > 0:
                    simmap[one +'----' + other] =  jaccsim
                    print one +'----' + other, simmap[one +'----' + other]
    return simmap
    
def inc_map(map, key, val=0):
    """Function: inc_map
    ----------------------
    Brief: increase map count
    Args:
        map: The count map
        key: The key
        val: The key count
    Returns:
        map: The updated count map
    """
    cnt = map.get(key, 0)
    map[key] = cnt + val
    return map

    
def cal_prob(uniq_map_forward={}, uniq_map_backward={}):
    forward_to_map = {}
    forward_from_map = {}
    backward_to_map = {}
    backward_from_map = {}
    prob_map = {}
    for key in uniq_map_forward.keys():
        val = uniq_map_forward.get(key, 0)
        pair = key.split("--->") 
        from_ = pair[0]
        to_ = pair[1]
        cnt = forward_from_map.get(from_, 0)
        cnt += val
        forward_from_map[from_] = cnt
        cnt = forward_to_map.get(to_, 0)
        cnt += val  
        forward_to_map[to_] = cnt
        
    for key in uniq_map_backward.keys():
        val = uniq_map_backward.get(key, 0)
        pair = key.split("--->") 
        from_ = pair[0]
        to_ = pair[1]
        cnt = backward_from_map.get(from_, 0)
        cnt += val
        backward_from_map[from_] = cnt
        cnt = backward_to_map.get(to_, 0)
        cnt += val  
        backward_to_map[to_] = cnt         

    print 'uniq_map_forward', uniq_map_forward
    print 'forward_from_map', forward_from_map
    print 'forward_to_map', forward_to_map
    print 'uniq_map_backward', uniq_map_backward
    print 'backward_from_map', backward_from_map
    print 'backward_to_map', backward_to_map

    for key in uniq_map_forward.keys():
        val = uniq_map_forward.get(key, 0)
        pair = key.split("--->") 
        from_ = pair[0]
        to_ = pair[1]        
        prob_map[from_+"----"+to_] = float(uniq_map_forward.get(from_+"--->"+to_,0)) / float(forward_from_map.get(from_,0) + forward_to_map.get(to_,0) - uniq_map_forward.get(from_+"--->"+to_,0)) * float(uniq_map_backward.get(to_+"--->"+from_,0)) / float(backward_from_map.get(to_,0) + backward_to_map.get(from_,0) - uniq_map_backward.get(to_+"--->"+from_,0))
        print key, prob_map[from_+"----"+to_]
            
    return prob_map
    
def cal_rank(uniq_map_forward={}, uniq_map_backward={}):
    joinprob = cal_prob(uniq_map_forward, uniq_map_backward)
    uniq_pair_map = {}
    uniq_single_map = {}
    prop_uniq_pair_map = {}

    logging.info("-------------------------------------------------------")
    logging.info("------------- Rank Start .......-----------------------")
    logging.info("-------------------------------------------------------")
    for key in uniq_map_forward.keys():
        val = uniq_map_forward.get(key, 0)
        pair = key.split("--->")
        ##  http://stackoverflow.com/questions/1097908/how-do-i-sort-unicode-strings-alphabetically-in-python
        sorted_pair = sorted(pair)
        newkey = "<-->".join(sorted_pair);
        cnt = uniq_pair_map.get(newkey, 0)
        cnt = cnt + val
        uniq_pair_map[newkey] = cnt

        cnt = uniq_single_map.get(pair[0], 0)
        cnt = cnt + val
        uniq_single_map[pair[0]] = cnt
        cnt = uniq_single_map.get(pair[1], 0)
        cnt = cnt + val
        uniq_single_map[pair[1]] = cnt

    for key in uniq_map_backward.keys():
        val = uniq_map_backward.get(key, 0)
        pair = key.split("--->")
        sorted_pair = sorted(pair)
        newkey = "<-->".join(sorted_pair);
        cnt = uniq_pair_map.get(newkey, 0)
        cnt = cnt + val
        uniq_pair_map[newkey] = cnt

        cnt = uniq_single_map.get(pair[0], 0)
        cnt = cnt + val
        uniq_single_map[pair[0]] = cnt
        cnt = uniq_single_map.get(pair[1], 0)
        cnt = cnt + val
        uniq_single_map[pair[1]] = cnt

    for key in uniq_pair_map.keys():
        pair = key.split("<-->")
        cnt = uniq_pair_map.get(key, 0)
        cnt = float(cnt) / (uniq_single_map.get(pair[0], 0) + uniq_single_map.get(pair[1], 0))
        prop_uniq_pair_map[key] = cnt

    import cStringIO
    output = cStringIO.StringIO()
    print >>output, "concept1, concept2, frequency, probability, joinprobability"
    for key in uniq_pair_map.keys():
         pair = key.split("<-->")
         print >>output, "%s,%s,%s,%s,%s" % (pair[0], pair[1],
                                          str(uniq_pair_map.get(key)),str(prop_uniq_pair_map.get(key)), str(joinprob.get(pair[1]+"----"+pair[0],0)))
    contents = output.getvalue()
    output.close()
    outputstream = codecs.open("output_" + current_time_tag() + ".csv", mode='w', encoding='utf-8')
    outputstream.write(contents.decode('utf-8'))
    outputstream.close()
    logging.info("-------------------------------------------------------")
    logging.info("-------------  ....... Rank End -----------------------")
    logging.info("-------------------------------------------------------")


def gdf_prop_gen():
    """gdf_prop_gen: Generate the GDF format for  GUESS/Gephi data format.


    :return:
    """


def gdf_gen(uniq_map_forward={}, uniq_map_backward={}):
    """gdf_gen: Generate the GDF format for GUESS/Gephi data format.
        nodedef> name, label, category, style, color
        edgedef> name1, name2, color, weight, label, directed
    """
    node_from = {}
    for key in uniq_map_forward.keys():
        fr = key.split("--->")[0]
        node_from[fr] = 1

    for key in uniq_map_backward.keys():
        fr = key.split("--->")[0]
        node_from[fr] = 2

    logging.info("-------------------------------------------------------")
    logging.info("------------- GDF Start ........-----------------------")
    logging.info("-------------------------------------------------------")
    import cStringIO
    output = cStringIO.StringIO()
    # print "nodedef> name VARCHAR, label VARCHAR, category VARCHAR, style VARCHAR, color VARCHAR"
    print >>output, "nodedef> name VARCHAR, label VARCHAR, category VARCHAR, style VARCHAR"
    for i in node_from.keys():
        print >>output, i.encode('utf-8'), ",",i.encode('utf-8') ,",",node_from.get(i) ,",",node_from.get(i)

    print >>output, "edgedef> name1, name2, directed Boolean, weight INT, label VARCHAR, color VARCHAR"
    for key in uniq_map_forward.keys():
        pair = key.split("--->")
        print >>output, pair[0].encode('utf-8'), ",",pair[1].encode('utf-8') ,", true ,",uniq_map_forward.get(key) ,",", key.encode('utf-8'), 'blue'

    for key in uniq_map_backward.keys():
        pair = key.split("--->")
        print >>output, pair[0].encode('utf-8'), ",",pair[1].encode('utf-8') ,", true ,",uniq_map_backward.get(key) ,",", key.encode('utf-8'), 'brown'
    contents = output.getvalue()
    output.close()
    outputstream = codecs.open("output_" + current_time_tag() + ".gdf", mode='w', encoding='utf-8')
    outputstream.write(contents.decode('utf-8'))
    outputstream.close()
    logging.info("-------------------------------------------------------")
    logging.info("-------------  ........ GDF End -----------------------")
    logging.info("-------------------------------------------------------")


def dump(data, path, id):
    if not os.path.exists(path):
        os.makedirs(path)
    file = os.path.join(path, id)
    output = open(file, 'wb')
    pickle.dump(data, output)
    output.close()

def load(path, id):
    file = os.path.join(path, id)
    pkl_file = open(file, 'rb')
    data = pickle.load(pkl_file)
    return data
    
def jaccard_similarity(arr1, arr2):
    print arr1
    print arr2
    a = set(arr1)
    b = set(arr2)
    inter = set.intersection(*[a,b])
    union = set.union(*[a,b])
    similarity = float(len(inter)*1.0/len(union)) #similarity belongs to [0,1] 1 means its exact replica.
    return similarity

if __name__ == '__main__':
    # unittest.main()
    print "Start..."
    dir_train = sys.argv[1]
    dir_test = sys.argv[2]

    
    logging.info("")
    logging.info('=====================================================================================')
    logging.info("")
    
    simmap = dir_sim(dir_train, dir_test)

    import cStringIO
    output = cStringIO.StringIO()
    print >>output, "concept1, concept2, jaccard_similarity"
    for key in simmap.keys():
         pair = key.split("----")
         print >>output, "%s,%s,%s" % (pair[0], pair[1],
                                          str(simmap.get(key)))
    contents = output.getvalue()
    output.close()
    outputstream = codecs.open("sim_" + current_time_tag() + ".csv", mode='w', encoding='utf-8')
    outputstream.write(contents.decode('utf-8'))
    outputstream.close()






