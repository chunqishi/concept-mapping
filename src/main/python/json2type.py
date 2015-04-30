import sys
import json
import codecs


def escape_newline(s):
    return s.replace('\n', '\\n').replace('\r', '')

def tree2dict(node, treelist, fullpath):
    if type(node) is not dict:
        # print type(node)
        # print 'node is ', node
        treelist.append((fullpath, escape_newline(unicode(node).strip())))
        return treelist
    for k in node.items():
        # print type(k[0])
        v = node.get(k[0])
        subpath = fullpath + "." + k[0]
        if type(v) is dict:
            tree2dict(v, treelist, subpath)
        elif type(v) is list:
            for i in range(len(v)):
                tree2dict(v[i], treelist, subpath+"["+str(i)+"]")
        else:
            treelist.append((subpath, escape_newline(unicode(v).strip())))
            # print subpath, " = ", v.encode('utf-8')
    return treelist

if __name__ == "__main__":
    inputstream = codecs.open(sys.argv[1], encoding='utf-8')
    outputstream = codecs.open(sys.argv[2], mode='w', encoding='utf-8')

    jsonText = inputstream.read()
    data = json.loads(jsonText)
    # my_ordered_dict = json.loads(jsonText, object_pairs_hook=collections.OrderedDict)
    treelist = tree2dict(data, [] , "")
    treelist.sort()
    # print treelist
    for item in treelist:
        outputstream.write("%s\n" % "\t".join(item))
    # print treedict
    #
    # # print my_ordered_dict.keys()
    # for k,v in data.items():
    #    print k # shows name and properties
    #    print type(data[k])
