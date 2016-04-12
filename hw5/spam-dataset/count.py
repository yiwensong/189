import glob
import re

spams = glob.iglob('spam/*')
hams  = glob.iglob('ham/*')
OUTPUT = 'featurize2.py'

def dict_compare(k1,k2):
    if abs(d[k1]) < abs(d[k2]):
        return 1
    return -1

def sort_dict_by_val(dictionary):
    sorted_keys = dictionary.keys()
    sorted_keys.sort(cmp=dict_compare)
    return sorted_keys

d = dict()

def get_dict(files,cnt_weight):
    for fname in files:
        with open(fname,'r') as f:
            buf = f.read()
            words = filter(lambda a: a not in ['','a','the','to','and','is'],re.split('\W+',buf))
            puncts = filter(lambda a: a not in [' ','\r','\n',',','.','\'','\x01',''] , re.findall('\W',buf))
            for i in (words + puncts):
                if i in d:
                    d[i] += cnt_weight
                else:
                    d[i] = cnt_weight

    keys = sort_dict_by_val(d)
    values = [d[k] for k in keys]
    return keys,values


FCN_START = 'def F'

def generate_freq(n,text):
    FCN_START = 'def F'
    NUM = str(n)
    FCN_MID = '(text,freq):\n    return float(freq[\''
    TEXT = text
    FCN_END = '\'])\n'
    return FCN_START + NUM + FCN_MID + TEXT + FCN_END

def generate_count(n,text):
    FCN_START = 'def F'
    NUM = str(n)
    FCN_MID = '(text,freq):\n    return text.count(\''
    TEXT = text
    FCN_END = '\')\n'
    return FCN_START + NUM + FCN_MID + TEXT + FCN_END

def add_feature(n):
    return '    feature.append(F' + str(n) + '(text, freq))\n'

def add_feature_function(n):
    FCN_START = 'def generate_feature_vector(text,freq):\n    feature = []\n'
    FCN_END = '    return feature'
    FCN_MID = ''
    for i in xrange(n):
        FCN_MID += add_feature(i)
    return FCN_START + FCN_MID + FCN_END


FEATURES_TO_USE = 200

def codegen():
    global keys,values
    _,__ = get_dict(spams,-1)
    keys,values = get_dict(hams,1)
    
    with open(OUTPUT,'w') as f_out:
        with open('HEAD','r') as f_in:
            f_out.write(f_in.read())
        for i in xrange(FEATURES_TO_USE):
            feature = keys[i]
            if re.match('\W+',feature) is not None:
                f_out.write(generate_freq(i,feature))
            else:
                f_out.write(generate_count(i,feature))
        f_out.write(add_feature_function(FEATURES_TO_USE))
        with open('TAIL','r') as f_in:
            f_out.write(f_in.read())
    return 




codegen()