import sys
import os
import getopt


def usage():
    print 'python exp.py [option]...'
    print '  -m \t method'
    print '  -r \t result file = <method>_<dataset>.result'
    print '  -d \t dataset'
    print '  -t \t experiment times = 1'
    sys.exit()

def average(infile, fout):
    fin = open(infile, 'r')
    total = 0
    count_time = [0.0, 0.0, 0.0, 0.0, 0.0]
    for line in fin:
        total += 1
        line = line.rstrip().strip()
        times = line.rstrip().split(' ')
        idx = 0
        for t in times:
            count_time[idx] += float(t)
            idx += 1
    for ct in count_time:
        fout.write(str(ct/total) + ' ')
    fout.write('\n')
    fin.close()

def get_density_list(path):
    my_list = os.listdir(path)
    my_list.sort()
    return my_list

def get_size_list(path):
    my_list = os.listdir(path)
    num_list = []
    for sz in my_list:  
        num_list.append(int(sz[1:]))
    num_list.sort()
    sorted_list = []
    for num in num_list:
        sorted_list.append('N'+str(num))
    return sorted_list

DATA_ROOT = '../Dataset'
method = ''
res_file = ''
dataset = ''
rounds = 1

try:
    opts, args = getopt.getopt(sys.argv[1:], 'hm:r:d:t:')
except getopt.GetoptError:
    usage()

for opt, arg in opts:
    if opt == '-h':
        usage()
    elif opt == '-m':
        method = arg
    elif opt == '-r':
        res_file = arg
    elif opt == '-d':
        dataset = arg
    elif opt == '-t':
        rounds = int(arg)

if method == '' or dataset == '':
    usage()
if res_file == '':
    res_file = method + '_' + dataset + '.result'


if os.path.exists(method):
    print '%s running... exit' % (method)
    sys.exit()

if os.path.exists(res_file):
    print '%s exist, please remove it first' % (res_file)
    sys.exit()


# Experiments start
os.system('make %s' % (method))

fp = open(res_file, 'w')
density_list = get_density_list('%s/%s' % (DATA_ROOT, dataset))
size_list = get_size_list('%s/%s/%s' % (DATA_ROOT, dataset, density_list[0]))
for d in density_list:
    fp.write(d+'\n')
    for sz in size_list:
        infile = DATA_ROOT + '/' + dataset + '/' + d + '/' + sz
        tmpfile = 'tmp' + method
        cmd = './%s %s %s > /dev/null 2>> %s' % (method, infile, sz[1:], tmpfile)
        os.system('rm -f %s' % (tmpfile))
        for i in range(rounds):
            print '%d %s' % (i, cmd)
            os.system(cmd)
            os.system('echo "" >> %s' % (tmpfile))
        average(tmpfile, fp)
fp.close()

os.system('rm -f %s %s' % (method, tmpfile))


