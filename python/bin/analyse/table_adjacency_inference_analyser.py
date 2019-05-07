import argparse
import os
import sys
from collections import Counter
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__),'../../')))

def str2bool(v):
    if type(v) == bool:
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Plot clustering model output')
parser.add_argument('input', help="Path to the config file which was used to train")
parser.add_argument('config', help="Config section within the config file")
parser.add_argument('--figures', help="Whether to show 3d plots", default=False)
args = parser.parse_args()

if __name__ != "__main__":
    print("Can't import this file")
    exit(0)

show_3d_figures = str2bool(args.figures)

import matplotlib as mpl
if not show_3d_figures:
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys
import math
import gzip
import pickle
import configparser as cp
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.backends.backend_pdf
from libs.configuration_manager import ConfigurationManager as gconfig
import libs.plots as plots
import subprocess




def analyse(args):
    gconfig.init(args.input, args.config)
    out_path = gconfig.get_config_param('test_out_path', 'str')
    out_path = os.path.join(out_path, 'visualizations')
    subprocess.call("mkdir -p %s" % (out_path), shell=True)

    store = []
    inference_files = os.path.join(gconfig.get_config_param('test_out_path', 'str'), 'inference_output_files.txt')

    sample_number = 0
    Gt_in_Pred_counter=[{'cells':0,'rows':0,'cols':0},{'cells':0,'rows':0,'cols':0},{'cells':0,'rows':0,'cols':0},{'cells':0,'rows':0,'cols':0}]
    Pred_not_in_Gt_counter=[{'cells':0,'rows':0,'cols':0},{'cells':0,'rows':0,'cols':0},{'cells':0,'rows':0,'cols':0},{'cells':0,'rows':0,'cols':0}]
    M_values=[0,0,0,0]
    M_list=[]
    totalimages=[0,0,0,0]

    alagsay=0
    clique_lengths=[dict(),dict(),dict()]
    arrnames = ['cells', 'rows', 'cols']

    exitflag=False
    with open(inference_files) as f:
        content = f.readlines()
        cells=0
        rows=0
        cols=0
        imgcount=0
        graphsizesum=0
        sumofsq=0
        sumofx = 0
        sumofsq_cliquetime=[0,0,0]
        sumofx_cliquetime=[0,0,0]
        maxtimes=[0,0,0]

        for i in content:
            with gzip.open(i.strip()) as f:
                data = pickle.load(f)
                for j in data:
                    result = j
                    image = result['image']
                    ground_truths = result['sampled_ground_truths']
                    predictions = result['sampled_predictions']
                    indices = result['sampled_indices']
                    global_features = result['global_features']
                    vertex_features = result['vertex_features']
                    graphsizesum+=global_features[2]
                    imgcount+=1
                    sumofsq+=(global_features[2]**2)
                    sumofx+=global_features[2]
                    incrementalvariance=(sumofsq/imgcount)-((sumofx/imgcount)**2)
                    print('Graph average: ',graphsizesum/imgcount,', std: ',np.sqrt(incrementalvariance))

                    difficultylevel=int(global_features[3])-1

                    #print(indices[0].shape)
                    #print(indices[0])


                    gtpred,prednotgt,m,clengths,ctime=plots.plot_few(out_path, sample_number, result)
                    for p in range(3):
                        if(ctime[p]>maxtimes[p]):
                            maxtimes[p]=ctime[p]
                        sumofx_cliquetime[p]+=ctime[p]
                        sumofsq_cliquetime[p]+=(ctime[p]**2)
                        temptime=np.sqrt((sumofsq_cliquetime[p]/imgcount)-((sumofx_cliquetime[p]/imgcount)**2))
                        print('\n',arrnames[p],' Clique time average:',sumofx_cliquetime[p]/imgcount,', std:',temptime)
                        print('Max ',arrnames[p],' time:',maxtimes[p])
                    # for i,subdict in enumerate(clengths):
                    #     clique_lengths[i]=Counter(clique_lengths[i])+Counter(subdict)

                    print('M:',m)
                    M_list.append(m)

                    Gt_in_Pred_counter[difficultylevel]['cells'] += gtpred['cells']
                    Gt_in_Pred_counter[difficultylevel]['rows']+=gtpred['rows']
                    Gt_in_Pred_counter[difficultylevel]['cols'] += gtpred['cols']

                    Pred_not_in_Gt_counter[difficultylevel]['cells'] += prednotgt['cells']
                    Pred_not_in_Gt_counter[difficultylevel]['rows']+=prednotgt['rows']
                    Pred_not_in_Gt_counter[difficultylevel]['cols'] += prednotgt['cols']

                    if(m==1.0):
                      M_values[difficultylevel]+=1


                    sample_number += 1
                    totalimages[difficultylevel]+=1

                    # if(sample_number%200==0):
                    #     print('Cliques output: ')
                    #     for i in range(3):
                    #         sum=0
                    #         countfreq=0
                    #         subarr=clique_lengths[i]
                    #         stdsum=0
                    #         for key in subarr.keys():
                    #             sum+=(key*subarr[key])
                    #             countfreq += subarr[key]
                    #
                    #         mean = sum / countfreq
                    #
                    #         for key in subarr.keys():
                    #             stdsum += (subarr[key]) * np.square(abs(mean - key))
                    #
                    #         stdval=np.sqrt(stdsum/countfreq)
                    #         print(arrnames[i], ' average : ', mean,', std: ',stdval)
                    #     if(sample_number>=2000):
                    #         exitflag=True
                    #         break

            if(exitflag):
                break



        #print('cells:',cells/sample_number,'row:',rows/sample_number,' column:',cols/sample_number)
        for counters,name in zip([Gt_in_Pred_counter,Pred_not_in_Gt_counter],['GT in Pred','Pred not in GT']):
            print(name,': ')
            for i in range(4):
                print('Diffculty level: ',i+1)
                subarr=counters[i]
                totalimagesoflevel=totalimages[i]
                print('Total images:',totalimagesoflevel)
                for name in arrnames:
                    print(name,': ',subarr[name]/totalimagesoflevel)

        for i in range(4):
            print('Difficulty level:',i+1,' M value:',M_values[i]/totalimages[i])
        #print('New metric:',alagsay/totalimagesoflevel)


        # f=open('Mpoints','wb')
        # pickle.dump(M_list,f)
        # f.close()




    # TODO: Hassan
    # Iterate and calculate accuracy etc
    for i in store:
        pass


if __name__ != '__main__':
    print("Can't import  this as a module")

parser = argparse.ArgumentParser(description='Run analysis')
parser.add_argument('input', help="Path to config file")
parser.add_argument('config', help="Config section within the config file")
args = parser.parse_args()

analyse(args)
