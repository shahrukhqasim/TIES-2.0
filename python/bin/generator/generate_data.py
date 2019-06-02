import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from DataGeneration.TFGeneration.GenerateTFRecord import *
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--filesize',type=int,default=1)            #Number of images in one tfrecord
parser.add_argument('--threads',type=int,default=1)             #One thread will work on one tfrecord
parser.add_argument('--outpath',default='../../../tfrecords/')  #directory to store tfrecords
parser.add_argument('--tfcount',type=int,default=1)             #number of tfrecords to generate
parser.add_argument('--imagespath',default='../../../Table_Detection_Dataset/unlv/train/images')
parser.add_argument('--ocrpath',default='../../../Table_Detection_Dataset/unlv/unlv_xml_ocr')
parser.add_argument('--tablepath',default='../../../Table_Detection_Dataset/unlv/unlv _xml_gt')
parser.add_argument('--writetoimg',type=int,default=0)
args=parser.parse_args()

writetoimg=False
if(args.writetoimg==1):
    writetoimg=True
t = GenerateTFRecord(args.outpath,args.filesize,args.tfcount,args.imagespath,
                     args.ocrpath,args.tablepath,writetoimg)
t.write_to_tf(args.threads)
