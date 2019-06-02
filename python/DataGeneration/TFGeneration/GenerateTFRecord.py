import tensorflow as tf
import numpy as np
import traceback
import cv2
import os
import string
import pickle
from multiprocessing import Process,Lock
from DataGeneration.TableGeneration.Table import Table
from multiprocessing import Process,Pool,cpu_count
import random
import argparse
from DataGeneration.TableGeneration.tools import *
import numpy as np
from selenium.webdriver import Firefox
from selenium.webdriver import PhantomJS
import warnings
from DataGeneration.TableGeneration.Transformation import *

def warn(*args,**kwargs):
    pass

class Logger:
    def __init__(self):
        pass
        #self.file=open('logtxt.txt','a+')

    def write(self,txt):
        file = open('logtxt.txt', 'a+')
        file.write(txt)
        file.close()

class GenerateTFRecord:
    def __init__(self, outpath,filesize,tfcount,unlvimagespath,unlvocrpath,unlvtablepath,writetoimg):
        self.outtfpath = outpath
        self.filesize=filesize
        self.num_of_tfs=tfcount
        self.unlvocrpath=unlvocrpath
        self.unlvimagespath=unlvimagespath
        self.unlvtablepath=unlvtablepath
        self.create_dir(self.outtfpath)
        self.pool=Pool(processes=cpu_count())
        self.writetoimg=writetoimg
        self.logger=Logger()
        #self.logdir = 'logdir/'
        #self.create_dir(self.logdir)
        #logging.basicConfig(filename=os.path.join(self.logdir,'Log.log'), filemode='a+', format='%(name)s - %(levelname)s - %(message)s')
        self.writer=None
        self.lock=Lock()
        self.num_of_max_vertices=900
        self.max_length_of_word=30
        self.row_min=3
        self.row_max=15
        self.col_min=3
        self.col_max=9
        self.minshearval=-0.1
        self.maxshearval=0.1
        self.minrotval=-0.01
        self.maxrotval=0.01
        self.num_data_dims=5
        self.filecounter=1
        self.max_height=768
        self.max_width=1366
        self.threadscounter=0

        #self.str_to_chars=lambda str:np.chararray(list(str))

    def create_dir(self,fpath):
        if(not os.path.exists(fpath)):
            os.mkdir(fpath)

    def str_to_int(self,str):
        intsarr=np.array([ord(chr) for chr in str])
        padded_arr=np.zeros(shape=(self.max_length_of_word),dtype=np.int64)
        padded_arr[:len(intsarr)]=intsarr
        return padded_arr

    def convert_to_int(self, arr):
        return [int(val) for val in arr]

    def pad_with_zeros(self,arr,shape):
        dummy=np.zeros(shape,dtype=np.int64)
        dummy[:arr.shape[0],:arr.shape[1]]=arr
        return dummy

    def generate_tf_record(self, im, cellmatrix, rowmatrix, colmatrix, arr,difficultylevel):


        cellmatrix=self.pad_with_zeros(cellmatrix,(self.num_of_max_vertices,self.num_of_max_vertices))
        colmatrix = self.pad_with_zeros(colmatrix, (self.num_of_max_vertices, self.num_of_max_vertices))
        rowmatrix = self.pad_with_zeros(rowmatrix, (self.num_of_max_vertices, self.num_of_max_vertices))

        #im = np.array(cv2.imread(img_path, 0),dtype=np.int64)
        im=im.astype(np.int64)
        img_height, img_width=im.shape

        words_arr = arr[:, 1].tolist()
        no_of_words = len(words_arr)


        lengths_arr = self.convert_to_int(arr[:, 0])
        vertex_features=np.zeros(shape=(self.num_of_max_vertices,self.num_data_dims),dtype=np.int64)
        lengths_arr=np.array(lengths_arr).reshape(len(lengths_arr),-1)
        sample_out=np.array(np.concatenate((arr[:,2:],lengths_arr),axis=1))
        vertex_features[:no_of_words,:]=sample_out


        #vertex_text=np.chararray(shape=(self.num_of_max_vertices,self.max_length_of_word))
        #vertex_text[:no_of_words,:]=list(map(self.str_to_chars, words_arr))
        #vertex_text=words_arr+[""]*(self.num_of_max_vertices-len(words_arr))

        vertex_text = np.zeros((self.num_of_max_vertices,self.max_length_of_word), dtype=np.int64)
        vertex_text[:no_of_words]=np.array(list(map(self.str_to_int,words_arr)))


        feature = dict()
        feature['image'] = tf.train.Feature(float_list=tf.train.FloatList(value=im.astype(np.float32).flatten()))
        feature['global_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=np.array([img_height, img_width,no_of_words,difficultylevel]).astype(np.float32).flatten()))
        feature['vertex_features'] = tf.train.Feature(float_list=tf.train.FloatList(value=vertex_features.astype(np.float32).flatten()))
        feature['adjacency_matrix_cells'] = tf.train.Feature(int64_list=tf.train.Int64List(value=cellmatrix.astype(np.int64).flatten()))
        feature['adjacency_matrix_cols'] = tf.train.Feature(int64_list=tf.train.Int64List(value=colmatrix.astype(np.int64).flatten()))
        feature['adjacency_matrix_rows'] = tf.train.Feature(int64_list=tf.train.Int64List(value=rowmatrix.astype(np.int64).flatten()))
        feature['vertex_text'] = tf.train.Feature(int64_list=tf.train.Int64List(value=vertex_text.astype(np.int64).flatten()))

        all_features = tf.train.Features(feature=feature)


        seq_ex = tf.train.Example(features=all_features)
        return seq_ex

    def generate_tables(self,driver,N_imgs,output_file_name):
        #np.random.seed(time.time())
        #arr = np.random.randint(1, self.max_rows, (N_imgs, 2))
        row_col_min=[self.row_min,self.col_min]
        row_col_max=[self.row_max,self.col_max]
        arr = np.random.uniform(low=row_col_min, high=row_col_max, size=(N_imgs, 2))
        all_difficulty_levels=[0,0,0,0]
        arr[:,0]=arr[:,0]+2
        data_arr=[]
        exceptioncount=0
        for i, subarr in enumerate(arr):

            rows = int(round(subarr[0]))
            cols = int(round(subarr[1]))

            exceptcount=0
            while(True):
                try:
                    table = Table(rows,cols,self.unlvimagespath,self.unlvocrpath,self.unlvtablepath)

                    same_cell_matrix,same_col_matrix,same_row_matrix, id_count, html_content,difficultylevel= table.create()
                    #print('table creation time:',time.time()-start1)


                    im,bboxes = html_to_img(driver, html_content, id_count, self.max_height, self.max_width)
                    # apply_shear: bool - True: Apply Transformation, False: No Transformation

                    #probability weight for shearing to be 25%
                    apply_shear = random.choices([True, False],weights=[0.25,0.75])[0]

                    if(apply_shear==True):
                        shearval = np.random.uniform(self.minshearval, self.maxshearval)
                        rotval = np.random.uniform(self.minrotval, self.maxrotval)
                        #print('\nApplying shear:',' shearval:',shearval,' rotval:',rotval)
                        im, bboxes = Transform(im, bboxes, shearval, rotval, self.max_width, self.max_height)
                        if(shearval!=0.0 and rotval!=0.0):
                            difficultylevel=4

                    if(self.writetoimg):
                        dirname='level'+str(difficultylevel)
                        self.create_dir(dirname)
                        self.create_dir(os.path.join(dirname,'html'))
                        self.create_dir(os.path.join(dirname, 'img'))
                        f=open(os.path.join(dirname,'html',str(i)+output_file_name.replace('.tfrecord','.html')),'w')
                        f.write(html_content)
                        f.close()
                        im.save(os.path.join(dirname,'img',str(i)+output_file_name.replace('.tfrecord','.png')), dpi=(600, 600))

                    #print('difficultylevel:',difficultylevel)

                    #print('html to img time:',time.time()-start2)
                    data_arr.append([[same_row_matrix, same_col_matrix, same_cell_matrix, bboxes,[difficultylevel]],[im]])
                    all_difficulty_levels[difficultylevel-1]+=1
                    break
                    #pickle.dump([same_row_matrix, same_col_matrix, same_cell_matrix, bboxes], infofile)
                except Exception as e:
                    traceback.print_exc()
                    exceptcount+=1
                    if(exceptioncount>10):
                        return None
                    #traceback.print_exc()
                    #print('\nException No.', exceptioncount, ' File: ', str(output_file_name))
                    #logging.error("Exception Occured "+str(output_file_name),exc_info=True)
        if(len(data_arr)!=N_imgs):
            print('Images not equal to the required size.')
            return None
        return data_arr,all_difficulty_levels

    def draw_col_matrix(self,im,arr,matrix):
        no_of_words=len(arr)
        colors = np.random.randint(0, 255, (no_of_words, 3))
        arr = arr[:, 2:]

        print('arr shape:',arr.shape)
        print('matrix shape:',matrix.shape)
        print('colors shape:', colors.shape)
        print('image shape:',im.shape)
        print('matrix\n',matrix)



        im=im.astype(np.uint8)
        im=np.dstack((im,im,im))
        #im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        # for x in range(no_of_words):
        #     indices = np.argwhere(matrix[x] == 1)
        #     for index in indices:
        #         cv2.rectangle(im, (int(arr[index, 0]), int(arr[index, 1])),
        #                       (int(arr[index, 2]), int(arr[index, 3])),
        #                       (0,0,255), 1)
        x=1
        indices = np.argwhere(matrix[x] == 1)
        for index in indices:
            cv2.rectangle(im, (int(arr[index, 0])-3, int(arr[index, 1])-3),
                          (int(arr[index, 2])+3, int(arr[index, 3])+3),
                          (0,255,0), 1)

        x = 4
        indices = np.argwhere(matrix[x] == 1)
        for index in indices:
            cv2.rectangle(im, (int(arr[index, 0]) - 3, int(arr[index, 1]) - 3),
                          (int(arr[index, 2]) + 3, int(arr[index, 3]) + 3),
                          (0, 0, 255), 1)

        #im=cv2.equalizeHist(im)
        cv2.imwrite('hassan22.jpg',im)


    def write_tf(self,filesize,threadnum):

        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        opts = Options()
        opts.set_headless()
        assert opts.headless
        #driver=PhantomJS()
        driver = Firefox(options=opts)
        while(True):
            starttime = time.time()

            output_file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20)) + '.tfrecord'
            print('Thread: ',threadnum,' Started:', output_file_name)

            data_arr,all_difficulty_levels = self.generate_tables(driver, filesize, output_file_name)
            
            #print('\nThread: ',threadnum,'data arr returned with :',len(data_arr))
            if(data_arr is not None):
                if(len(data_arr)==filesize):
                    with tf.python_io.TFRecordWriter(os.path.join(self.outtfpath,output_file_name),options=options) as writer:
                        try:
                            for subarr in data_arr:
                                arr=subarr[0]

                                img=np.asarray(subarr[1][0],np.int64)[:,:,0]
                                colmatrix = np.array(arr[1],dtype=np.int64)
                                cellmatrix = np.array(arr[2],dtype=np.int64)
                                rowmatrix = np.array(arr[0],dtype=np.int64)
                                bboxes = np.array(arr[3])
                                difficultylevel=arr[4][0]
                                seq_ex = self.generate_tf_record(img, cellmatrix, rowmatrix, colmatrix, bboxes,difficultylevel)
                                writer.write(seq_ex.SerializeToString())
                            print('Thread :',threadnum,' Completed in ',time.time()-starttime,' ' ,output_file_name,'with len:',(len(data_arr)))
                            print('level 1: ',all_difficulty_levels[0],', level 2: ',all_difficulty_levels[1],', level 3: ',all_difficulty_levels[2],', level 4: ',all_difficulty_levels[3])
                        except Exception as e:
                            traceback.print_exc()
                            self.logger.write(traceback.format_exc())
                            # print('Thread :',threadnum,' Removing',output_file_name)
                            # os.remove(os.path.join(self.outtfpath,output_file_name))

        driver.stop_client()
        driver.quit()


    def write_to_tf(self,max_threads):

        starttime=time.time()
        threads=[]
        for threadnum in range(max_threads):
            proc = Process(target=self.write_tf, args=(self.filesize, threadnum,))
            proc.start()
            threads.append(proc)

        for proc in threads:
            proc.join()
        print(time.time()-starttime)
