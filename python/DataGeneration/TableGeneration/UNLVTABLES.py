import os
import cv2
from xml.etree import ElementTree
from tqdm import tqdm
import numpy as np
import pickle

class UNLVTABLES:

    def __init__(self,images_path,ocr_path,table_path,outputdir):
        self.images_path=images_path
        self.ocr_path=ocr_path
        self.table_path=table_path

        self.outputimgsdir=self.create_dir(os.path.join(self.create_dir(outputdir),'images'))
        self.outputbbdir=self.create_dir(outputdir,'bb')

        self.all_tables_data=[]

        self.all_words = []
        self.all_numbers = []
        self.all_others = []
        self.pickle_filename='unlv_distribution'

    def create_dir(self,path):
        if(not os.path.exists(path)):
            os.mkdir(path)
        return path

    def load_from_pickle(self):
        file=open(self.pickle_filename,'rb')
        #print('\nloaded from pickle')
        self.all_tables_data=pickle.load(file)
        file.close()
        return self.get_words_numbers_others()


    def store_to_pickle(self):
        #print('\nSaved Distribution')
        file=open(self.pickle_filename,'wb')
        pickle.dump(self.all_tables_data,file)
        file.close()

    def get_words_numbers_others(self):
        for arr in self.all_tables_data:
            counter = arr[1][0][1]
            self.all_words += counter['alphabet']
            self.all_numbers += counter['number']
            self.all_others += counter['other']
        return self.all_words,self.all_numbers,self.all_others

    def store_tables_and_bboxes(self):
        for filename in tqdm(os.listdir(self.images_path)):
            im = cv2.imread(os.path.join(self.images_path, filename))

            root = ElementTree.parse(os.path.join(self.table_path, filename.replace('.png', '.xml'))).getroot()
            im, table_coords,row_col_counter = self.table_rectangle(root, im)

            root = ElementTree.parse(os.path.join(self.ocr_path, filename.replace('.png', '.xml'))).getroot()
            im, tables_counters = self.words_rectangles(root, table_coords,row_col_counter, im)
            self.all_tables_data+=[[filename,tables_counters]]
            #cv2.imwrite(os.path.join('myimages',filename),im)

        self.store_to_pickle()
        #return self.get_words_numbers()
        return self.all_tables_data


    def get_distribution(self):

        if(os.path.exists(self.pickle_filename)):
            self.load_from_pickle()
            return self.get_words_numbers_others()
            #return self.get_words_numbers()

        # if(not os.path.exists('myimages')):
        #     os.mkdir('myimages')
        print('\nProcessing UNLV for distribution:')
        for filename in tqdm(os.listdir(self.images_path)):
            im = cv2.imread(os.path.join(self.images_path, filename))

            root = ElementTree.parse(os.path.join(self.table_path, filename.replace('.png', '.xml'))).getroot()
            im, table_coords,row_col_counter = self.table_rectangle(root, im)

            root = ElementTree.parse(os.path.join(self.ocr_path, filename.replace('.png', '.xml'))).getroot()
            im, tables_counters = self.words_rectangles(root, table_coords,row_col_counter, im)
            self.all_tables_data+=[[filename,tables_counters]]
            #cv2.imwrite(os.path.join('myimages',filename),im)

        self.store_to_pickle()
        #return self.get_words_numbers()
        return self.all_tables_data

    def get_transformed_pts(self,pts1, pts2, dim, imshape):
        return (int((pts1[0] / imshape[1]) * dim[0]), int((pts1[1] / imshape[0]) * dim[1])), (
        int((pts2[0] / imshape[1]) * dim[0]), int((pts2[1] / imshape[0]) * dim[1]))

    def get_numpy_coords(self,root, height):
        # this function will return in format of x0,y0,x1,y1
        coords_text = np.array([[coords.attrib, coords.text.strip()] for coords in root.iter('word')])
        all_coords = np.array([coords_text[:, 0]]).transpose()
        all_text = np.array([coords_text[:, 1]]).transpose()
        all_coords = np.array([[int(coords[0]['left']), height - int(coords[0]['top']), int(coords[0]['right']),
                                height - int(coords[0]['bottom'])] for coords in all_coords])
        return all_coords, all_text

    def get_gt_within_table(self,table_coords, words_coords, all_text):

        final_words_coords = []
        final_text = []
        for i in range(len(table_coords)):
            table_coord = np.array([table_coords[i, :]])
            mask = np.concatenate(([np.all(words_coords[:, :2] >= table_coord[:, :2], axis=1)],
                                   [np.all(words_coords[:, 2:] <= table_coord[:, 2:], axis=1)]), axis=0).transpose()
            trues = np.array([[True, True]])
            mask = np.all(mask == trues, axis=1)
            final_words_coords.append(words_coords[mask])
            final_text.append(all_text[mask])

        return np.array(final_words_coords), np.array(final_text)

    def words_rectangles(self,root, table_coords,row_col_counter, im):

        table_coords = np.array(table_coords)
        #word means alphabetic word, number means digit, other is a combination of both

        height, width, _ = im.shape
        all_words_coords, all_text = self.get_numpy_coords(root, height)
        masked_words_coords, masked_text = self.get_gt_within_table(table_coords, all_words_coords, all_text)
        #combinedcounter will have row_col_counter and word_type_counters combined
        combined_counter=[]
        for i in range(len(masked_words_coords)):
            word_type_counters = {'alphabet': [], 'number': [], 'other': []}
            words_coords = masked_words_coords[i]

            text = masked_text[i]

            for coords, s_txt in zip(words_coords, text):
                coords = np.array(coords)

                original_word=s_txt[0]
                temp_chr=original_word
                temp_chr=temp_chr.replace(',','').replace('-','').replace('.','').lower()


                # if(temp_chr[0] in alphabets):
                #     if(temp_chr in numbers):
                #         word_type_counters['other'].append(temp_chr)
                #     else:
                #         word_type_counters['alphabet'].append(temp_chr)
                # elif(temp_chr in numbers):
                #     word_type_counters['number'].append(temp_chr)

                if (temp_chr.isalpha()):
                    #print(original_word,temp_chr,' is alphabet')
                    word_type_counters['alphabet'].append(original_word)

                elif(temp_chr.isnumeric() or temp_chr.isdecimal()):
                    if(temp_chr.count('.')<=1):
                        #print(original_word,temp_chr, ' is number')
                        word_type_counters['number'].append(original_word)
                    else:
                        #print(original_word,temp_chr, ' is other')
                        word_type_counters['other'].append(original_word)
                else:
                    #print(original_word,temp_chr, ' is other')
                    word_type_counters['other'].append(temp_chr)


                x0, y0, x1, y1 = coords[0], coords[1], coords[2], coords[3]
                pts1, pts2 = (x0, y0), (x1, y1)
                cv2.rectangle(im, pts1, pts2, (0, 0, 255), 2)

            combined_counter.append([row_col_counter[i],word_type_counters])
        return im,combined_counter

    def table_rectangle(self,root, im):
        height, width, _ = im.shape
        table_coords = []
        colors=[(255,0,0),(0,255,0)]
        types=['row','column']
        combined_counter=[]
        for coords in root.iter('Table'):
            counter={'row':0,'column':0}

            coords_2 = coords.attrib
            x0, x1, y0, y1 = int(coords_2['x0']), int(coords_2['x1']), int(coords_2['y0']), int(coords_2['y1'])
            pts1, pts2 = (x0, y0), (x1, y1)
            table_coords.append([x0, y0, x1, y1])
            #cv2.rectangle(im, pts1, pts2, (0, 0, 255), 2)
            for child in coords:
                if(child.tag.lower()=='cell'):
                    break
                idx=types.index(child.tag.lower())
                coords_2 = child.attrib
                x0, x1, y0, y1 = int(coords_2['x0']), int(coords_2['x1']), int(coords_2['y0']), int(coords_2['y1'])
                pts1, pts2 = (x0, y0), (x1, y1)
                cv2.rectangle(im, pts1, pts2, colors[idx], 2)
                counter[types[idx]]+=1

            counter['row']+=1
            counter['column']+=1
            combined_counter.append(counter)


        return im, table_coords,combined_counter