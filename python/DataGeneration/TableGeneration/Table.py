import random
import numpy as np
from DataGeneration.TableGeneration.Distribution import Distribution
import time


class Table:
    #Table Type:
    # 0: regular headers
    # 1: irregular headers

    #Border Type:
    # 0: complete border, 1: completely w/o borders, 2: with lines underhead, 3: internal borders

    def __init__(self,no_of_rows,no_of_cols,images_path,ocr_path,gt_table_path):
        self.distribution=Distribution(images_path,ocr_path,gt_table_path)
        #self.distribution_data=self.distribution.get_distribution()
        self.all_words,self.all_numbers,self.all_others=self.distribution.get_distribution()
        #self.words_distribution, self.numbers_distribution,self.others_distribution = len(self.all_words), len(self.all_numbers)
        self.no_of_rows=no_of_rows
        self.no_of_cols=no_of_cols
        self.tables_categories = {'types': [0, 1], 'probs': [0.5, 0.5]}
        self.borders_categories = {'types': [0, 1, 2, 3], 'probs': [0.25, 0.25, 0.25, 0.25]}
        self.border_type = random.choices(self.borders_categories['types'], weights=self.borders_categories['probs'])[0]
        self.table_type = random.choices(self.tables_categories['types'], weights=self.tables_categories['probs'])[0]
        self.spanflag=False

        self.idcounter=0
        #cell_types matrix will have 'n' and 'w' where 'w' means word and 'n' means number
        self.cell_types=np.chararray(shape=(self.no_of_rows,self.no_of_cols))

        #headers matrix will have 's' and 'h' where 'h' means header and 's' means simple text
        self.headers=np.chararray(shape=(self.no_of_rows,self.no_of_cols))

        #value at a pos means number of cols to span and -1 will show to skip that cell as part of spanned cols
        self.col_spans_matrix=np.zeros(shape=(self.no_of_rows,self.no_of_cols))

        #value at a pos means number of rows to span and -1 will show to skip that cell as part of spanned rows
        self.row_spans_matrix=np.zeros(shape=(self.no_of_rows,self.no_of_cols))
        #table matrix will have all the generated text (words and numbers)
        self.missing_cells=[]

        #will keep track of how many top rows and how many left columns are being considered as headers
        self.header_count={'r':2,'c':0}
        self.data_matrix = np.empty(shape=(self.no_of_rows,self.no_of_cols),dtype=object)


    def get_log_value(self):
        import math
        return int(math.log(self.no_of_rows*self.no_of_cols,2))

    def define_col_types(self):
        len_all_words=len(self.all_words)
        len_all_numbers=len(self.all_numbers)
        len_all_others=len(self.all_others)

        total = len_all_words+len_all_numbers+len_all_others

        prob_words = len_all_words / total
        prob_numbers = len_all_numbers / total
        prob_others=len_all_others/total
        # 0: number - 1: word
        for i,type in enumerate(random.choices(['n','w','r'], weights=[prob_numbers,prob_words,prob_others], k=self.no_of_cols)):
            self.cell_types[:,i]=type
        #make header to be word only
        self.cell_types[0:2,:]='w'

        # all text to be simple
        self.headers[:] = 's'
        #only first row is header (for now)
        self.headers[0:2, :] = 'h'


    def generate_random_text(self,type):
        result=''
        ids=[]
        if(type=='n'):
            out= random.sample(self.all_numbers,1)
        elif(type=='r'):
            out=random.sample(self.all_others,1)
        else:
            text_len=random.randint(1,2)
            out= random.sample(self.all_words,text_len)

        for e in out:
            result+='<span id='+str(self.idcounter)+'>'+str(e)+' </span>'
            ids.append(self.idcounter)
            self.idcounter+=1
        return result,ids

    # def generate_content(self):
    #
    #     for r in range(self.no_of_rows):
    #         for c in range(self.no_of_cols):
    #             out=self.generate_random_text(self.cell_types[r,c].decode('utf-8'))
    #             self.table[r][c]=' '.join(out)


    def agnostic_span_indices(self,maxvalue,max_lengths=-1):

        # This function can be used for row and col span indices, both
        span_indices = []
        span_lengths = []
        span_count = random.randint(1, 3)
        if(span_count>=maxvalue):
            return [],[]
        #print('\n\nmaxvalue, spancoujnt',maxvalue,span_count)
        indices = sorted(random.sample(list(range(0, maxvalue)), span_count))

        starting_index = 0
        for i, index in enumerate(indices):
            if (starting_index > index):
                continue



            max_lengths=maxvalue-index
            if(max_lengths<2):
                break
            len_span = random.randint(1, max_lengths)

            if (len_span > 1):
                span_lengths.append(len_span)
                span_indices.append(index)
                starting_index = index + len_span
        # print('span indices:', span_indices)
        # print('span lengths:',span_lengths)

        return span_indices, span_lengths


    def make_header_col_spans(self):
        header_span_indices,header_span_lengths=[],[]

        header_span_indices, header_span_lengths = self.agnostic_span_indices(self.no_of_cols)

        row_span_indices=[]
        for index,length in zip(header_span_indices,header_span_lengths):
            #0th row as for header
            self.spanflag=True
            self.col_spans_matrix[0,index]=length
            self.col_spans_matrix[0,index+1:index+length]=-1
            row_span_indices+=list(range(index,index+length))
        b=list(filter(lambda x: x not in row_span_indices, list(range(self.no_of_cols))))
        self.row_spans_matrix[0,b]=2
        self.row_spans_matrix[1,b]=-1

        if(self.table_type==1): #if irregular
            self.create_irregular_header()


    def create_irregular_header(self):
        colnumber=0
        #random row spans
        #-2 to exclude top 2 rows of header and -1 so it won't occupy the complete column
        span_indices, span_lengths = self.agnostic_span_indices(self.no_of_rows-2)
        span_indices=[x+2 for x in span_indices]

        for index, length in zip(span_indices, span_lengths):
            self.spanflag=True
            # 0th row as for header
            self.row_spans_matrix[index,colnumber]=length
            self.row_spans_matrix[index+1:index+length,colnumber]=-1
        self.headers[:,colnumber]='h'
        self.header_count['c']+=1



    def generate_missing_cells(self):
        missing=np.random.random(size=(self.get_log_value(),2))
        missing[:,0]=(self.no_of_rows - 1 - self.header_count['r'])*missing[:,0]+self.header_count['r']
        missing[:, 1] = (self.no_of_rows -1 - self.header_count['c']) * missing[:, 1] + self.header_count['c']
        for arr in missing:
            self.missing_cells.append((int(arr[0]), int(arr[1])))

    def create_style(self):
        style = "<head><style>"
        style += "html{width:1366px;height:768px;background-color: white;}table{"

        # random center align
        if (random.randint(0, 1) == 1):
            style += "text-align:center;"

        style += """border-collapse:collapse;}td,th{padding:6px;padding-left: 15px;padding-right: 15px;"""

        if(self.border_type==0):
            style += """ border:1px solid black;} """
        elif(self.border_type==2):
            style += """border-bottom:1px solid black;}"""
        elif(self.border_type==3):
            style+="""border-left: 1px solid black;}
                       th{border-bottom: 1px solid black;} table tr td:first-child, 
                       table tr th:first-child {border-left: 0;}"""
        else:
            style+="""}"""


        style += "</style></head>"
        return style

    def create_html(self):

        temparr=['td', 'th']
        html="""<html>"""
        html+=self.create_style()
        html+="""<body><table>"""
        for r in range(self.no_of_rows):
            html+='<tr>'
            for c in range(self.no_of_cols):

                row_span_value = int(self.row_spans_matrix[r, c])
                col_span_value = int(self.col_spans_matrix[r, c])
                htmlcol = temparr[['s', 'h'].index(self.headers[r][c].decode('utf-8'))]

                if (row_span_value == -1):
                    self.data_matrix[r, c] = self.data_matrix[r - 1, c]
                    continue
                elif(row_span_value>0):
                    html += '<' + htmlcol + ' rowspan=\"' + str(row_span_value) + '"'
                else:
                    if(col_span_value==0):
                        if (r, c) in self.missing_cells:
                            html += """<td></td>"""
                            continue
                    if (col_span_value == -1):
                        self.data_matrix[r, c] = self.data_matrix[r, c - 1]
                        continue
                    html += '<' + htmlcol + """ colspan=""" + str(col_span_value)




                # htmlcol=temparr[['s','h'].index(self.headers[r][c].decode('utf-8'))]

                # if(row_span_value!=0):
                #     html+='<'+htmlcol+' rowspan=\"'+str(row_span_value)+'"'
                # else:
                #     if(col_span_value==-1):
                #         self.data_matrix[r,c] = self.data_matrix[r,c-1]
                #         continue
                #
                #     html+='<'+htmlcol+""" colspan="""+str(col_span_value)

                out,ids = self.generate_random_text(self.cell_types[r, c].decode('utf-8'))
                html+='>'+out+'</'+htmlcol+'>'

                self.data_matrix[r,c]=ids

                #html+='<'+htmlcol+'>'+''.join(out)+'</'+htmlcol+'>'
            html += '</tr>'

        html+="""</table></body></html>"""
        return html

    def create_matrix(self,arr,ids):
        matrix=np.zeros(shape=(ids,ids))
        for subarr in arr:
            for element in subarr:
                matrix[element,subarr]=1
        return matrix

    def create_col_matrix(self):
        all_cols=[]

        for col in range(self.no_of_cols):
            single_col = []
            for subarr in self.data_matrix[:,col]:
                if(subarr is not None):
                    single_col+=subarr
            all_cols.append(single_col)
        return self.create_matrix(all_cols,self.idcounter)

    def create_row_matrix(self):
        all_rows=[]

        for row in range(self.no_of_rows):
            single_row=[]
            for subarr in self.data_matrix[row,:]:
                if(subarr is not None):
                    single_row+=subarr
            all_rows.append(single_row)
        return self.create_matrix(all_rows,self.idcounter)

    def create_cell_matrix(self):
        all_cells=[]
        for row in range(self.no_of_rows):
            for col in range(self.no_of_cols):
                if(self.data_matrix[row,col] is not None):
                    all_cells.append(self.data_matrix[row,col])
        return self.create_matrix(all_cells,self.idcounter)

    def difficulty_level(self):
        #variables to consider:
        #- self.spanflag - self.tabletype(1 means spanned rows) - self.bordertype
        difficultylevel=1
        if(self.spanflag==False):
            if(self.border_type==0):
                difficultylevel=1
            else:
                difficultylevel=2
        else:
            difficultylevel=3

        return difficultylevel


    def create(self):
        self.define_col_types()
        self.generate_missing_cells()
        local_span_flag=random.choices([True,False],weights=[0.5,0.5])[0]
        local_span_flag=True
        if(local_span_flag):
            self.make_header_col_spans()
        html=self.create_html()
        cells_matrix,cols_matrix,rows_matrix=self.create_cell_matrix(),\
                                             self.create_col_matrix(),\
                                             self.create_row_matrix()
        difficultylevel=self.difficulty_level()
        # print('Headers:',self.headers)
        # print('colspans:', self.col_spans_matrix)
        # print('rowspans:', self.row_spans_matrix)
        # print('datamatrix:', self.data_matrix)
        # print('missing cells:', self.missing_cells)
        return cells_matrix,cols_matrix,rows_matrix,self.idcounter,html,difficultylevel

        #
        # # self.add_spans()
        # f=open('file.html','w')
        # f.write(html)
        # f.close()
