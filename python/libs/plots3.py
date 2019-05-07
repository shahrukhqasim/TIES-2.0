from libs.configuration_manager import ConfigurationManager as gconfig
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import matplotlib.backends.backend_pdf
import networkx as nx



def get_cliques(adjacency_matrix,pval):


    rows, cols = np.where(adjacency_matrix == 1)
    #print(pval,' Number of ones: ',len(rows))

    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)

    # not_connected_nodes = []
    # for component in nx.connected_components(gr):
    #     component=list(component)
    #     if len(component) == 1:  # if it's not connected, there's only one node inside
    #         not_connected_nodes.append(component[0])
    # for node in not_connected_nodes:
    #     gr.remove_node(node)  # delete non-connected nodes

    #print('Edges:',len(rows), ' After deleting nodes:',len(gr.nodes()))
    cliques = list(nx.find_cliques(gr))
    cliques=[sorted(clique) for clique in cliques]
    return cliques

    #print(list(nx.find_cliques(gr)))

    #nx.draw(gr, node_size=20)
    #nx.draw(gr)
    #plt.savefig(filepath)

def calculate_acc(_gt,_pred, n):
    _pred = _pred[0:n, 0:n]
    _gt = _gt[0:n, 0:n]
    zeros=np.zeros(shape=(n,n))
    zeros[_gt ==_pred]=1
    meanvalue=np.mean(zeros[:])
    #print('Mean value:',meanvalue)
    #
    # cliquepreds = get_cliques(_pred,'Pred:')
    # cliquegt = get_cliques(_gt,'Gt:')
    # #print('predcliques:',cliquepreds)
    # correct = 0
    # total = 0
    # for subarr in cliquegt:
    #     total += 1
    #     if (subarr in cliquepreds):
    #         correct += 1
    #     # else:
    #     #]
            #print(subarr,'not in it')
    #return correct/total,cliquepreds,cliquegt
    return meanvalue

def plot_few(output_path, sample_number, data, test_samples=None):

    image = data['image']  # [max_height, max_width]
    gts = data['sampled_ground_truths']  # 3 x [max_entries, num_samples]
    preds = data['sampled_predictions']  # 3 x [max_entries, num_samples]
    #print(sample_number)
    #print(preds[1].shape)

    difficultylevel=data['global_features'][3]

    sampled_indices = data['sampled_indices']  # [max_entries, num_samples]
    num_vertices = data['global_features'][gconfig.get_config_param("dim_num_vertices", "int")]
    height, width = data['global_features'][gconfig.get_config_param("dim_height", "int")], \
                    data['global_features'][gconfig.get_config_param("dim_width", "int")]

    max_height, max_width = gconfig.get_config_param("max_image_height", "float"), \
                            gconfig.get_config_param("max_image_width", "float")

    height_inches = 10 * max_height / max_width

    vertices = data['vertex_features']  # [max_entries, num_vertex_features]

    height, width = int(height), int(width)
    num_vertices = int(num_vertices)

    image = image.astype(np.uint8)

    image = image[:, :, 0]

    assert len(gts) == 3 and len(preds) == 3

    # image = image[0:height, 0:width]
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    file_path = lambda type, it: os.path.join(output_path, ("%05d_%s.pdf") % (it, type))

    output_file_name_cells = file_path('cells', sample_number)
    output_file_name_rows = file_path('rows', sample_number)
    output_file_name_cols = file_path('cols', sample_number)
    output_file_paths = [output_file_name_cells, output_file_name_rows, output_file_name_cols]

    x1s = vertices[:, gconfig.get_config_param("dim_vertex_x_position", "int")]
    y1s = vertices[:, gconfig.get_config_param("dim_vertex_y_position", "int")]
    x2s = vertices[:, gconfig.get_config_param("dim_vertex_x2_position", "int")]
    y2s = vertices[:, gconfig.get_config_param("dim_vertex_y2_position", "int")]

    if test_samples is None:
        test_samples = np.random.randint(0, num_vertices, size=10)

    color_1 = (0, 0, 255)  # (Blue)
    color_2 = (0, 255, 0)  # (Green)
    color_3 = (255, 0, 0)  # (Red)
    color_4 = (255, 51, 153)  # (Pink)
    color_5 = (255, 153, 0)  # (Orange)


    accuracy_metric={'cells':0.0,'row':0.0,'column':0.0}
    arrnames=['cells','row','column']
    print('\n\nSample number new:',sample_number)

    for matrixtype in range(3):
        #print('\n',arrnames[type],':')
        #Get the pred for cells, rows and columns in sequence
        _pred = preds[matrixtype]

        # Get the gt for cells, rows and columns in sequence
        _gt = gts[matrixtype]

        # Compute Accuracy
        #calculate_acc(_gt,_pred, num_vertices)



        #print('\n',arrnames[type],': ' ,correct/total)
        #accuracy_metric[arrnames[matrixtype]],predcliques,gtcliques=calculate_acc(_gt,_pred,num_vertices)
        accuracy_metric[arrnames[matrixtype]]=calculate_acc(_gt, _pred, num_vertices)

        #continue
        samples = sampled_indices[matrixtype]

        samples_per_vertex = samples.shape[1]



        FP = 0
        #colors=np.random.randint(0,255,(len(predcliques),3))

        colors=[[0,0,255],[0,255,0],[255,0,0],[255,255,0],[0,255,255],[255,0,255]]
        lencolors=len(colors)

        already_added_ids=[]
        image_copy = image.copy()
        overlay=image.copy()
        alpha=0.4

        # for i,subrow in enumerate(predcliques):
        #     c=colors[i%lencolors]
        #     for id in subrow:
        #         if(id in already_added_ids):
        #             cv2.rectangle(overlay, (int(x1s[id])+10, int(y1s[id])), (int(x2s[id]+10), int(y2s[id])), (int(c[0]), int(c[1]), int(c[2])),-1)
        #         else:
        #             cv2.rectangle(overlay, (x1s[id], y1s[id]), (x2s[id], y2s[id]),(int(c[0]),int(c[1]),int(c[2])),-1)
        #         already_added_ids.append(id)
        #
        # cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha,0, image_copy)
        # print('Preds visualized')
        # # cv2.rectangle(image_copy, (1,1), (30,30),
        # #               color=(255,0,0))
        #
        # plt.figure(figsize=(10, height_inches), dpi=300)
        # plt.imshow(image_copy)
        # #plt.savefig(output_file_paths[matrixtype].replace('.pdf','.jpg'))

        for sample in range(len(test_samples)):
            sample_index = test_samples[sample]
            image_copy = image.copy()
            for i in range(samples_per_vertex):
                sample_index_pair = samples[sample_index, i]
                if _pred[sample_index, i] == 0 and _gt[sample_index, i] == 0:  # Blue
                    # print("here 1")
                    cv2.rectangle(image_copy, (x1s[sample_index_pair], y1s[sample_index_pair]),
                                  (x2s[sample_index_pair], y2s[sample_index_pair]), color=color_1)
                elif _pred[sample_index, i] == 1 and _gt[sample_index, i] == 1:  # Green
                    # print("here 2")
                    cv2.rectangle(image_copy, (x1s[sample_index_pair], y1s[sample_index_pair]),
                                  (x2s[sample_index_pair], y2s[sample_index_pair]), color=color_2)
                elif _pred[sample_index, i] == 1 and _gt[sample_index, i] == 0:  # Red
                    #print("here 3")
                    FP+=1
                    cv2.rectangle(image_copy, (x1s[sample_index_pair], y1s[sample_index_pair]),
                                  (x2s[sample_index_pair], y2s[sample_index_pair]), color=color_3)
                elif _pred[sample_index, i] == 0 and _gt[sample_index, i] == 1:  # Pink
                    #print("here 4")
                    FP+=1
                    cv2.rectangle(image_copy, (x1s[sample_index_pair], y1s[sample_index_pair]),
                                  (x2s[sample_index_pair], y2s[sample_index_pair]), color=color_4)
                else:
                    assert False


            cv2.rectangle(image_copy, (x1s[sample_index], y1s[sample_index]), (x2s[sample_index], y2s[sample_index]),
                          color=color_5)
            # cv2.rectangle(image_copy, (1,1), (30,30),
            #               color=(255,0,0))

            plt.figure(figsize=(10, height_inches), dpi=300)
            plt.imshow(image_copy)
            #plt.savefig(output_file_paths[type].replace('.pdf','.jpg'))
        #print('False Positives for ',arrnames[type], FP)

        pdf = matplotlib.backends.backend_pdf.PdfPages(output_file_paths[matrixtype])

        for fig in range(1, plt.gcf().number + 1):  ## will open an empty extra figure :(
            pdf.savefig(fig)
        pdf.close()
        plt.close('all')
        #print("PDF written")
    return accuracy_metric


