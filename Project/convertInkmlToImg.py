import sys, os, argparse
import xml.etree.ElementTree as ET
import numpy as np
from skimage.draw import line
from skimage.io import imread, imsave
import scipy.ndimage as ndimage
import pickle

import warnings
warnings.filterwarnings('ignore')

def parse_inkml(inkml_file_abs_path):
    if inkml_file_abs_path.endswith('.inkml'):
        tree = ET.parse(inkml_file_abs_path)
        root = tree.getroot()
        doc_namespace = "{http://www.w3.org/2003/InkML}"
        'Stores traces_all with their corresponding id'
        traces_all_list = [{'id': trace_tag.get('id'),
                            'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
                                            for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                                        else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
                                            for axis_coord in coord.split(' ')] \
                                    for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                                    for trace_tag in root.findall(doc_namespace + 'trace')]

        'convert in dictionary traces_all  by id to make searching for references faster'
        traces_all = {}
        for t in traces_all_list:
            traces_all[t["id"]] = t["coords"]
        #print("traces_alllalalalal",traces_all)
        #traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))
        return traces_all
    else:
        print('File ', inkml_file_abs_path, ' does not exist !')
        return {}

#get traces of data from inkml file and convert it into bmp image
def get_traces_data(traces_dict, id_set = None):
    'Accumulates traces_data of the inkml file'
    traces_data_curr_inkml=[]
    if id_set == None:
        id_set = traces_dict.keys()
    #this range is specified by values specified in the lg file
    for i in id_set: #use function for getting the exact range
        traces_data_curr_inkml.append(traces_dict[i])
        #print("trace for stroke"+str(i)+" :"+str(traces_data_curr_inkml))
    #convert_to_imgs(traces_data_curr_inkml, box_axis_size=box_axis_size)
    return traces_data_curr_inkml

def get_min_coords(traces):
    x_coords = [coord[0] for coord in traces]
    #print("xcoords"+str(x_coords))
    y_coords = [coord[1] for coord in traces]
    min_x_coord=min(x_coords)
    min_y_coord=min(y_coords)
    max_x_coord=max(x_coords)
    max_y_coord=max(y_coords)
    return min_x_coord, min_y_coord, max_x_coord, max_y_coord

'shift pattern to its relative position'
def shift_trace(traces, min_x, min_y):
    shifted_trace = [[coord[0] - min_x, coord[1] - min_y] for coord in traces]
    return shifted_trace

'Scaling: Interpolates a pattern so that it fits into a box with specified size'
def scaling(traces, scale_factor=1.0):
    interpolated_trace = []
    'coordinate convertion to int type necessary'
    interpolated_trace = [[round(coord[0] * scale_factor), round(coord[1] * scale_factor)] for coord in traces]
    return interpolated_trace

def center_pattern(traces, max_x, max_y, box_axis_size):
    x_margin = int((box_axis_size - max_x) / 2)
    y_margin = int((box_axis_size - max_y) / 2)
    return shift_trace(traces, min_x= -x_margin, min_y= -y_margin)

def draw_pattern(traces,pattern_drawn, box_axis_size):
    ' SINGLE POINT TO DRAW '
    if len(traces) == 1:
            x_coord = traces[0][0]
            y_coord = traces[0][1]
            pattern_drawn[y_coord, x_coord] = 0.0 #0 means black
    else:
        ' TRACE HAS MORE THAN 1 POINT '
        'Iterate through list of traces endpoints'
        for pt_idx in range(len(traces) - 1):
                'Indices of pixels that belong to the line. May be used to directly index into an array'
                #print("draw line : ",traces[pt_idx], traces[pt_idx+1])
                linesX = linesY = []
                oneLineX, oneLineY = line(r0=traces[pt_idx][1], c0=traces[pt_idx][0],
                                   r1=traces[pt_idx + 1][1], c1=traces[pt_idx + 1][0])

                linesX = np.concatenate(
                    [ oneLineX, oneLineX, oneLineX+1 ])
                linesY = np.concatenate(
                    [ oneLineY+1, oneLineY, oneLineY])

                linesX[linesX<0] = 0
                linesX[linesX>=box_axis_size] = box_axis_size-1

                linesY[linesY<0] = 0
                linesY[linesY>=box_axis_size] = box_axis_size-1

                pattern_drawn[ linesX, linesY] = 0.0
                # pattern_drawn[ oneLineX, oneLineY ] = 0.0
    return pattern_drawn

def convert_to_imgs(traces_data, box_axis_size): #trace_all contains coords only for 1 id
    pattern_drawn = np.ones(shape=(box_axis_size, box_axis_size), dtype=np.float32)
    # Special case of inkml file with zero trace (empty)
    if len(traces_data) == 0:
        return np.matrix(pattern_drawn * 255, np.uint8)

    'mid coords needed to shift the pattern'
    #print("traces_all['coords']"+str(traces_data))
    min_x, min_y, max_x, max_y = get_min_coords([item for sublist in traces_data for item in sublist]  )
    #print("min_x, min_y, max_x, max_y",min_x, min_y, max_x, max_y)
    'trace dimensions'
    trace_height, trace_width = max_y - min_y, max_x - min_x
    if trace_height == 0:
        trace_height += 1
    if trace_width == 0:
        trace_width += 1
    '' 'KEEP original size ratio' ''
    trace_ratio = (trace_width) / (trace_height)
    box_ratio = box_axis_size / box_axis_size #Wouldn't it always be 1
    scale_factor = 1.0
    '' 'Set \"rescale coefficient\" magnitude' ''
    if trace_ratio < box_ratio:
        scale_factor = ((box_axis_size-1) / trace_height)
    else:
        scale_factor = ((box_axis_size-1) / trace_width)
    #print("scale f : ", scale_factor)
    for traces_all in traces_data:
        'shift pattern to its relative position'
        shifted_trace= shift_trace(traces_all, min_x=min_x, min_y=min_y)
        #print("shifted : "  , shifted_trace)
        'Interpolates a pattern so that it fits into a box with specified size'
        'method: LINEAR INTERPOLATION'
        try:
            scaled_trace = scaling(shifted_trace,scale_factor)
            #print("inter : ", scaled_trace)
        except Exception as e:
            print(e)
            print('This data is corrupted - skipping.')

        'Get min, max coords once again in order to center scaled patter inside the box'
        #min_x, min_y, max_x, max_y = get_min_coords(interpolated_trace)
        centered_trace = center_pattern(scaled_trace, max_x=trace_width*scale_factor, max_y=trace_height*scale_factor, box_axis_size=box_axis_size-1)
        #print(" centered : " , centered_trace)
        'Center scaled pattern so it fits a box with specified size'
        pattern_drawn = draw_pattern(centered_trace, pattern_drawn,box_axis_size=box_axis_size)
        #print("pattern size", pattern_drawn.shape)
        #print(np.matrix(pattern_drawn, np.uint8))
    return np.matrix(pattern_drawn * 255, np.uint8)

## Functions dedicated to generating hypotheses
# It takes as input inkml traces (strokes as sequences of 2D points)
#                   one or several LG hypotheses
#                   the dimension of the output images
# It returns an np.array containing imgs generated from the hyptoheses
#   the first dimension is the number of hypotheses
#   the seconds and third dimension are the 2D dimensions (dim x dim)
# See the function called genHypothesesExample for more info.
def genImgFromLGHypotheses(inkmltraces, LG, dim):
    if type(LG) == str: LG = LG.split("\n")
    # Get list of stroke ids selected for each hypothetic symbol in LG
    symList = parseLG(LG)
    # Get traces belonging to each hypothetic symbol in symList
    tracesSelList = getStrokesFromLG(inkmltraces, symList)
    imgs = np.ndarray([len(LG), dim, dim])
    for idx, trace in enumerate(tracesSelList):
        imgs[idx] = convert_to_imgs(trace, dim)
    return imgs
def parseLG(LG):
    sym = []
    for lg in LG:
        # Remove whitespace and split by ","
        lg = lg.replace(" ", "").replace('\n', '').split(',')
        #print(lg)
        if lg[0] != "O": continue
        # Select symbol id and associated stroke id (only integers)
        sym.append([ lg[1], [ l for l in lg[4:]] ])
    return sym
def getStrokesFromLG(inkmltraces, symList):
    strokes = []
    for sym in symList:
        strokes.append(get_traces_data(inkmltraces, sym[1]))
    return strokes
def genHypothesesExample(file):
    # This example will fail if the inkml file does not have at least 4 strokes
    #   KeyError: '3'
    if os.path.isdir(file):
        file = file + os.sep + os.listdir(file)[np.random.randint(len(file))]
    print("genHypothesesExample: file =", file)
    # Get all traces/strokes from the inkml file
    traces = parse_inkml(file)
    # Use a simple lg of 3 symbols and 4 strokes
    LG = ['O, sym1, 2, 1.0, s1',
          'O, sym2, +, 1.0, s2, s3',
          'O, sym3, 2, 1.0, s4']
    # Main call, this function returns all images
    #   3 here, according to our lg hypotheses above (var LG)
    imgs = genImgFromLGHypotheses(traces, LG, 300)
    # Add padding and gaussian blur yourself
    # Plot returned images
    import matplotlib.pyplot as plt
    N = imgs.shape[0]
    for i in range(N):
        ax = plt.subplot(np.ceil(np.sqrt(N)), np.ceil(np.sqrt(N)), i+1)
        plt.imshow(imgs[i], origin='upper', alpha = 1)
        ax.set_xticklabels([]);ax.set_yticklabels([])
    plt.show()

if __name__ == '__main__':
    # Uncomment to test these new functions
    # genHypothesesExample(sys.argv[1]):exit()

    ''' 2 usages :
    convertInkmlToImg.py file.inkml (dim) (padding) (outDir)
    convertInkmlToImg.py folder (dim) (padding) (outDir)

    Example
    python3 convertInkmlToImg.py ../../../DB_CRHOME/task2-validation-isolatedTest2013b 28 2 
    '''
    if len(sys.argv) < 2:
        print('\n + Usage:', sys.argv[0], ' (file|folder) dim padding outdir')
        print('\t+ {:<20} - required str'.format("(file|folder)"))
        print('\t+ {:<20} - optional int (def = 300)'.format("dim"))
        print('\t+ {:<20} - optional int (def =  0)'.format("padding"))
        exit()
    else:
        if os.path.isfile(sys.argv[1]):
            FILES = [sys.argv[1]]
        else:
            from glob import glob
            if sys.argv[1][-1] != os.sep: sys.argv[1] += os.sep
            FILES = glob(sys.argv[1]+os.sep+"*.inkml")
        
        folder_name = sys.argv[1].split(os.sep)[-2]

        save_path = "data_png_" + folder_name if len(sys.argv) < 5 else sys.argv[4]+"data_png_" + folder_name

        print("to : " + save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        dim = 300 if len(sys.argv) < 3 else int(sys.argv[2])
        padding = 0 if len(sys.argv) < 4 else int(sys.argv[3])

        print("Starting inkml to png conversion on {} file{}\n".format(
            len(FILES), "s" if len(FILES) > 1 else ""
            ))

        for idx, file in enumerate(FILES):

            img_path = os.sep.join(file.split(os.sep)[:-1])
            img_name = file.split(os.sep)[-1]
            img_basename = ".".join(img_name.split(".")[:-1])

            if os.path.isfile(save_path + os.sep + img_basename + '.png'): continue

            if not os.path.isfile(img_path + os.sep + img_name):
                print("\n\nInkml file not found:\n\t{}".format(img_path + os.sep + img_name))
                exit()

            traces = parse_inkml(img_path + os.sep + img_name)

            selected_tr = get_traces_data(traces)
            im = convert_to_imgs(selected_tr, dim)

            if padding > 0:
                im = np.lib.pad(im, (padding, padding), 'constant', constant_values=255)
            
            im = ndimage.gaussian_filter(im, sigma=(.5, .5), order=0)

            imsave(save_path + os.sep + img_basename + '.png',im)

            print("\t\t\rfile: {:>10} | {:>6}/{:}".format(img_basename, idx+1, len(FILES)), end="")

    print("\n\nFinished")
