"""
-------------
This is the multi-process version
"""
import os
import codecs
import numpy as np
import math
from dota_utils import GetFileFromThisRootDir
import cv2
import shapely.geometry as shgeo
import dota_utils as util
import copy
from multiprocessing import Pool
from functools import partial
import time
import argparse

def get_has_airport_dict(names):
    '''

    :param names: name list.  name format: SAR_000201937_005_001_005_L2___7733_7573_15913_16584  ~  name__x1_y1_x2_y2
    :return: {key=name: value=[x1, y1, x2, y2]}, value is location of this image which contain airport
    '''
    delta = 500
    has_airport_dict = {}
    for name in names:
        img_name, new_loc = name.split('___')[0], name.split('___')[1].split('_')
        x1, y1, x2, y2 = int(new_loc[0])-delta, int(new_loc[1])-delta, int(new_loc[2])+delta, int(new_loc[3])+delta  # add 1000: aviod offset influence
        has_airport_dict[img_name] = has_airport_dict.get(img_name, []) + [x1, y1, x2, y2]
    return has_airport_dict

#names = ['SAR_000201937_005_001_005_L2___6039_9267_14220_18278', 'SAR_000210633_006_001_005_L2___2631_12522_13571_20296', 'SAR_000240448_003_001_008_L2___6208_13787_18091_22623', 'SAR_000242559_010_001_005_L2___7895_10735_16358_21863', 'SAR_000251628_002_001_008_L2___7332_12651_19348_22122', 'SAR_000256381_001_001_003_L2___8834_9235_17535_22897', 'SAR_000262176_004_001_004_L2___5433_8017_11762_20747', 'SAR_000300666_003_001_003_L2___10139_12503_19595_19928', 'SAR_000301490_005_001_002_L2___7778_9695_15447_20259', 'SAR_000302054_006_001_002_L2___7978_9197_16060_19509', 'SAR_000302243_001_001_005_L2___2256_15644_13603_24240', 'SAR_000302812_002_001_009_L2___4337_10815_16553_16884', 'SAR_000302909_051_001_002_L2___8655_9871_12948_20745', 'SAR_000304534_004_001_003_L2___9711_10122_19899_18404', 'SAR_000305052_003_001_007_L2___7172_10796_15854_19327', 'SAR_000305958_002_001_002_L2___10032_9123_14660_20626', 'SAR_000305958_005_001_006_L2___4676_14282_16785_22665', 'SAR_000306374_011_001_006_L2___12233_12233_21878_19290', 'SAR_000306995_004_001_004_L2___7189_13021_17962_20307', 'SAR_000307868_001_001_003_L2___8119_12389_19712_18672', 'SAR_000310015_001_001_004_L2___6662_10199_17585_17306', 'SAR_000311503_005_001_003_L2___3811_8812_13043_18883', 'SAR_000311504_013_001_002_L2___4183_13460_8913_25830', 'SAR_000311672_001_001_002_L2___6319_12997_18838_24479', 'SAR_000314140_018_001_002_L2___7681_8539_17534_22114', 'SAR_000314811_003_001_003_L2___8290_10814_12296_22029', 'SAR_000316280_005_001_002_L2___12462_10814_19276_20887', 'SAR_000319504_004_001_004_L2___5628_11685_12931_24488', 'SAR_000319927_001_001_004_L2___9627_10482_20324_15546', 'SAR_000320741_002_001_004_L2___10039_12142_18626_21250', 'SAR_000321737_005_001_003_L2___11419_8421_20178_16708', 'SAR_000322764_005_001_002_L2___4499_13198_12436_22335', 'SAR_000324250_003_001_012_L2___3371_9292_9869_20192', 'SAR_000326271_007_001_017_L2___12653_8927_22228_17062', 'SAR_000328955_001_001_008_L2___7044_9013_15323_17226', 'SAR_000334407_005_001_004_L2___7811_10464_19886_16318', 'SAR_000342384_013_001_004_L2___9474_11191_19987_22637', 'SAR_000343685_010_001_004_L2___4025_15755_14586_24412', 'SAR_000349887_001_001_008_L2___7738_12019_21404_20416', 'SAR_000350409_004_001_003_L2___7551_12873_21435_22802', 'SAR_000350491_024_001_003_L2___9510_12447_13455_24369', 'SAR_000351041_004_001_003_L2___6431_13667_21349_22243', 'SAR_000351514_001_001_005_L2___6157_8782_14412_17235', 'SAR_000360308_004_001_002_L2___3394_9150_13134_19628', 'SAR_000388150_006_001_002_L2___3254_11125_16272_21797', 'SAR_000395312_001_001_002_L2_HH___7553_7504_14389_17140', 'SAR_000396183_001_001_005_L2___6225_8122_20322_15505', 'SAR_000396640_001_001_002_L2_HH___6526_7056_11257_15921']
names = ['download_andersen_1___26388_12603_34068_17526', 'download_japan_jiashouna_1___11520_15247_18078_22338']
has_airport_dict = get_has_airport_dict(names)
def filter_non_airport_img(left, up, right, down, img_name, has_airport_dict, subsize1):
    '''
        filter imgs which not contain airport.
    :param left:
    :param up:
    :param img_name:
    :param has_airport_dict:
    :param subsize1: big_img(airport_img) size
    :param subsize2: small_img(to detect plane img) size
    :return:
    '''

    loc = has_airport_dict.get(img_name, [])
    if loc == []:
        return True

    if loc[0] >= right:
        return False
    if left >= loc[2]:
        return False
    if loc[3] <= up:
        return False
    if loc[1] >= down:
        return False
    return True

def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def split_single_warp(name, split_base, rate, extent):
    split_base.SplitSingle(name, rate, extent)

class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 code = 'utf-8',
                 gap=512,
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext = '.tif',
                 padding=True,
                 num_process=8
                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        :param padding: if to padding the images so that all the images have the same size
        """
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = os.path.join(self.basepath, 'images')
        self.labelpath = os.path.join(self.basepath, 'labelTxt')
        self.outimagepath = os.path.join(self.outpath, 'images')
        self.outlabelpath = os.path.join(self.outpath, 'labelTxt')
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.padding = padding
        self.num_process = num_process
        self.pool = Pool(num_process)
        print('padding:', padding)

        # pdb.set_trace()
        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
        if not os.path.isdir(self.outimagepath):
            # pdb.set_trace()
            os.mkdir(self.outimagepath)
        if not os.path.isdir(self.outlabelpath):
            os.mkdir(self.outlabelpath)
        # pdb.set_trace()
    ## point: (x, y), rec: (xmin, ymin, xmax, ymax)
    # def __del__(self):
    #     self.f_sub.close()
    ## grid --> (x, y) position of grids
    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def saveimagepatches(self, img, subimgname, left, up):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            #print('count:', count)
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
                outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
                count = count + 1
            elif (count == (pos + 1)%5):
                count = count + 1
                continue

            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        mask_poly = []
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                         (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]),
                                         (obj['poly'][6], obj['poly'][7])])
                if (gtpoly.area <= 0):
                    continue
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

                # print('writing...')
                if (half_iou == 1):
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    outline = ' '.join(list(map(str, polyInsub)))
                    outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    f_out.write(outline + '\n')
                elif (half_iou > 0):
                #elif (half_iou > self.thresh):
                  ##  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    out_poly = list(inter_poly.exterior.coords)[0: -1]
                    if len(out_poly) < 4:
                        continue

                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])

                    if (len(out_poly) == 5):
                        #print('==========================')
                        out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                    elif (len(out_poly) > 5):
                        """
                            if the cut instance is a polygon with points more than 5, we do not handle it currently
                        """
                        continue
                    if (self.choosebestpoint):
                        out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                    polyInsub = self.polyorig2sub(left, up, out_poly2)

                    for index, item in enumerate(polyInsub):
                        if (item <= 1):
                            polyInsub[index] = 1
                        elif (item >= self.subsize):
                            polyInsub[index] = self.subsize
                    outline = ' '.join(list(map(str, polyInsub)))
                    if (half_iou > self.thresh):
                        outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        ## if the left part is too small, label as '2'
                        outline = outline + ' ' + obj['name'] + ' ' + '2'
                    f_out.write(outline + '\n')
                #else:
                 #   mask_poly.append(inter_poly)
        self.saveimagepatches(resizeimg, subimgname, left, up)

    def SplitSingle(self, name, rate, extent):
        """
            split a single image and ground truth
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        print("processing:", name)
        img = cv2.imread(os.path.join(self.imagepath, name + extent))
        if np.shape(img) == ():
            return
        fullname = os.path.join(self.labelpath, name + '.txt')
        objects = util.parse_dota_poly2(fullname)
        for obj in objects:
            obj['poly'] = list(map(lambda x:rate*x, obj['poly']))
            #obj['poly'] = list(map(lambda x: ([2 * y for y in x]), obj['poly']))

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + '___' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                if filter_non_airport_img(left, up, right, down, name, has_airport_dict, subsize1=6144) == True:
                    self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):
        """
        :param rate: resize rate before cut
        """
        imagelist = GetFileFromThisRootDir(self.imagepath)
        imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]
        if self.num_process == 1:
            for name in imagenames:
                self.SplitSingle(name, rate, self.ext)
        else:

            # worker = partial(self.SplitSingle, rate=rate, extent=self.ext)
            worker = partial(split_single_warp, split_base=self, rate=rate, extent=self.ext)
            self.pool.map(worker, imagenames)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='which folder need to split.')
    parser.add_argument('--split', help='location of img/label after split')
    args=parser.parse_args()
    split = splitbase(args.source, args.split, gap=200, subsize=1024) 
    '''
    split = splitbase(r'/home/dingjian/data/dota/val',
                       r'/home/dingjian/data/dota/valsplit',
                      gap=200,
                      subsize=1024,
                      num_process=8
                      )
    '''
    split.splitdata(1)

