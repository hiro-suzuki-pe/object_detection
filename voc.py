# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os.path as osp
import xml.etree.ElementTree as ElementTree
import numpy as np


def make_filepath_list(rootpath):
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')
    
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')
    
    train_img_list = list()
    train_anno_list = list()
    
    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
        
    val_img_list = list()
    val_anno_list = list()
    
    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_template % file_id)
        anno_path = (annopath_template % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)
        
    return train_img_list, train_anno_list, val_img_list, val_anno_list

class GetBBoxAndLabel(object):
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height):

        annotation = []
        xml = ElementTree.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            bndbox = []
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            grid = ['xmin', 'ymin', 'xmax', 'ymax']
            
            for gr in (grid):
                axis_value = int(bbox.find(gr).text) - 1
                if gr == 'xmin' or gr == 'xmax':
                    axis_value /= width
                else:
                    axis_value /= height
                bndbox.append(axis_value)

            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            annotation += [bndbox]

        return np.array(annotation)
