import os,sys
import random

import pandas as pd
import csv
from tqdm import tqdm
from shutil import copyfile

ROOT_DOR_SLICE = '../../data/trainData/slice-level'
csv_name_slice = 'Slice_level_label.csv'
slice_roots = ['Cap', 'Covid-19']

ROOT_DOR_SUBJECT = '../../data/trainData/subject-level'
csv_name_subject = 'pseudo_label.csv'

RESULT_DOR = '../../data'

TRAIN_RATE = 0.6
VAL_RATE = 0.3
TEST_RATE = 0.1
assert TRAIN_RATE + VAL_RATE + TEST_RATE - 1.0 < 0.0001

random.seed(123)

def create_dir_not_exist(path):
    for length in range(1, len(path.split(os.path.sep))):
        check_path = os.path.sep.join(path.split(os.path.sep)[:(length+1)])
        if not os.path.exists(check_path):
            os.mkdir(check_path)
            print(f'Created Dir: {check_path}')


def get_slice_path_labels():
    image_labels = []

    df = pd.read_csv(os.path.join(ROOT_DOR_SLICE, csv_name_slice), index_col=0)
    for slice_root in slice_roots:
        person_root = os.path.join(ROOT_DOR_SLICE, slice_root)
        person_names = os.listdir(person_root)

        i = 0
        while i < len(person_names):
            if not person_names[i].startswith('cap') and not person_names[i].startswith('P'):
                person_names.pop(i)
                i -= 1
            i += 1

        for pname in tqdm(person_names):
            try:
                assert pname.startswith('cap') or pname.startswith('P')
            except:
                print(person_names)
                raise

            image_root = os.path.join(person_root, pname)
            image_names = os.listdir(image_root)

            i = 0
            while i < len(image_names):
                if not image_names[i].startswith('CT'):
                    image_names.pop(i)
                    i -= 1
                i += 1

            for iname in image_names:
                assert len(iname) == 10
                imindex = int(iname[2:6])

                label = int(df.loc[pname][imindex])
                ifullname = os.path.join(image_root, iname)

                if pname.startswith('P') and label:
                    label = 2

                image_labels.append((ifullname, label))

    return image_labels

def get_subject_path_labels():
    with open(os.path.join(ROOT_DOR_SUBJECT,csv_name_subject), 'r') as f:
        csvreader = csv.reader(f)
        subject_list = list(csvreader)
        subject_list=[(os.path.join(ROOT_DOR_SUBJECT,os.path.sep.join(item[1:])),int(item[0])) for item in subject_list]
        return subject_list

def shuffle_save():
    image_labels_slice = get_slice_path_labels()
    image_labels_subject = get_subject_path_labels()
    image_labels_slice.sort()
    image_labels_subject.sort()
    image_labels=image_labels_slice+image_labels_subject
    
    print(image_labels[:10])

    random.shuffle(image_labels)

    dor=os.path.join(RESULT_DOR,'pseudo')
    create_dir_not_exist(os.path.join(dor, "train"))
    create_dir_not_exist(os.path.join(dor, "test"))
    create_dir_not_exist(os.path.join(dor, "val"))


    for i, (name, label) in tqdm(enumerate(image_labels)):
        if i <= len(image_labels) * TRAIN_RATE:
            root = 'train'
        elif i <= len(image_labels) * (TRAIN_RATE + VAL_RATE):
            root = 'val'
        else:
            root = 'test'

        img_name='img'+str(i).rjust(5,'0')+'_'+str(label)+'.png'
        newname = os.path.join(dor,root)
        newname = os.path.join(newname,img_name)

        copyfile(name,newname)

shuffle_save()
