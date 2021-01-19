import os
import random

import pandas as pd
from tqdm import tqdm
from shutil import copyfile

ROOT_DOR = '..\\..\\data\\trainData\\slice-level'
csv_name = 'Slice_level_label.csv'
slice_roots = ['Cap', 'Covid-19']

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

    df = pd.read_csv(os.path.join(ROOT_DOR, csv_name), index_col=0)
    for slice_root in slice_roots:
        person_root = os.path.join(ROOT_DOR, slice_root)
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


def shuffle_save():
    image_labels = get_slice_path_labels()
    image_labels.sort()

    random.shuffle(image_labels)

    dor_split=ROOT_DOR.split(os.path.sep)
    dor_split=dor_split[:-2]+['first_classifier']
    dor=os.path.sep.join(dor_split)
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
        s = name.split(os.path.sep)
        s[-1]='img'+str(i).rjust(5,'0')+'_'+str(label)+'.png'
        s = s[:3] + [s[-1]]
        s.insert(-1, 'first_classifier')
        s.insert(-1, root)
        newname = os.path.sep.join(s)

        copyfile(name,newname)

shuffle_save()
