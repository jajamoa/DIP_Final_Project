import os,sys,cv2
import shutil
import random
import multiprocessing

ROOT = r'../../data'
ROOT_SUBJECT_LEVEL = os.path.join(ROOT, 'trainData', 'subject-level')
ROOT_SLICE_LEVEL = os.path.join(ROOT, 'trainData', 'slice-level')
ROOT_NEW = os.path.join(ROOT, 'end-to-end-filtered-he2')

assert os.path.exists(ROOT)
assert os.path.exists(ROOT_SLICE_LEVEL)
assert os.path.exists(ROOT_SUBJECT_LEVEL)

def create_dir_not_exist(path):
    for length in range(1, len(path.split(os.path.sep))):
        check_path = os.path.sep.join(path.split(os.path.sep)[:(length+1)])
        if not os.path.exists(check_path):
            os.mkdir(check_path)
            print(f'Created Dir: {check_path}')

def remove_invalid(patient_root:str):
    assert os.path.exists(patient_root)
    assert os.path.isdir(patient_root)

    image_list=os.listdir(patient_root)
    for img in image_list:
        if img.startswith('.'):
            root_del=os.path.join(patient_root,img)
            os.remove(root_del)

def checkSuffix(file_list):
    img_suffixs=['png']
    for file in file_list:
        if not (file.split('.')[-1] in img_suffixs):
            file_list.remove(file)
    return file_list

SEED = 10

list_patient_roots = []
enum = {'Cap': '1', 'Covid-19': '2', 'Non-infected': '0'}

for typ in enum.keys():
    root_typ = os.path.join(ROOT_SUBJECT_LEVEL, typ)
    for patient in os.listdir(root_typ):
        root_patient = os.path.join(root_typ, patient)
        assert os.path.exists(root_patient)
        if os.path.isdir(root_patient):
            list_patient_roots.append(
                (root_patient, enum[typ])
            )

typename = ['Cap', 'Covid-19']
for typ in typename:
    root_typ = os.path.join(ROOT_SLICE_LEVEL, typ)
    for patient in os.listdir(root_typ):
        root_patient = os.path.join(root_typ, patient)
        assert os.path.exists(root_patient)
        if os.path.isdir(root_patient):
            list_patient_roots.append(
                (root_patient, enum[typ])
            )

random.seed(SEED)
random.shuffle(list_patient_roots)
prp = {'train': 0.6, 'valid': 0.3, 'test': 0.1}

assert not os.path.exists(ROOT_NEW)
os.mkdir(ROOT_NEW)
for set_typ in prp.keys():
    root_set_typ = os.path.join(ROOT_NEW, set_typ)
    assert not os.path.exists(root_set_typ)
    os.mkdir(root_set_typ)

def copytree_he(proot,newroot):
    for item in os.listdir(proot):
        prev_filename = os.path.join(proot, item)
        new_filename = os.path.join(newroot, item)
        create_dir_not_exist(newroot)
        move_HE_img(prev_filename, new_filename)
#     sys.exit()

def move_HE_img(src_path,tar_path):
    image=cv2.imread(src_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.equalizeHist(image)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    image=clahe.apply(image)
    cv2.imwrite(tar_path,image)
    
# proot:'……\\trainData\\slice-level\\Covid-19\\P047'
# ROOT_NEW:'……\\trainData\\end-to-end
def move(para):
    i = para[0]
    proot = para[1][0]
    label = para[1][1]

    assert os.path.exists(proot)
    assert os.path.isdir(proot)
    remove_invalid(proot)

    if (os.listdir(proot) == []):
        print(f'{proot} IS EMPTY!!')
        return

    proot_split = proot.split(os.path.sep)

    if i <= int(prp['train'] * len(list_patient_roots)):
        newroot_split = ROOT_NEW.split(os.path.sep) + ['train']
    elif i <= int((prp['valid'] + prp['train']) * len(list_patient_roots)):
        newroot_split = ROOT_NEW.split(os.path.sep) + ['valid']
    else:
        newroot_split = ROOT_NEW.split(os.path.sep) + ['test']

    proot_split[-1] = label + '_' + proot_split[-1]
    if proot_split[-3] == 'slice-level':
        proot_split[-1] += '_slice_level'

    newroot_split += [proot_split[-1]]
    newroot = os.path.sep.join(newroot_split)

    copytree_he(proot,newroot)
    
    train_len = len(os.listdir(os.path.join(ROOT_NEW, 'train')))
    valid_len = len(os.listdir(os.path.join(ROOT_NEW, 'valid')))
    test_len = len(os.listdir(os.path.join(ROOT_NEW, 'test')))
    print(f'TRAIN: {train_len} VALID: {valid_len} TEST: {test_len}')

p = multiprocessing.Pool(80)
porc_list_patient_roots = [(i, (proot, label)) for i, (proot, label) in enumerate(list_patient_roots)]
p.map(move, porc_list_patient_roots)
p.close()
p.join()