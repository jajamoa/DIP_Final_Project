import csv
import os


def split_data(data,txt_path,ratio=[0.6,0.2,0.2]):
    total=len(data)
    train_size=round(total*ratio[0])
    val_size=round(total*ratio[1])
    test_size=total-train_size-val_size
    with open(txt_path+'train.txt','w') as file:
        for pic in data[:train_size]:
            file.write(pic+'\n')
    with open(txt_path+'val.txt','w') as file:
        for pic in data[train_size:train_size+val_size]:
            file.write(pic+'\n')
    with open(txt_path+'test.txt','w') as file:
        for pic in data[train_size+val_size:]:
            file.write(pic+'\n')



if __name__=='__main__':
    label_file_name='slice-level/Slice_level_label.csv'

    COVID_txt_path="dataTxt/COVID/"
    CAP_txt_path="dataTxt/CAP/"
    Normal_txt_path="dataTxt/Normal/"

    with open(label_file_name,'r') as file:
        f=csv.reader(file)
        labels_origin=list(f)
        patients_COVID=labels_origin[1:56]
        patients_CAP=labels_origin[56:]

        root_path='slice-level/'

        COVID_path='Covid-19/'
        CAP_path='Cap/'

        data_Normal=[]
        data_COVID=[]
        data_CAP=[]

        for patient in patients_COVID:
            Patient_path=COVID_path+patient[0]

            num_slice=len(os.listdir(root_path+Patient_path))

            
            for i,label in enumerate(patient[1:num_slice+1]):
                # print("#{}:\t{}".format(i,label))
                pic_name=COVID_path+patient[0]+'/CT%04d.png'%(i)
                if label=='1':
                    data_COVID.append(pic_name)
                else:
                    data_Normal.append(pic_name)

            break

        for patient in patients_CAP:
            Patient_path=CAP_path+patient[0]

            num_slice=len(os.listdir(root_path+Patient_path))

            
            for i,label in enumerate(patient[1:num_slice+1]):
                # print("#{}:\t{}".format(i,label))
                pic_name=CAP_path+patient[0]+'/CT%04d.png'%(i)
                if label=='1':
                    data_CAP.append(pic_name)
                else:
                    data_Normal.append(pic_name)

            break

        print(len(data_COVID))
        print(len(data_CAP))
        print(len(data_Normal))

        split_data(data_COVID,COVID_txt_path)
        split_data(data_CAP,CAP_txt_path)
        split_data(data_Normal,Normal_txt_path)


        
