import os
import numpy as np

class Classification():

    def __init__(self):

        ## __var parsing
        train_img_dir = para_dict'train_img_dir']
        test_img_dir = para_dict'test_img_dir']
        label_dir = para_dict['label_dir']


        ## __get label names
        if label_dir is None:
            label_dict = self.__get_label_dict(train_img_dir)


        ## __get training path
        train_path, train_labels = self.__get_paths_labels(train_img_dir, label_dict)

        ## __get testing path

        ## convet local to global



    ## get label function
    def __get_label_dict(self, img_dir):
        label_dict = dict()
        count = 0

        for obj in os.scandir(img_dir):
            if obj.is_dir():
                label_dict[obj.name] = count
                count += 1

        if count == 0:
            print("No label found")
            return None
        else:
            return label_dict


    ## Train path extract
    def __get_paths_labels(self, img_dir, label_dict):
        img_format = {'jpg', 'png', 'bmp'}

        dirs = [obj.path for obj in os.scandir(img_dir) if obj.is_dir()]
        if len(dirs) == 0:
            print("no folder found")

        else:
            for dirs_path in dirs:
                path_temp = [file.path for file in os.scandir(dirs_path) if file.name.split('.')[-1] in img_format]

                if len(path_temp) == 0:
                    print("No img found...")
                else:
                    label_num = dirs_path.split('\\')[-1]
                    label_num = label_dict[label_num]

                    label_temp = np.ones(len(path_temp), dtype = np.int32) * label_num







if __name__ == "__main__":
    para_dict = 'home'
    cls = Classification(para_dict)