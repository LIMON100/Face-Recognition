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
            label_dir = self.__get_label_dict(train_img_dir)


        ## __get training path

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


if __name__ == "__main__":
    para_dict = 'home'
    cls = Classification(para_dict)