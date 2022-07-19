import os
import numpy as np
import tensorflow as tf

class Classification():

    def __init__(self,para_dict):
        #----var parsing
        train_img_dir = para_dict['train_img_dir']
        test_img_dir = para_dict['test_img_dir']
        label_dict = para_dict['label_dict']

        #----get label names to number dictionary
        if label_dict is None:
            label_dict = self.__get_label_dict(train_img_dir)
            #print(label_dict)


        #----read training set paths and labels
        train_paths, train_labels = self.__get_paths_labels(train_img_dir,label_dict)
        print("train path shape:{}, train label shape:{}".format(train_paths.shape,train_labels.shape))

        #----read test set paths and labels
        if test_img_dir is not None:
            test_paths, test_labels = self.__get_paths_labels(test_img_dir,label_dict)
            print("test path shape:{}, test label shape:{}".format(test_paths.shape, test_labels.shape))

        #----local var to global
        self.train_img_dir = train_img_dir
        self.label_dict = label_dict
        self.train_paths = train_paths
        self.train_labels = train_labels

        if test_img_dir is not None:
            self.test_img_dir = test_img_dir
            self.test_paths = test_paths
            self.test_labels = test_labels


    def model_init(self, para_dict):

        ##--var pasrsing
        model_shape = para_dict['model_shape']
        infer_method = para_dict['infer_method']
        loss_method = para_dict['loss_method']
        opti_method = para_dict['opti_method']

        ##--tf placeholder declaration
        tf_input = tf.placeholder(shape=model_shape, dtype=tf.float32, name='input')


        ##--inference section
        if infer_method == "":
            output = self.simple_resnet(tf_input)


        ##--loss section



        ##--optimizer section




    def __get_label_dict(self,img_dir):
        label_dict = dict()
        count = 0
        for obj in os.scandir(img_dir):
            if obj.is_dir():
                label_dict[obj.name] = count
                count += 1
        if count == 0:
            print("No dir in the ",img_dir)
            return None
        else:
            return label_dict

    def __get_paths_labels(self,img_dir,label_dict):
        #----var
        img_format = {'png', 'jpg', 'bmp'}
        re_paths = list()
        re_labels = list()

        #----read dirs
        dirs = [obj.path for obj in os.scandir(img_dir) if obj.is_dir()]
        if len(dirs) == 0:
            print("No dirs in the ",img_dir)
        else:
            #-----read paths of each dir
            for dir_path in dirs:
                path_temp = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
                if len(path_temp) == 0:
                    print("No images in the ",dir_path)
                else:
                    #----get the label number from class name
                    label_num = dir_path.split("\\")[-1]
                    label_num = label_dict[label_num]

                    #----create the label array
                    label_temp = np.ones(len(path_temp), dtype=np.int32) * label_num

                    #----collect paths and labels
                    re_paths.extend(path_temp)
                    re_labels.extend(label_temp)

            #----list to numpy array
            re_paths = np.array(re_paths)
            re_labels = np.array(re_labels)

            #----shuffle
            indice = np.random.permutation(re_paths.shape[0])
            re_paths = re_paths[indice]
            re_labels = re_labels[indice]

        return re_paths, re_labels



if __name__ == "__main__":
    train_img_dir = r"H:\Android\New 7\Masked_face\Deep Learning masked face detection, recognition\5. Tensorflow introduction and quick guide\train"
    test_img_dir = r"H:\Android\New 7\Masked_face\Deep Learning masked face detection, recognition\5. Tensorflow introduction and quick guide\val"
    label_dict = None

    para_dict = {"train_img_dir":train_img_dir, "test_img_dir":test_img_dir, "label_dict":label_dict}

    cls = Classification(para_dict)