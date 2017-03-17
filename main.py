from __future__ import division
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input
from keras.models import Model
import os, cv2, sys
import numpy as np
from config import *
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions
from models import sam_vgg, sam_resnet, schedule_vgg, schedule_resnet, kl_divergence, correlation_coefficient, nss


def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith('.mat')]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_val_path + f for f in os.listdir(fixs_val_path) if f.endswith('.mat')]
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()
    fixs.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        Y = preprocess_maps(maps[counter:counter+b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps(fixs[counter:counter + b_s], shape_r_out, shape_c_out)
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian], [Y, Y, Y_fix]
        counter = (counter + b_s) % len(images)


def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian]
        counter = (counter + b_s) % len(images)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise NotImplementedError
    else:
        phase = sys.argv[1]
        x = Input((3, shape_r, shape_c))
        x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

        if version == 0:
            m = Model(input=[x, x_maps], output=sam_vgg([x, x_maps]))
            print("Compiling SAM-VGG")
            m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
        elif version == 1:
            m = Model(input=[x, x_maps], output=sam_resnet([x, x_maps]))
            print("Compiling SAM-ResNet")
            m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
        else:
            raise NotImplementedError

        if phase == 'train':
            if nb_imgs_train % b_s != 0 or nb_imgs_val % b_s != 0:
                print("The number of training and validation images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
                exit()

            if version == 0:
                print("Training SAM-VGG")
                m.fit_generator(generator(b_s=b_s), nb_imgs_train, nb_epoch=nb_epoch,
                                validation_data=generator(b_s=b_s, phase_gen='val'), nb_val_samples=nb_imgs_val,
                                callbacks=[EarlyStopping(patience=3),
                                           ModelCheckpoint('weights.sam-vgg.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True),
                                           LearningRateScheduler(schedule=schedule_vgg)])
            elif version == 1:
                print("Training SAM-ResNet")
                m.fit_generator(generator(b_s=b_s), nb_imgs_train, nb_epoch=nb_epoch,
                                validation_data=generator(b_s=b_s, phase_gen='val'), nb_val_samples=nb_imgs_val,
                                callbacks=[EarlyStopping(patience=3),
                                           ModelCheckpoint('weights.sam-resnet.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True),
                                           LearningRateScheduler(schedule=schedule_resnet)])

        elif phase == "test":
            # Output Folder Path
            output_folder = 'predictions/'

            if len(sys.argv) < 2:
                raise SyntaxError
            imgs_test_path = sys.argv[2]

            file_names = [f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            file_names.sort()
            nb_imgs_test = len(file_names)

            if nb_imgs_test % b_s != 0:
                print("The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
                exit()

            if version == 0:
                print("Loading SAM-VGG weights")
                m.load_weights('weights/sam-vgg_salicon_weights.pkl')
            elif version == 1:
                print("Loading SAM-ResNet weights")
                m.load_weights('weights/sam-resnet_salicon_weights.pkl')

            print("Predicting saliency maps for " + imgs_test_path)
            predictions = m.predict_generator(generator_test(b_s=b_s, imgs_test_path=imgs_test_path), nb_imgs_test)[0]


            for pred, name in zip(predictions, file_names):
                original_image = cv2.imread(imgs_test_path + name, 0)
                res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
                cv2.imwrite(output_folder + '%s' % name, res.astype(int))
        else:
            raise NotImplementedError
