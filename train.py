import os
import cv2
import keras
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import tifffile as tiff
from PIL import Image
import imgaug.augmenters as iaa
import segmentation_models as sm
import argparse

input_size=500
def get_training_augmentation(input_size):
    aug = iaa.Sequential([iaa.Resize({"height":input_size,"width":input_size}),
                          iaa.SomeOf((0, 7),
                                     [iaa.Fliplr(0.5),
                                      iaa.Flipud(0.5),
                                      iaa.AdditiveGaussianNoise(scale=0.10 * 255),
                                      iaa.Dropout(p=0.1),
                                      iaa.GaussianBlur(sigma=0.2),
                                      iaa.PiecewiseAffine(scale=0.03),
                                      iaa.ElasticTransformation(alpha=0.1, sigma=5)]),
                          ])
    return aug
def get_val_augmentation(input_size):
    aug = iaa.Sequential([iaa.Resize({"height":input_size,"width":input_size})])
    return aug


class Dataset:
    CLASSES = ['background', 'building']

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = tiff.imread(self.images_fps[i])
        image = Image.fromarray(image, 'RGBA')
        image = image.convert('RGB')
        image = np.array(image)
        mask = cv2.imread(self.masks_fps[i], 0)
        background_mask = (mask == 0)
        building_mask = 1 - background_mask
        mask = np.stack([building_mask], axis=-1)
        mask = mask.astype('int8')

        # apply augmentations
        if self.augmentation:
            mask = SegmentationMapsOnImage(mask, shape=image.shape)
            image, mask = self.augmentation(image=image, segmentation_maps=mask)
            mask = mask.arr
        # apply preprocessing
        if self.preprocessing:
            mask = SegmentationMapsOnImage(mask, shape=image.shape)
            image, mask = self.preprocessing(image=image, segmentation_maps=mask)
            mask = mask.arr
        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloader(keras.utils.Sequence):
    def __init__(self, dataset, batch_size=3, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

def main(model_name, weight_path, train_dir, train_label_dir, val_dir, val_label_dir, test_dir, test_label_dir, is_fine_tuning, batch_size, epochs):
    BACKBONE = 'resnet50'
    CLASSES = ['background', 'building']
    LR = 0.0001
    preprocess_input = sm.get_preprocessing(BACKBONE)
    n_classes = 1
    activation = 'sigmoid'
    input_size = 480
    if model_name.lower() == 'unet':
        model = sm.Unet(BACKBONE,
                        classes=n_classes,
                        activation=activation,
                        input_shape=(input_size, input_size, 3),
                        encoder_weights='imagenet',
                        encoder_freeze=is_fine_tuning,
                        )
    elif model_name.lower() == 'pspnet' :
        model = sm.PSPNet(BACKBONE,
                          classes=n_classes,
                          activation=activation,
                          input_shape=(input_size, input_size, 3),
                          encoder_weights='imagenet',
                          encoder_freeze=is_fine_tuning,
                          psp_dropout=0.1,
                          downsample_factor=4)
    else:
        raise Exception(model_name + ' not supported')
    optim = keras.optimizers.SGD(lr=LR, momentum=0.9, nesterov=True)
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    model.compile(optim, total_loss, metrics)
    if weight_path:
        model.load_weights(weight_path)
    train_dataset = Dataset(
        train_dir,
        train_label_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(input_size)
    )
    train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    assert train_dataloader[0][0].shape == (batch_size, input_size, input_size, 3)
    assert train_dataloader[0][1].shape == (batch_size, input_size, input_size, n_classes)

    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]
    if val_dir and val_label_dir:
        valid_dataset = Dataset(
            val_dir,
            val_label_dir,
            classes=CLASSES,
            augmentation=get_val_augmentation(input_size)
        )
        valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False)
        history = model.fit_generator(
            train_dataloader,
            steps_per_epoch=len(train_dataloader),
            epochs=epochs,
            callbacks=callbacks,
            validation_data=valid_dataloader,
            validation_steps=len(valid_dataloader),
        )
    else:
        history = model.fit_generator(
            train_dataloader,
            steps_per_epoch=len(train_dataloader),
            epochs=epochs,
            callbacks=callbacks,
        )
    if test_dir and test_label_dir:
        test_dataset = Dataset(
            test_dir,
            test_label_dir,
            classes=CLASSES,
            augmentation=get_val_augmentation(input_size),
        )
        test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)
        scores = model.evaluate_generator(test_dataloader)
        print("Loss: {:.5}".format(scores[0]))
        for metric, value in zip(metrics, scores[1:]):
            print("mean {}: {:.5}".format(metric.__name__, value))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',type=str, default='unet', help='model to use', choices=['unet','pspnet'])
    parser.add_argument('--weight',type=str, help='weight file to load, useful for fine-tuning')
    parser.add_argument('--train_img_dir', type=str, required=True)
    parser.add_argument('--train_label_dir', type=str, required=True)
    parser.add_argument('--val_img_dir', type=str, help='validation img dir')
    parser.add_argument('--val_label_dir', type=str, help='validation mask dir')
    parser.add_argument('--test_img_dir', type=str)
    parser.add_argument('--test_label_dir',type=str)
    parser.add_argument('--is_fine_tuning',type=int,default=0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--epochs', type=int,default=100)
    args = parser.parse_args()
    m = main(args.model,args.weight,args.train_img_dir,args.train_label_dir, args.val_img_dir, args.val_label_dir, args.test_img_dir, args.test_label_dir,
             args.is_fine_tuning,args.batch_size, args.epochs)