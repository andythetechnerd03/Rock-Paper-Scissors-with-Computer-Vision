from tensorflow.keras.preprocessing.image import ImageDataGenerator
# create a data generator
def main():
    datagen = ImageDataGenerator(
            samplewise_center=True,  # set each sample mean to 0
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True,
            brightness_range=(0.5,1.5)) # we don't expect Bo to be upside-down so we will not flip vertically

    train_it = datagen.flow_from_directory('train_data/', 
                                        color_mode='rgb', 
                                        class_mode='binary' ,
                                        batch_size=500,
                                        save_to_dir='train_data/scissors',
                                        save_prefix='scissorsaug_',
                                        save_format='jpg',
                                        classes=['scissors'],
                                        follow_links=True,
                                        target_size=(256,256),
                                        keep_aspect_ratio=True)

    '''test_it = datagen.flow_from_directory('test_data/', 
                                        color_mode='rgb', 
                                        class_mode='binary' ,
                                        batch_size=1,
                                        save_to_dir='aug_test/',
                                        save_prefix="test",
                                        follow_links=True,
                                        save_format='jpg')'''
    return train_it

if __name__ == "__main__":
    train = main()
    train.next()

