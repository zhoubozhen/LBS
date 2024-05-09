from keras import layers



def VGG(img_input):
    # Block 1
    # 256,256,1 -> 256,256,64 -> 256,256,64
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    feat1 = x  # 256,256,64
    x = layers.BatchNormalization()(x, training = True)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)  # 256,256,64 -> 128,128,64

    # Block 2
    # 128,128,64 -> 128,128,128 -> 128,128,128
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)  # 128,128,64 -> 128,128,128
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)  # 128,128,128 -> 128,128,128
    feat2 = x  # 128,128,128
    x = layers.BatchNormalization()(x, training=True)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)  # 128,128,128 -> 64,64,128

    # Block 3
    # 64,64,128 -> 64,64,256 -> 64,64,256
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)  # 64,64,128 -> 64,64,256
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)  # 64,64,256 -> 64,64,256

    feat3 = x  # 64,64,256
    x = layers.BatchNormalization()(x, training=True)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)  # 64,64,256 -> 32,32,256

    # Block 4
    # 32,32,256 -> 32,32,512 -> 32,32,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)  # 32,32,256 -> 32,32,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)  # 32,32,512 -> 32,32,512

    feat4 = x  # 32,32,512
    x = layers.BatchNormalization()(x, training=True)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)  # 32,32,512 -> 16,16,512

    # Block 5
    # 16,16,512 -> 16,16,512 -> 16,16,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)  # 16,16,512 -> 16,16,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)  # 16,16,512 -> 16,16,512

    feat5 = x  # 16,16,512
    return feat1, feat2, feat3, feat4, feat5
