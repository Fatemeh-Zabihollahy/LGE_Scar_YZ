'''
Created on Mon Oct 29 16:18:44 2018

Author: Fatemeh Zabihollahy
'''
#%%
import numpy
from PIL import Image
from numpy import *
import math
import sklearn
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture
import scipy
from scipy.ndimage.interpolation import zoom
from skimage.measure import block_reduce
import skimage
from skimage import morphology
from skimage.morphology import erosion
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.merge import concatenate
#from keras import backend as K
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import nibabel as nib
import tensorflow
import glob
from matplotlib import pyplot as plt

path1 = r'C:\Users\Fatemeh\Desktop\LGE Cardiac MRI\LGE Images nii'
LGEs = glob.glob(path1 + "/*")

path2 = r'C:\Users\Fatemeh\Desktop\LGE Cardiac MRI\Myocardial Masks nii'
MYOs = glob.glob(path2 + "/*")

path3 = r'C:\Users\Fatemeh\Desktop\LGE Cardiac MRI\Scar Masks nii'
SCARs = glob.glob(path3 + "/*")

#%%
def data_sample_visualization(data, k, mask):    
    lge_sample = data[k,:]
    lge_sample = lge_sample.reshape(x_unet, y_unet)
    print(numpy.max(lge_sample))
    lge_img = Image.fromarray(lge_sample*255)
    lge_img.show()
    
    mask_sample = mask[k,:]
    mask_sample = mask_sample.reshape(x_unet, y_unet)
    print(numpy.max(mask_sample))
    mask_img = Image.fromarray(mask_sample*255)
    mask_img.show()

#%%
x_unet = 256
y_unet = 256

data_train = numpy.zeros((1,x_unet*y_unet))
mask_train =  numpy.zeros((1,x_unet*y_unet))

for n in range(18): 
    
    data_lge = nib.load(LGEs[n]);
    lge = data_lge.get_data()
    x,y,z = lge.shape
    
    #lge = block_reduce(lge, block_size=(downsample_factor, downsample_factor,1), func=numpy.mean)  
    
    data_myo = nib.load(MYOs[n]);
    myo = data_myo.get_data()
    
    data_scar = nib.load(SCARs[n]);
    scar = data_scar.get_data()
    
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if myo[i,j,k] == 0:
                    lge[i,j,k] = 0
#%  
    lge_norm = numpy.zeros((x,y,z))
    for slice_no in range (x):
        lge_slice = lge[slice_no,:,:]
        for a in range (y):
            for b in range (z):
                if lge_slice[a,b] > 1000:
                    lge_slice[a,b] = numpy.median(lge_slice)
        if (numpy.max(lge_slice != 0)):            
            lge_slice = (lge_slice-lge_slice.min())/(lge_slice.max()-lge_slice.min())
            lge_norm[slice_no,:,:] = lge_slice
   
    data = numpy.zeros((1,x_unet*y_unet))
    mask = numpy.zeros((1,x_unet*y_unet))
    
    x_pad = int(x_unet - y)
    y_pad = int(y_unet - z)
    
    for page in range(0,x):    
        lge_slice = lge_norm[page,:,:]
        myo_slice = myo[page,:,:]
        scar_slice = scar[page,:,:]
        
        if (numpy.max(myo_slice) != 0):
            lge_slice = numpy.pad(lge_slice, ((0, x_pad),(0, y_pad)), 'wrap')
            scar_slice = numpy.pad(scar_slice, ((0, x_pad),(0, y_pad)), 'wrap')
            
            #LGE_slice = scipy.ndimage.filters.median_filter(LGE_slice,3)
            '''
            lge_flipud = numpy.flipud(LGE_slice)
            lge_fliplr = numpy.fliplr(LGE_slice)
            lge_rot = numpy.rot90(LGE_slice)
            myo_flipud = numpy.flipud(myo_slice)
            myo_fliplr = numpy.fliplr(myo_slice)
            myo_rot = numpy.rot90(myo_slice)
            '''
            lge_slice = lge_slice.reshape(1,(x_unet*y_unet))
            scar_slice = scar_slice.reshape(1, (x_unet*y_unet)) 
            '''
            lge_fud_reshape = lge_flipud.reshape(1,(x_unet*y_unet))
            lge_flr_reshape = lge_fliplr.reshape(1,(x_unet*y_unet))
            lge_rot_reshape = lge_rot.reshape(1,(x_unet*y_unet))
            myo_fud_reshape = myo_flipud.reshape(1,(x_unet*y_unet))
            myo_flr_reshape = myo_fliplr.reshape(1,(x_unet*y_unet))
            myo_rot_reshape = myo_rot.reshape(1,(x_unet*y_unet))
            
            data = numpy.vstack((data, LGE_slice, lge_fud_reshape, lge_flr_reshape, lge_rot_reshape ))
            mask = numpy.vstack((mask,myo_slice, myo_fud_reshape, myo_flr_reshape, myo_rot_reshape ))
            ''' 
            data = numpy.vstack((data,lge_slice ))
            mask = numpy.vstack((mask,scar_slice))

    data = numpy.delete(data, (0), axis=0)     
    data_train = numpy.vstack((data_train, data))   
    
    mask = numpy.delete(mask, (0), axis=0)     
    mask_train = numpy.vstack((mask_train, mask)) 
        
data_train = numpy.delete(data_train, (0), axis=0) 
mask_train = numpy.delete(mask_train, (0), axis=0) 
#% reshape training dataset
data_train = data_train.reshape(data_train.shape[0], x_unet, y_unet, 1)
mask_train = mask_train.reshape(mask_train.shape[0], x_unet, y_unet, 1)
#%%
data_sample_visualization(data_train, 70, mask_train)
#%% U-net1 Architecture
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


filter_no = 16

inputs = Input((x_unet, y_unet, 1))

conv1 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv1)
conv1 = BatchNormalization()(conv1)

pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
conv2 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv2)
conv2 = BatchNormalization()(conv2)

pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
conv3 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv3)
conv3 = BatchNormalization()(conv3)

pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
conv4 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = Dropout(0.5)(conv4)

pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
conv5 = Conv2D(filter_no*16, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool4)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(filter_no*16, 3, strides=(1, 1), activation = 'relu', padding = 'same')(conv5)
conv5 = BatchNormalization()(conv5)

up1 = UpSampling2D(size = (2,2))(conv5)
merge1 = concatenate([conv4,up1], axis = 3)
conv6 = Conv2D(filter_no*8, 3, strides=(1, 1), activation = 'relu', padding = 'same')(merge1)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv6)
conv6 = BatchNormalization()(conv6)

up2 = UpSampling2D(size = (2,2))(conv6)
merge2 = concatenate([conv3,up2], axis = 3)
conv7 = Conv2D(filter_no*4, 3, strides=(1, 1), activation = 'relu', padding = 'same')(merge2)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(filter_no*4, 3, strides=(1, 1), activation = 'relu', padding = 'same')(conv7)
conv7 = BatchNormalization()(conv7)

up3 = UpSampling2D(size = (2,2))(conv7)
merge3 = concatenate([conv2,up3], axis = 3)
conv8 = Conv2D(filter_no*2, 3, strides=(1, 1), activation = 'relu', padding = 'same')(merge3)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv8)
conv8 = BatchNormalization()(conv8)

up4 = UpSampling2D(size = (2,2))(conv8)
merge4 = concatenate([conv1,up4], axis = 3)
conv9 = Conv2D(filter_no, 3, strides=(1, 1), activation = 'relu', padding = 'same')(merge4)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(filter_no, 3, strides=(1, 1), activation = 'relu', padding = 'same')(conv9)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(2, 3, strides=(1, 1), activation = 'relu', padding = 'same')(conv9)
conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = Model(input = inputs, output = conv9)

model.compile(optimizer='adadelta', loss=dice_coef_loss, metrics=[dice_coef])
#model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])	
model.summary()
#%% U-net2 Architecture
'''
filter_no = 32
inputs = Input((x_unet, y_unet, 1))
#s = Lambda(lambda x: x / 255) (inputs)

conv1 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv1)
conv1 = BatchNormalization()(conv1)
#conv1 = Dropout(0.5)(conv1)

pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
conv2 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv2)
conv2 = BatchNormalization()(conv2)
#conv2 = Dropout(0.5)(conv2)

pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
conv3 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv3)
conv3 = ZeroPadding2D(padding=(1, 1))(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Dropout(0.5)(conv3)

pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
conv4 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(filter_no*8, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = Dropout(0.5)(conv4)

up1 = UpSampling2D(size = (2,2))(conv4)
merge1 = merge([conv3,up1], mode = 'concat', concat_axis = 3)
conv5 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge1)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(filter_no*4, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv5)
conv5 = BatchNormalization()(conv5)
#conv5 = Dropout(0.5)(conv5)

up2 = UpSampling2D(size = (2,2))(conv5)
up2 = Cropping2D(cropping=((2, 2), (2, 2)))(up2)
merge2 = merge([conv2,up2], mode = 'concat', concat_axis = 3)
conv6 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge2)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(filter_no*2, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv6)
conv6 = BatchNormalization()(conv6)
#conv6 = Dropout(0.5)(conv6)

up3 = UpSampling2D(size = (2,2))(conv6)
merge3 = merge([conv1,up3], mode = 'concat', concat_axis = 3)
conv7 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(merge3)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(filter_no, 3,  strides=(1, 1), activation = 'relu', padding = 'same')(conv7)
conv7= BatchNormalization()(conv7)
#conv7 = Dropout(0.5)(conv7)
conv7 = Conv2D(2, 3, strides=(1, 1), activation = 'relu', padding = 'same')(conv7)
conv7 = Conv2D(1, 1, activation = 'sigmoid')(conv7)

model = Model(input = inputs, output = conv7)

#sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#model.compile(loss='mean_squared_error', optimizer=sgd, metrics = ['accuracy'])

model.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])	
model.summary()
'''

#%% Train Model
fname= "scar_yz4.hdf5"
#earlystopper = EarlyStopping(patience=20, verbose=1)
checkpointer = ModelCheckpoint(fname, verbose=1, save_best_only=True)
results = model.fit(data_train, mask_train, validation_split=0.2, shuffle=True, batch_size=10, epochs=120, callbacks=[checkpointer])		
#%%
# summarize history for accuracy
plt.plot(results.history['dice_coef'])
plt.plot(results.history['val_dice_coef'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%% Save trained model and the network weights
fname= "scar_yz3.hdf5"
model.save(fname, overwrite = True)
#model.save_weights("MYO_seg_unet3_weights.h5")

#%%
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

fname= "scar_yz4.hdf5"
model = load_model(fname, custom_objects={'dice_coef': dice_coef,'dice_coef_loss': dice_coef_loss})

#%%
def Plot_Results(img_test,seg_clean,mask_clean):
    
    img_test = img_test.reshape( x_unet, y_unet)
    img_test = img_test[:y, :z]

    fig, axes = plt.subplots(1, 3, figsize=(8, 8))
    ax = axes.flatten()
    
    ax[0].imshow(img_test.reshape(y,z), cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(seg_clean, [0.5], colors='r')
    ax[0].contour(mask_clean, [0.5], colors='b')
    #ax[0].set_title("Morphological ACWE segmentation", fontsize=12)
    
    ax[1].imshow(seg_clean, cmap="gray")
    ax[2].imshow(mask_clean, cmap="gray")

def model_evaluate(data,mask):
    dsc = []
    acc = []
    prec = []
    rec = []
    # define test_model(data,mask)
    for k in range(len(data)):
        mask_sample = mask[k,:,:,:]
        mask_sample = mask_sample.reshape(x_unet, y_unet)
        #if (numpy.max(mask_sample) == 1):
        img_test = data[k,:, :, :]
        img_test = img_test.reshape(1, x_unet, y_unet, 1)
        img_pred = model.predict(img_test, batch_size=1, verbose=1)
        img_pred = img_pred.reshape(x_unet, y_unet)
        img_pred  = (img_pred  > 0.5).astype(numpy.uint8)
        
        seg_clean = numpy.array(img_pred, bool)
        seg_clean = morphology.remove_small_objects(seg_clean,100) 
        seg_clean = seg_clean*1 
        
        seg_clean = scipy.ndimage.morphology.binary_dilation(seg_clean, iterations=2)
        seg_clean = scipy.ndimage.morphology.binary_erosion(seg_clean)
        #seg_clean = scipy.ndimage.morphology.binary_fill_holes(seg_clean) 
        seg_clean = seg_clean*1    
        seg_clean = seg_clean[:y, :z]
        
        mask_clean = scipy.ndimage.morphology.binary_dilation(mask_sample,  iterations=2)
        mask_clean = scipy.ndimage.morphology.binary_erosion(mask_clean)
        #mask_clean = scipy.ndimage.morphology.binary_fill_holes(mask_clean) 
        mask_clean = mask_clean*1
        mask_clean = mask_clean[:y, :z]
        
        #seg_clean = morphological_chan_vese(img_pred, 5, init_level_set=seg_clean, smoothing=2)
        #Plot_Results(img_test,seg_clean,mask_clean)
                
        y_true = numpy.reshape(mask_clean, (y*z,1))
        y_pred = numpy.reshape(seg_clean, (y*z,1))
        
        dsc = numpy.append(dsc,f1_score(y_true, y_pred, average='macro'))
        acc = numpy.append(acc,accuracy_score(y_true, y_pred))
        prec = numpy.append(prec,precision_score(y_true, y_pred, average='macro'))
        rec = numpy.append(rec,recall_score(y_true, y_pred, average='macro'))
    
    dsc = round(numpy.median(dsc)*100,2)
    acc = round(numpy.median(acc)*100,2)
    prec = round(numpy.median(prec)*100,2)
    rec = round(numpy.median(rec)*100,2)
    return(dsc,acc,prec,rec)
#%% Create test dataset including test images and their corresponding masks
x_unet = 256
y_unet = 256
dice_index =[]
accuracy = []
precision = []
recall = []

for n in range(18,34): 
    
    data_lge = nib.load(LGEs[n])
    lge = data_lge.get_data()
    x,y,z = lge.shape
    
    #lge = block_reduce(lge, block_size=(downsample_factor, downsample_factor,1), func=numpy.mean)  
    
    data_myo = nib.load(MYOs[n]);
    myo = data_myo.get_data()
    
    data_scar = nib.load(SCARs[n]);
    scar = data_scar.get_data()
    
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if myo[i,j,k] == 0:
                    lge[i,j,k] = 0
 
    lge_norm = numpy.zeros((x,y,z))
    for slice_no in range (x):
        lge_slice = lge[slice_no,:,:]
        for a in range (y):
            for b in range (z):
                if lge_slice[a,b] > 1000:
                    lge_slice[a,b] = numpy.median(lge_slice)
        if (numpy.max(lge_slice != 0)):            
            lge_slice = (lge_slice-lge_slice.min())/(lge_slice.max()-lge_slice.min())
            lge_norm[slice_no,:,:] = lge_slice
   
    data = numpy.zeros((1,x_unet*y_unet))
    mask = numpy.zeros((1,x_unet*y_unet))
    
    x_pad = int(x_unet - y)
    y_pad = int(y_unet - z)
    
    for page in range(0,x):    
        lge_slice = lge_norm[page,:,:]
        myo_slice = myo[page,:,:]
        scar_slice = scar[page,:,:]
        
        if (numpy.max(myo_slice) != 0):   
            lge_slice = numpy.pad(lge_slice, ((0, x_pad),(0, y_pad)), 'wrap')
            scar_slice = numpy.pad(scar_slice, ((0, x_pad),(0, y_pad)), 'wrap')
            
            lge_slice = lge_slice.reshape(1,(x_unet*y_unet))
            scar_slice = scar_slice.reshape(1, (x_unet*y_unet)) 
            
            data = numpy.vstack((data,lge_slice ))
            mask = numpy.vstack((mask,scar_slice))

    data = numpy.delete(data, (0), axis=0)       
    
    mask = numpy.delete(mask, (0), axis=0)     
    data = data.reshape(data.shape[0], x_unet, y_unet, 1)
    mask = mask.reshape(mask.shape[0], x_unet, y_unet, 1)
    
    p1,p2,p3,p4 = model_evaluate(data,mask)
    dice_index = numpy.append(dice_index,p1)
    accuracy = numpy.append(accuracy,p2)
    precision = numpy.append(precision,p3)
    recall = numpy.append(recall,p4)

#data_sample_visualization(data_test, 15, mask_test)

#%      
print('Mean Values:')    
print('DI is :', round(numpy.mean(dice_index),2) , '+', round(numpy.std(dice_index),2))
print('Acc. is :', round(numpy.mean(accuracy),2), '+', round(numpy.std(accuracy),2))
print('Precision is :', round(numpy.mean(precision),2), '+', round(numpy.std(precision),2))
print('Recall is :', round(numpy.mean(recall),2), '+', round(numpy.std(recall),2))

print('Median Values:') 
print('DI is :', round(numpy.median(dice_index),2) , '+', round(numpy.std(dice_index),2))
print('Acc. is :', round(numpy.median(accuracy),2), '+', round(numpy.std(accuracy),2))
print('Precision is :', round(numpy.median(precision),2), '+', round(numpy.std(precision),2))
print('Recall is :', round(numpy.median(recall),2), '+', round(numpy.std(recall),2))
#%%
'''
k = 18
img_test = data[k,:, :, :]
img_test = img_test.reshape(1, x_unet, y_unet, 1)
img_pred = model.predict(img_test, batch_size=1, verbose=1)
img_pred = img_pred.reshape(x_unet, y_unet)
img_pred  = (img_pred  > 0.5).astype(numpy.uint8)

img_test = img_test.reshape( x_unet, y_unet)
img_test = img_test[:x, :y]

seg_clean = numpy.array(img_pred, bool)
seg_clean = morphology.remove_small_objects(seg_clean,20) 
seg_clean = seg_clean*1 

seg_clean = scipy.ndimage.morphology.binary_dilation(seg_clean,  iterations=4)
seg_clean = scipy.ndimage.morphology.binary_erosion(seg_clean)
#seg_clean = scipy.ndimage.morphology.binary_fill_holes(seg_clean) 
#snake = morphological_chan_vese(img_pred, 10, init_level_set=seg_clean, smoothing=4)
seg_clean = seg_clean[:x, :y]
#alaki = Image.fromarray(seg_clean *255)
#alaki.show()

mask_sample = mask[k,:,:,:]
mask_sample = mask_sample.reshape(x_unet, y_unet)

mask_clean = scipy.ndimage.morphology.binary_dilation(mask_sample,  iterations=2)
mask_clean = scipy.ndimage.morphology.binary_erosion(mask_clean)
#mask_clean = scipy.ndimage.morphology.binary_fill_holes(mask_clean) 
mask_clean = mask_clean*1
mask_clean = mask_clean[:x, :y]
#q = savgol_filter(mask_sample, 3, 2)

#mask_img = Image.fromarray(mask_sample*255)
#mask_img.show()

fig, axes = plt.subplots(1, 3, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(img_test.reshape(x,y), cmap="gray")
ax[0].set_axis_off()
ax[0].contour(seg_clean, [0.5], colors='r')
ax[0].contour(mask_clean, [0.5], colors='b')
#ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

ax[1].imshow(seg_clean, cmap="gray")
ax[2].imshow(mask_clean, cmap="gray")

#%
y_true = numpy.reshape(mask_clean, (x*y,1))
y_pred = numpy.reshape(seg_clean, (x*y,1))
print(f1_score(y_true, y_pred, average='macro'))
'''