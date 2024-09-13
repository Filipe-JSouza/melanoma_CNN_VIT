import os
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121 as convnet
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dirname1=('/tcc/complete_dataset/melanoma')
dirname2=('/tcc/complete_dataset/naevus')
extension=('.jpg')
dir2save1=('/tcc/complete_dataset/embedding_melanoma')
dir2save2=('/tcc/complete_dataset/embedding_naevus')

if not os.path.exists(dir2save1):
    os.makedirs(dir2save1)

if not os.path.exists(dir2save2):
    os.makedirs(dir2save2)


#dirname = dirname1
dirname = dirname2

model = convnet(include_top=False, weights='imagenet', pooling='avg')

#Organizando e numerando as imagens do diretorio
def inumerate_dir(dirname, extension):
 img_names= sorted(os.listdir(dirname))
 num_imgs= len(img_names)
 renamed_file=[]
 for i,img_name in enumerate(img_names, start=1):
  img_rename= '{:03d}{:s}'.format(i, extension)
  img_path= os.path.join(dirname,img_name)
  img_rename_path= os.path.join(dirname,img_rename)
  os.rename(img_path, img_rename_path)
  renamed_file.append(img_rename)

 return renamed_file

#Criando as features(vetor caracteristica)
def create_feature(dir2save, renamed_file):
 
 for num, indv_img in enumerate(renamed_file, start=1):
  img = image.load_img(os.path.join(dirname,indv_img), target_size=(224, 224))
  x = image.img_to_array(img)
  x = tf.expand_dims(x, axis=0)
  x = preprocess_input(x)
  features = model.predict(x)

 # salvando a imagem
  if num == 1:
      mat = features
  else:
      mat = np.concatenate((mat, features), axis=0)
  
  path_dir = os.path.join(dir2save, 'features.mat')
  data_save = {'features' : mat}
  sio.savemat(path_dir, data_save)
  
 

#sys.exit()
#renamed_file= inumerate_dir(dirname1, extension)
#create_feature(dir2save1, renamed_file)

#renamed_file= inumerate_dir(dirname2, extension)
#create_feature(dir2save2, renamed_file)

#features_melanoma=sio.loadmat(dir2save1 + '/features.mat')
#features_melanoma=features_melanoma['features']
#features_naevus=sio.loadmat(dir2save2 + '/features.mat')
#features_naevus=features_naevus['features']
