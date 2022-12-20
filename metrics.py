from tensorflow.keras import backend as K

def f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def loadModelTest():
    dir_loaded ="TrainedModels/resnet50-71.h5"
    loaded_model = tf.keras.models.load_model(dir_loaded, custom_objects={'f1':f1})
    # MobileNet
    dir_loaded2 ="TrainedModels/resnet50-71.h5"
    loaded_model2 = tf.keras.models.load_model(dir_loaded, custom_objects={'f1':f1})
    if loaded_model2 is None:
        return "Failed"
    return "Success"

def predictTest(image):
    img = cv2.imread('testDir/Prediction/1.2.826.0.1.3680043.8.498.78821712082546284288318241228027008291.png')
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[224, 224, 3])
    # images_list = []
    images_list = []
    images_list.append(np.array(img))
    x = np.asarray(images_list)
    loaded_model.predict(x)