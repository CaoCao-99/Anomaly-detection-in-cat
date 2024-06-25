import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from collections import Counter
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model ,models, layers, optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping,LambdaCallback
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from datetime import datetime
import pickle
from contextlib import redirect_stdout


actions = np.array(['normal', 'abnormal'])
LABEL = ['NORMAL', 'ABNORMAL']
label_map = {label:num for num, label in enumerate(actions)}
print(label_map)

# Initialize lists to store sequences and labels
sequences, labels = [], []
#하이퍼 파라미터 값 설정    
epochs = 200
batch = 40
lr = 0.001
sequence_length = 40 # 작게 설정
num_features = 30
num_classes = 2
pass_frames = 2 # 넘기는 개수(4개에 한번 씩)
latent_vector = 1

def write_file(file_path,file_content):
    # 파일 쓰기
    with open(file_path, 'a') as file:
        file.write(file_content)


def Store_data(x_data,y_data,path):
    flatten_x_train = x_data.reshape(x_data.shape[0],-1)
    # 주어진 데이터
    data_train = pd.DataFrame({'Features': flatten_x_train.tolist(), 'Labels': y_data.tolist()})
    # CSV 파일로 저장
    data_train.to_csv(path, index=False)



def save_data(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data(filename):
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data



# 모델 저장 폴더 생성

folder_path = '/home/sm32289/Cat_Pose/AutoEncoder/Model'

# 폴더 내 숫자로 구성된 서브폴더 찾기
subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) and f.isdigit()]

# 숫자로 구성된 서브폴더가 없으면 1로 시작하는 폴더 생성
if not subfolders:
    new_folder_name = '1'
else:
    # 가장 큰 숫자 찾기
    max_number = max(map(int, subfolders))
    # 다음 폴더의 숫자 계산
    new_folder_number = max_number + 1
    new_folder_name = str(new_folder_number)

# 새로운 폴더 경로 생성
save_model_dir = os.path.join(folder_path, new_folder_name)

# 새로운 폴더 생성
os.makedirs(save_model_dir,exist_ok = True)
txtfile_path = os.path.join(save_model_dir, 'info.txt')


write_file(txtfile_path, f'epochs : {epochs}\nbatch : {batch}\nsequence_length : {sequence_length}\npass_frames = {pass_frames}\nlatent_Vector_size = {latent_vector}')

# Limping
cat_normal_path2 =  "/home/sm32289/Cat_Pose/KeypointDetection_Yolo/Test_video/normal_cat_full"
cat_normal_path = "/home/sm32289/Cat_Pose/KeypointDetection_Yolo/predict/Normal/keypoints"
keypoints_files = [f for f in os.listdir(cat_normal_path)]

for keypoints in keypoints_files:
    cnt = 0
    # original_video_path = os.path.join(cat_normal_path2, keypoints)
    # now_frame = cv2.VideoCapture(original_video_path).read
    keypoints_path = os.path.join(cat_normal_path, keypoints)
    keypoint_file = [f for f in os.listdir(keypoints_path)]
    window = []
    for i in range(len(keypoint_file)):
        key = os.path.join(keypoints_path, f'{i}.npy')
        if os.path.exists(key):
            k = np.load(key)
            if len(k) == num_features:
                if 0 in k:
                    cnt = 0
                    window = []
                else:
                    window.append(k[:num_features])
                    cnt += 1
        if(cnt == sequence_length):
            new_list = [window[i] for i in range(0, len(window), pass_frames)]
            sequences.append(new_list)
            window = []
            labels.append(label_map['normal'])
            cnt = 0
# 정상 행동 데이터 개수
# print(labels[labels == 0])///
write_file(txtfile_path,f'Normal data : {labels.count(0)}\n')

# Throwing-up
cat_throwingup_path = "/home/sm32289/Cat_Pose/KeypointDetection_Yolo/predict/throwing-up-refined/keypoints"
keypoints_files = [f for f in os.listdir(cat_throwingup_path)]

for keypoints in keypoints_files:
    cnt = 0
    keypoints_path = os.path.join(cat_throwingup_path, keypoints)
    keypoint_file = [f for f in os.listdir(keypoints_path)]
    window = []
    for i in range(len(keypoint_file)):
        key = os.path.join(keypoints_path, f'{i}.npy')
        if os.path.exists(key):
            k = np.load(key)
            if len(k) == num_features:
                if 0 in k:
                    cnt = 0
                    window = []
                else:
                    window.append(k[:num_features])
                    cnt += 1
        if(cnt == sequence_length):
            new_list = [window[i] for i in range(0, len(window), pass_frames)]
            sequences.append(new_list)
            window = []
            labels.append(label_map['abnormal'])
            cnt = 0
    
    
# Sneezing
cat_sneezing_path = "/home/sm32289/Cat_Pose/KeypointDetection_Yolo/predict/Cat_Sneezing/keypoints"
keypoints_files = [f for f in os.listdir(cat_sneezing_path)]
for keypoints in keypoints_files:
    cnt = 0
    keypoints_path = os.path.join(cat_sneezing_path, keypoints)
    keypoint_file = [f for f in os.listdir(keypoints_path)]
    window = []
    for i in range(len(keypoint_file)):
        key = os.path.join(keypoints_path, f'{i}.npy')
        if os.path.exists(key):
            k = np.load(key)
            if len(k) == num_features:
                if 0 in k:
                    cnt = 0
                    window = []
                else:
                    window.append(k[:num_features])
                    cnt += 1
        if(cnt == sequence_length):
            new_list = [window[i] for i in range(0, len(window), pass_frames)]
            sequences.append(new_list)
            window = []
            labels.append(label_map['abnormal'])
            cnt = 0
    

# Limping
cat_limping_path = "/home/sm32289/Cat_Pose/KeypointDetection_Yolo/predict/cat_limping_refined/keypoints"
keypoints_files = [f for f in os.listdir(cat_limping_path)]
for keypoints in keypoints_files:
    cnt=0
    keypoints_path = os.path.join(cat_limping_path, keypoints)
    keypoint_file = [f for f in os.listdir(keypoints_path)]
    window = []
    for i in range(len(keypoint_file)):
        key = os.path.join(keypoints_path, f'{i}.npy')
        if os.path.exists(key):
            k = np.load(key)
            if len(k) == num_features:
                if 0 in k:
                    cnt = 0
                    window = []
                else:
                    window.append(k[:num_features])
                    cnt += 1
        if(cnt == sequence_length):
            new_list = [window[i] for i in range(0, len(window), pass_frames)]
            sequences.append(new_list)
            window = []
            labels.append(label_map['abnormal'])
            cnt = 0

write_file(txtfile_path,f'AbNormal data : {labels.count(1)}\n')
# Convert labels to numpy array
write_file(txtfile_path,f'np.array(sequence).shape : {np.array(sequences).shape}\n')
labels = np.array(labels)
X = np.array(sequences)
y = labels
#y = to_categorical(labels).astype(int)
# Split into train, valid, and test 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
storepath = os.path.join(save_model_dir,'Data')
#Store Data
os.makedirs(storepath,exist_ok = True)

# print(os.path.join(storepath,'train.pkl'))
save_data(os.path.join(storepath,'trainx.pkl'), x_train)
save_data(os.path.join(storepath,'trainy.pkl'), y_train)
save_data(os.path.join(storepath,'validx.pkl'), x_valid)
save_data(os.path.join(storepath,'validy.pkl'), y_valid)
save_data(os.path.join(storepath,'testx.pkl'), x_test)
save_data(os.path.join(storepath,'testy.pkl'), y_test)


# For training the autoencoder, split 0 / 1
x_train_y0 = x_train[y_train == 0]#5567
x_train_y1 = x_train[y_train == 1]#634


x_valid_y0 = x_valid[y_valid == 0]#1384
x_valid_y1 = x_valid[y_valid == 1]#167
write_file(txtfile_path,f'x_train_y0 : {len(x_train_y0)}\n')
write_file(txtfile_path,f'x_train_y1 : {len(x_train_y1)}\n')
write_file(txtfile_path,f'x_valid_y0 : {len(x_valid_y0)}\n')
write_file(txtfile_path,f'x_valid_y1 : {len(x_valid_y1)}\n')



log_dir = os.path.join(save_model_dir,"logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Early Stopping 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=30)



# 모델 컴파일 및 학습 시 Early Stopping 콜백 추가

# ModelCheckpoint 콜백 설정
# checkpoint_filepath = '/home/sm32289/Cat_Pose/AutoEncoder/checkpoint/weights_epoch_{epoch:02d}.h5'
# model_checkpoint = ModelCheckpoint(checkpoint_filepath, save_best_only = True )#save_freq=save_frequency
# Callback to save the model when loss exceeds a certain threshold
save_on_high_loss = ModelCheckpoint('/home/sm32289/Cat_Pose/AutoEncoder/checkpoint/model_on_high_loss.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='max', save_freq='epoch', verbose=1)

autoencoder = models.Sequential()
# Encoder
autoencoder.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(sequence_length//pass_frames, num_features)))
autoencoder.add(tf.keras.layers.Dense(64, activation='relu'))
autoencoder.add(tf.keras.layers.Dense(32, activation='relu'))
autoencoder.add(tf.keras.layers.Dense(16, activation='relu'))
autoencoder.add(tf.keras.layers.Dense(latent_vector, activation='relu')) #8개로 줄이긴
# Decoder
autoencoder.add(tf.keras.layers.Dense(16, activation='relu'))
autoencoder.add(tf.keras.layers.Dense(32, activation='relu'))
autoencoder.add(tf.keras.layers.Dense(64, activation='relu'))
autoencoder.add(tf.keras.layers.Dense(128, activation='relu'))
autoencoder.add(tf.keras.layers.Dense(num_features, activation="relu"))

# Write the model summary to the text file
with open(txtfile_path, 'a') as file:
    with redirect_stdout(file):
        autoencoder.summary()
# lstm_ae.summary()


# lstm_ae.summary()

# compile
autoencoder.compile(loss='mse', optimizer=optimizers.Adam(lr),metrics=["acc"])
# print(x_train_y0)
# print(x_valid_y0)
# fit
history = autoencoder.fit(x_train_y0, x_train_y0,
                     epochs=epochs, batch_size=batch,
                     validation_data=(x_valid_y0, x_valid_y0), callbacks=[early_stopping])

history_save_dir = os.path.join(save_model_dir,'history.txt')


autoencoder.save(os.path.join(save_model_dir,'model.h5'))
autoencoder.save(os.path.join(save_model_dir,'model'))

with open(history_save_dir, 'a') as file:
    file.write(str(history.history))


plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss1')
#plt.axis([0, epochs, -1, 1])  # [xmin, xmax, ymin, ymax]
plt.legend()
plt.xlabel('Epoch'); plt.ylabel('loss')
plt.savefig(os.path.join(save_model_dir,'loss.png'))

# Get MSE
def get_mse(data, model):
    mse = []
    data_prediction = model.predict(data)
    for i in range(data.shape[0]):
        mse.append(np.mean(np.power(((data[i])) - (data_prediction[i]), 2)))
    return mse


mse = get_mse(x_valid, autoencoder)
# print(np.array(mse).shape)
# print(len(mse))
# print(len(y_valid))
# print(mse)
#print(mse)
# 예제에서는 y_valid의 각 행의 평균값을 사용하여 이진 레이블을 생성합니다.
error_df = pd.DataFrame({'Reconstruction_error':mse, 
                         'True_class':y_valid})
precision_rt, recall_rt, threshold_rt = metrics.precision_recall_curve(error_df['True_class'], error_df['Reconstruction_error'])
# print(precision_rt)
# print(recall_rt)
# print(threshold_rt)
plt.figure(figsize=(8, 5))
plt.plot(threshold_rt, precision_rt[1:], label='Precision')
plt.plot(threshold_rt, recall_rt[1:], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.savefig(os.path.join(save_model_dir,'valid_data.png'))

# best position of threshold
index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precision_rt, recall_rt)) if p==r][0]
threshold_fixed = threshold_rt[index_cnt]
write_file(txtfile_path, f'precision : {precision_rt[index_cnt]}, recall : {recall_rt[index_cnt]}, threshold : {threshold_fixed}')
# fixed Threshold


test_x_predictions = autoencoder.predict(x_test)
mse = get_mse(x_test, autoencoder)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': y_test})

groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Break" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.savefig(os.path.join(save_model_dir,'test_data.png'))

LABELS = ['NORMAL', 'ABNORMAL']
# classification by threshold
pred_y = [1 if e > threshold_fixed else 0 for e in error_df['Reconstruction_error'].values]

conf_matrix = metrics.confusion_matrix(error_df['True_class'], pred_y)
plt.figure(figsize=(7, 7))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class'); plt.ylabel('True Class')
plt.savefig(os.path.join(save_model_dir,'test_data_heatmap.png'))


false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(error_df['True_class'], error_df['Reconstruction_error'])
roc_auc = metrics.auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate'); plt.xlabel('False Positive Rate')
plt.savefig(os.path.join(save_model_dir,'ROC.png'))

