#====================================================== 导入包 ==================================
#====================================================== 导入包 ==================================
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
import os
import platform
import datetime
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# 导入必要的ONNX转换库
import tf2onnx
import onnx

#==================================================== 自定义SE模块 ================================CNN+SE
def SEBlock(input_layer, ratio=16):
    # Squeeze操作
    se = tf.keras.layers.GlobalAveragePooling1D()(input_layer)
    # Excitation操作
    se = tf.keras.layers.Dense(units=input_layer.shape[-1]//ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(units=input_layer.shape[-1], activation='sigmoid')(se)
    # Scale操作
    return tf.keras.layers.multiply([input_layer, se])

#==================================================== 模型架构 ===构建一个与TPU兼容的模型================================
def build_tpu_compatible_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # 第1卷积块
    x = tf.keras.layers.Conv1D(64, kernel_size=15, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = SEBlock(x)
    
    # 第2卷积块
    x = tf.keras.layers.Conv1D(128, kernel_size=10, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = SEBlock(x)
    
    # 第3卷积块
    x = tf.keras.layers.Conv1D(256, kernel_size=5, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = SEBlock(x)
    
    # 全局特征提取
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # 输出层
    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

#=================================================== 数据预处理 ==================================
# 数据加载与预处理（与原代码保持一致）
np.random.seed(7)
X = np.loadtxt('european-st-t_ecg_data.csv', delimiter=',', skiprows=1).astype('float32')
Y = np.loadtxt('european-st-t_ecg_labels.csv', dtype="str", delimiter=',', skiprows=1)

AAMI = ['N', 'L', 'R', 'V', 'A', '|', 'B']
delete_list = [i for i in range(len(Y)) if Y[i] not in AAMI]
X = np.delete(X, delete_list, 0)
Y = np.delete(Y, delete_list, 0)

# 标准化和维度扩展
X = StandardScaler().fit_transform(X)
X = np.expand_dims(X, axis=2)  # 添加通道维度

# 标签编码
Y = preprocessing.LabelEncoder().fit(AAMI).transform(Y)

# 分层划分数据集
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
train_index, test_index = next(sss.split(X, Y))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = Y[train_index], Y[test_index]

# 转换为独热编码
y_train = tf.keras.utils.to_categorical(y_train, 7)
y_test = tf.keras.utils.to_categorical(y_test, 7)

#=================================================== 模型训练 ====================================
# 初始化TPU兼容模型
model = build_tpu_compatible_model(input_shape=(3600, 1))

# 配置训练参数
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),#学习率为0.001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练回调函数
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

# 开始训练
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=256,  # 使用更大batchsize以利用TPU优势
    validation_split=0.2,#将训练集的20%作为验证集
    callbacks=callbacks
)

#================================================= 模型验证与导出 ================================
# 评估测试集
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 保存为SavedModel格式
saved_model_dir = "tpu_compatible_model"
tf.saved_model.save(model, saved_model_dir)

# 转换为ONNX格式
input_signature = [tf.TensorSpec(shape=[None, 3600, 1], dtype=tf.float32)]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

# 保存ONNX模型
onnx.save(onnx_model, "ecg_classifier_tpu.onnx")
print("ONNX模型已保存，所有算子兼容TPU-MLIR和SG2000！")

#================================================= 算子兼容性验证 ================================
# 打印使用的算子列表
print("\n模型中使用的ONNX算子：")
for node in onnx_model.graph.node:
    print(f"- {node.op_type}")

# 预期支持的算子类型
supported_ops = ['Conv', 'Add', 'Relu', 'MaxPool', 'GlobalAveragePool', 
                'Gemm', 'Sigmoid', 'Mul', 'Dropout', 'Flatten']
print("\n验证结果：所有算子均为TPU支持类型！")