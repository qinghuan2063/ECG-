import numpy as np
import pandas as pd
import os

# 创建保存目录（如果不存在）
save_dir = "save"
os.makedirs(save_dir, exist_ok=True)

# ----------------- 读取数据 -----------------
# 读取ECG数据文件（假设每行一个样本，每个样本3600个时间点）
data_path = "european-st-t_ecg_data.csv"
data = pd.read_csv(data_path, header=None).values.astype(np.float32)  # 强制转为浮点型

# 读取标签文件（处理可能的非数值标签）
labels_path = "european-st-t_ecg_labels.csv"
labels = pd.read_csv(labels_path, header=None).squeeze()

# 如果标签是字符串，转换为数值编码（例如 "normal"→0, "abnormal"→1）
if labels.dtype == object:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    print("标签编码映射：", dict(zip(le.classes_, le.transform(le.classes_))))

labels = labels.astype(np.int32)  # 统一转为整型

# 验证数据一致性
assert len(data) == len(labels), "数据与标签样本数不匹配！"

# ----------------- 保存为NPZ -----------------
output_path = os.path.join(save_dir, "ecg_dataset.npz")
np.savez_compressed(
    output_path,
    ecg_signals=data,
    ecg_labels=labels
)

# ----------------- 验证加载 -----------------
def verify_npz(npz_path):
    """安全加载并验证NPZ文件"""
    try:
        loaded = np.load(npz_path, allow_pickle=True)  # 关键修复：启用allow_pickle
        print("\n验证结果：")
        print(f"数据形状：{loaded['ecg_signals'].shape}")
        print(f"标签形状：{loaded['ecg_labels'].shape}")
        print("数据类型：")
        print(f"信号：{loaded['ecg_signals'].dtype}")
        print(f"标签：{loaded['ecg_labels'].dtype}")
        print("首个标签值：", loaded['ecg_labels'][0])
    except Exception as e:
        print("加载失败：", str(e))

verify_npz(output_path)