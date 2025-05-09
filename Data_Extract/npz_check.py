import numpy as np

data = np.load("./Data_Extract/pose_tensor_npz/Y/Training/00001_H_A_SY_C1/00001_H_A_SY_C1.npz")
print(data.files)  # ['pose', 'label']

print(data['pose'].shape)  # 예: (600, 17, 3)
print(data['label'].shape)  # 예: (600,)

print(data['label'])