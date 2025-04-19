from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 1. 数据生成器
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 划分20%数据作为验证集
)

train_generator = train_datagen.flow_from_directory(
    '/content/data/chest_xray/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'  # 训练集
)

val_generator = train_datagen.flow_from_directory(
    '/content/data/chest_xray/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # 验证集
)

# 2. 构建模型
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # 二分类输出
])

# 3. 编译模型
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. 训练模型（train-model的核心部分）
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # 自动计算每轮步数
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator))
