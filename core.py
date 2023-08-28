import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('/home/ephemera/Project/RedWine/Data/winequality-red.csv')
# quality 값 변환 함수 정의
def transform_quality(value):
    if value <= 3:
        return 0
    elif value <= 6:
        return 1
    elif value <= 9:
        return 2
    elif value == 10:
        return 3
    



# quality 열 변환
data['quality'] = data['quality'].apply(transform_quality)

answer = data.pop('quality')

answer_encoded = to_categorical(answer,num_classes=4)
print(answer)
print(answer_encoded)
ds = tf.data.Dataset.from_tensor_slices((dict(data), answer_encoded))

feature_columns = []
for i in data:
    feature_columns.append(tf.feature_column.numeric_column(i))
# feature_columns.append(tf.feature_column.numeric_column('fixed acidity'))
# feature_columns.append(tf.feature_column.numeric_column('volatile acidity'))
# feature_columns.append(tf.feature_column.numeric_column('citric acidity'))
# feature_columns.append(tf.feature_column.numeric_column('residual sugar'))
# feature_columns.append(tf.feature_column.numeric_column('chlorides'))
# feature_columns.append(tf.feature_column.numeric_column('free sulfur dioxide'))
# feature_columns.append(tf.feature_column.numeric_column('total sulfur dioxide'))
# feature_columns.append(tf.feature_column.numeric_column('density'))
# feature_columns.append(tf.feature_column.numeric_column('pH'))
# feature_columns.append(tf.feature_column.numeric_column('sulphates'))
# feature_columns.append(tf.feature_column.numeric_column('alcohol'))

winemodel = tf.keras.models.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax')
])

validation_split=0.3
num_samples = len(data)
num_val_samples = int(validation_split * num_samples)
num_train_samples = num_samples - num_val_samples

# 학습 데이터셋
train_data = ds.take(num_train_samples)
train_data = train_data.shuffle(buffer_size=num_train_samples).batch(32)

# 검증 데이터셋
val_data = ds.skip(num_train_samples)
val_data = val_data.shuffle(buffer_size=num_val_samples).batch(16)


winemodel.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
ds_batch = ds.batch(32)
for i in range(10):
    print(f"{i}번째 Batch")
    winemodel.fit(train_data,validation_data=val_data,shuffle=True, epochs=30)