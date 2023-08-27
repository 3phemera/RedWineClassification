import pandas as pd
import tensorflow as tf

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

ds = tf.data.Dataset.from_tensor_slices((dict(data), answer))

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
