import pandas as pd

# from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.utils import to_categorical
import tensorflow as tf

class Classifier:
    def __init__(self):
        # Setting props
        self.sc = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.classifier = Sequential()

        self.data = self.read_data()
        self.model = self.load_model()

    def load_model(self):
        X = self.data.drop(['Learner'], axis=1)
        y = self.data['Learner']
        y = self.label_encoder.fit_transform(y)
        y = pd.DataFrame(y, columns=['Learner'])
        X_train, _X, y_train, _y= train_test_split(X, y, test_size=0.33, random_state=0)

        # Setting train props
        self.X_train = self.sc.fit_transform(X_train)
        self.y_train_encoded = to_categorical(y_train)

        model = None
        if not tf.io.gfile.exists('classifier/model/model.keras'):
            # Train and save model
            model = self.train_model()
        else:
            # Load saved model and weights
            model = tf.keras.models.load_model('classifier/model/model.keras')
            model.load_weights('classifier/model/model.weights.h5')
        return model

    def read_data(self):
        pd.set_option('future.no_silent_downcasting', True)
        data = pd.read_csv('classifier/data/data.csv')
        data = data.replace(to_replace={'Gender': {'Female': 1,'Male':0}})
        return data

    def train_model(self):
        self.classifier.add(Dense(units=64, kernel_initializer='uniform', activation='relu', input_dim=17))
        self.classifier.add(Dropout(0.2))
        self.classifier.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
        self.classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
        self.classifier.add(Dense(units=3, kernel_initializer='uniform', activation='softmax'))

        self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.classifier.fit(self.X_train, self.y_train_encoded, batch_size=10, epochs=100, verbose=0, validation_split=0.2)

        # Checkpoint
        self.classifier.save_weights('classifier/model/model.weights.h5')
        self.classifier.save('classifier/model/model.keras')

    # Learner type: Visual(V), Auditive(A), Kinesthetic(K)
    def classify(self, input):
        # Predict
        input_df = pd.DataFrame([input], columns=self.data.drop('Learner', axis=1).columns)
        input_df = self.sc.transform(input_df)

        # Get the result
        result = self.model.predict(input_df)
        result = result.argmax(axis=1)

        # Get the prediction
        prediction = ''
        if result == 0:
            prediction = 'Auditive'
        elif result == 1:
            prediction = 'Kinesthetic'
        elif result == 2:
            prediction = 'Visual'
        return prediction

classifier = Classifier()
# ans = classifier.classify(
    # format: [gender, age, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]
    # [0,21,2,3,3,4,4,4,5,5,5,5,5,5,5,3,3] # => Visual - OK
    # [0,16,4,5,2,2,4,5,5,4,3,4,3,2,1,4,3] # => Visual - OK
    # [1,21,4,2,5,2,2,4,5,2,5,4,4,2,4,4,4] # => Visual - Error
    # [1,21,5,5,5,5,4,4,4,4,4,4,3,4,5,5,5] # => Auditive - OK
    # [1,17,2,5,5,3,5,2,3,1,2,1,3,3,5,2,2] # => Auditive - OK
    # [1,18,3,5,3,3,4,2,4,4,3,3,4,2,2,3,4] # => Auditive - OK
    # [0,21,3,3,4,3,4,3,3,3,3,4,3,4,4,3,3] # => Kinesthetic - OK
    # [0,19,4,3,5,3,4,3,3,3,4,3,5,3,4,5,3] # => Kinesthetic - OK
    # [1,17,4,4,2,3,2,2,3,2,4,2,4,2,5,2,4] # => Kinesthetic - OK
#)