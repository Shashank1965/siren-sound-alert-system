from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.core.files.storage import FileSystemStorage
import os


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from django.shortcuts import render
from django.conf import settings

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            # Force status to 'waiting' server-side regardless of what was submitted
            user = form.save(commit=False)
            user.status = 'waiting'
            user.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.error(request, 'Registration failed: Email, Mobile, or Login ID may already exist.')
            print("Invalid form:", form.errors)
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid', '').strip()
        pswd = request.POST.get('pswd', '').strip()
        print("Login ID = ", loginid, ' Password = ', pswd)

        if not loginid or not pswd:
            messages.error(request, 'Please enter both Login ID and Password.')
            return render(request, 'UserLogin.html', {})

        try:
            # Step 1: Check if user with this loginid exists at all
            check = UserRegistrationModel.objects.get(loginid=loginid)
        except UserRegistrationModel.DoesNotExist:
            print('No user found with loginid:', loginid)
            messages.error(request, 'Invalid Login ID or Password.')
            return render(request, 'UserLogin.html', {})
        except Exception as e:
            print('Unexpected DB error:', str(e))
            messages.error(request, 'An unexpected error occurred. Please try again.')
            return render(request, 'UserLogin.html', {})

        # Step 2: Check password
        if check.password != pswd:
            print('Password mismatch for loginid:', loginid)
            messages.error(request, 'Invalid Login ID or Password.')
            return render(request, 'UserLogin.html', {})

        # Step 3: Check activation status
        print('Status is = ', check.status)
        if check.status != 'activated':
            messages.error(request, 'Your account has not been activated yet. Please contact the admin.')
            return render(request, 'UserLogin.html', {})

        # Step 4: Set session and redirect
        request.session['id'] = check.id
        request.session['loggeduser'] = check.name
        request.session['loginid'] = loginid
        request.session['email'] = check.email
        print("Login successful for user id:", check.id)
        return render(request, 'users/UserHome.html', {})

    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


from django.shortcuts import render
from django.conf import settings
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv1D, MaxPooling1D, GlobalMaxPool1D, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import backend as K

def training(request):
    # 1. Feature Extraction
    def features_extractor(file_name):
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features

    audio_dataset_path = os.path.join(settings.MEDIA_ROOT, 'sounds')
    extracted_features = []

    for path in os.listdir(audio_dataset_path):
        folder_path = os.path.join(audio_dataset_path, path)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(".wav"):
                    file_name = os.path.join(folder_path, file)
                    data = features_extractor(file_name)
                    extracted_features.append([data, path])

    # 2. Save and Load Features
    pkl_path = os.path.join(settings.MEDIA_ROOT, 'Extracted_Features.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(extracted_features, f)

    with open(pkl_path, 'rb') as f:
        Data = pickle.load(f)

    # 3. Prepare DataFrame
    df = pd.DataFrame(Data, columns=['feature', 'class'])
    X = np.array(df['feature'].tolist())
    Y = np.array(df['class'].tolist())

    labelencoder = LabelEncoder()
    y = to_categorical(labelencoder.fit_transform(Y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)

    X_train_features = X_train.reshape(len(X_train), -1, 1)
    X_test_features = X_test.reshape(len(X_test), -1, 1)

    # 4. Define CNN Model
    def cnn(optimizer="adam", activation="relu", dropout_rate=0.5):
        K.clear_session()
        inputs = Input(shape=(X_train_features.shape[1], X_train_features.shape[2]))
        conv = Conv1D(3, 13, padding='same', activation=activation)(inputs)
        if dropout_rate:
            conv = Dropout(dropout_rate)(conv)
        conv = MaxPooling1D(2)(conv)
        conv = Conv1D(16, 11, padding='same', activation=activation)(conv)
        if dropout_rate:
            conv = Dropout(dropout_rate)(conv)
        conv = MaxPooling1D(2)(conv)
        conv = GlobalMaxPool1D()(conv)
        conv = Dense(16, activation=activation)(conv)
        outputs = Dense(y_test.shape[1], activation='softmax')(conv)
        model = Model(inputs, outputs)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    # 5. Train Model
    model_cnn = cnn(optimizer="adam", activation="relu", dropout_rate=0)
    early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)

    history = model_cnn.fit(
        X_train_features, y_train,
        epochs=40,
        batch_size=64,
        validation_data=(X_test_features, y_test),
        callbacks=[early_stop]
    )

    # 6. Save model
    model_path = os.path.join(settings.MEDIA_ROOT, 'audio_cnn_model.h5')
    model_cnn.save(model_path)

    # 7. Evaluation
    _, acc = model_cnn.evaluate(X_test_features, y_test)

    y_pred = model_cnn.predict(X_test_features)
    conf_mat = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    # Use Agg backend for non-GUI environments
    import matplotlib
    matplotlib.use('Agg')

    # 8. Save Loss Plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(settings.MEDIA_ROOT, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()

    # 9. Save Accuracy Plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    acc_plot_path = os.path.join(settings.MEDIA_ROOT, 'accuracy_plot.png')
    plt.savefig(acc_plot_path)
    plt.close()

    # 10. Save Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labelencoder.classes_, yticklabels=labelencoder.classes_, cbar=False)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_plot_path = os.path.join(settings.MEDIA_ROOT, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.close()

    # 11. Return to Template
    context = {
        'accuracy': round(acc * 100, 2),
        'loss_plot': os.path.join(settings.MEDIA_URL, 'loss_plot.png'),
        'acc_plot': os.path.join(settings.MEDIA_URL, 'accuracy_plot.png'),
        'cm_plot': os.path.join(settings.MEDIA_URL, 'confusion_matrix.png'),
        'model_path': model_path
    }

    return render(request, 'users/train_results.html', context)

from django.shortcuts import render
from django.conf import settings
import os
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import LabelEncoder

def predict_audio(request):
    prediction = None
    predicted_class = None

    if request.method == 'POST' and request.FILES['audio_file']:
        audio_file = request.FILES['audio_file']
        
        # Save uploaded file
        file_path = os.path.join(settings.MEDIA_ROOT, audio_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        # Feature extraction
        def features_extractor(file_name):
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
            mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
            mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
            return mfccs_scaled_features

        features = features_extractor(file_path)
        input_features = np.array(features).reshape(1, -1, 1)

        # Load model
        model_path = os.path.join(settings.MEDIA_ROOT, 'audio_cnn_model.h5')
        model = load_model(model_path)

        # Load label encoder
        with open(os.path.join(settings.MEDIA_ROOT, 'Extracted_Features.pkl'), 'rb') as f:
            data = pickle.load(f)
        df = [row[1] for row in data]
        labelencoder = LabelEncoder()
        labelencoder.fit(df)

        # Predict
        prediction = model.predict(input_features)
        predicted_class = labelencoder.inverse_transform([np.argmax(prediction)])

    return render(request, 'users/predict.html', {
        'prediction': predicted_class[0] if predicted_class is not None else None,
        'audio_url': settings.MEDIA_URL + audio_file.name if predicted_class is not None else None
    })







