from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load trained image model and categories
model_path = r'E:/Graduation project/Project/flutter_application_1/lib/svm_model_with_vgg16_features2.pkl'
model = joblib.load(model_path)
categories = {
    0: "Acne",
    1: "Dermatitis",
    2: "Eczema",
    3: "Impetigo",
    4: "Psoriasis",
    5: "Scabies",
    6: "Tinea"
}

# Load VGG16 for feature extraction
vgg16_model = VGG16(weights='imagenet', include_top=False)

def extract_features(image_path):
    """Extract features from image using VGG16"""
    try:
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = vgg16_model.predict(img_array)
        return features.flatten()
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

def preprocess_image(image_path):
    """Prepare image for classification"""
    feature_vector = extract_features(image_path)
    feature_vector = feature_vector.reshape(1, -1)
    return feature_vector

def predict_category(image_path, model, categories):
    """Predict disease category from image"""
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)[0]
    prediction = int(prediction)
    if prediction not in categories:
        raise ValueError(f"Predicted class {prediction} is not in the categories dictionary.")
    return categories[prediction]

@app.route('/api/image', methods=['POST'])
def predict_image():
    """Handle image classification"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400

    img_file = request.files['image']
    if img_file.filename == '':
        return jsonify({'error': 'No selected image.'}), 400

    try:
        img_path = 'uploaded_image.jpg'
        img_file.save(img_path)

        try:
            img = Image.open(img_path)
            img.verify()
        except:
            os.remove(img_path)
            return jsonify({'error': 'Uploaded file is not a valid image.'}), 400

        label = predict_category(img_path, model, categories)
        os.remove(img_path)
        return jsonify({'response': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load and process text data
file_path = 'E:/Graduation project/Project/flutter_application_1/lib/augmented_datasetnew.csv'
data = pd.read_csv(file_path)
data['Text'] = data['Text'].astype(str).str.lower()
data['Text'] = data['Text'].str.replace(r'[^\w\s]', '', regex=True)

label_encoder = LabelEncoder()
data['Disease label'] = label_encoder.fit_transform(data['Disease name'])

X = data['Text']
y = data['Disease label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_tfidf, y_train)

svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgboost_model.fit(X_train_tfidf, y_train)

voting_model = VotingClassifier(
    estimators=[('Logistic Regression', logistic_model),
                ('SVM', svm_model),
                ('XGBoost', xgboost_model)],
    voting='soft'
)
voting_model.fit(X_train_tfidf, y_train)

joblib.dump(voting_model, 'voting_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

voting_model = joblib.load('voting_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/api/chat', methods=['POST'])
def predict_text():
    data = request.get_json()
    user_message = data.get('message', '').strip().lower()

    if not user_message:
        return jsonify({'response': "Please describe your skin symptoms so I can assist you."}), 400

    greetings = ['hi', 'hello', 'hey', 'مرحبا', 'السلام عليكم']
    if user_message in greetings:
        return jsonify({'response': "Welcome! I'm here to help with your skin issues. Please tell me about your symptoms."})

    try:
        # Check text clarity
        words = user_message.split()
        if len(words) < 3:
            return jsonify({
                'response': "Your description is a bit short. Please try to include more details like the affected area, duration, itching, etc. You can also upload an image for better help."
            })

        input_vector = tfidf_vectorizer.transform([user_message])
        probabilities = voting_model.predict_proba(input_vector)[0]
        top_indices = np.argsort(probabilities)[-2:][::-1]
        top_prediction = label_encoder.inverse_transform([top_indices[0]])[0]
        confidence = probabilities[top_indices[0]]

        if confidence < 0.4:
            return jsonify({'response': "Sorry, I couldn’t recognize your description. Please provide more details about your symptoms. You may also upload a photo to assist with the diagnosis."})

        elif confidence < 0.5:
            alt_prediction = label_encoder.inverse_transform([top_indices[1]])[0]
            return jsonify({
                'response': f"Based on your description, it could be either '{top_prediction}' or '{alt_prediction}', but I'm not completely sure. Please provide more details (e.g., when did it start, any itching or flaking?). You can also upload an image to help."
            })

        return jsonify({
            'response': f"Thank you for the information. Based on your description, it seems you may have '{top_prediction}'. If symptoms persist or worsen, it’s best to consult a dermatologist. You can also upload an image to confirm the diagnosis."
        })

    except Exception as e:
        return jsonify({'response': f"An error occurred while analyzing your message: {str(e)}"}), 500




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
