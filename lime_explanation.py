import matplotlib
matplotlib.use('Agg')  # Set the Matplotlib backend to 'Agg' to avoid GUI issues
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle
import os
import numpy as np

# Load the trained classification model
trained_model = load_model("final_model_resnet50_and_fully_connected_nn.h5")

# Load the LabelBinarizer
lb = pickle.load(open("label_binarizer.pkl", "rb"))

# Initialize ResNet50 for feature extraction
resnet_model = ResNet50(weights='imagenet', include_top=False)


# Function to preprocess and extract features from an image
def preprocess_and_extract(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    features = features.mean(axis=(1, 2))  # Global average pooling
    return img_array[0], features  # Return preprocessed image and extracted features


def predict_proba(images):
    features = []
    for img in images:
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = resnet_model.predict(img)
        feature = feature.mean(axis=(1, 2))
        features.append(feature)
    features = np.vstack(features)
    return trained_model.predict(features)


def explain_and_save_lime_image(img_path, output_folder='static/LIME_OUTPUTS/'):
    img_array, features = preprocess_and_extract(img_path)
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img_array, predict_proba, top_labels=1, hide_color=0, num_samples=400
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=10, hide_rest=False)

    # Save the explanation image in LIME_OUTPUTS folder
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'lime_result.png')

    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path
