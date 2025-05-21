import streamlit as st
import pandas as pd
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2

st.set_page_config(page_title="Deepfake Audio Detection", page_icon="")

class_names = ['real', 'fake']

def save_file(sound_file):
    os.makedirs('audio_files', exist_ok=True)  # Create folder if not exists
    with open(os.path.join('audio_files', sound_file.name), 'wb') as f:
        f.write(sound_file.getbuffer())
    return sound_file.name

def create_spectrogram(sound):
    audio_file = os.path.join('audio_files', sound)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file, sr=None)  # use original sampling rate
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr, ax=ax)
    ax.axis('off')

    plt.savefig('melspectrogram.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # close figure to free memory

    image_data = load_img('melspectrogram.png', target_size=(224, 224))
    st.image(image_data)
    return image_data

def predictions(image_data, model):
    img_array = np.array(image_data)
    img_array1 = img_array / 255.0
    img_batch = np.expand_dims(img_array1, axis=0)
    prediction = model.predict(img_batch)
    class_label = np.argmax(prediction)
    return class_label, prediction

def lime_predict(image_data, model):
    img_array = np.array(image_data)
    img_array1 = img_array / 255.0
    img_batch = np.expand_dims(img_array1, axis=0)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array1.astype('double'), 
        model.predict, 
        hide_color=0, 
        num_samples=1000
    )

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    class_label = np.argmax(model.predict(img_batch))
    temp, mask = explanation.get_image_and_mask(
        class_label, positive_only=False, num_features=8, hide_rest=True
    )
    axs[0].imshow(image_data)
    axs[0].axis('off')
    axs[0].set_title("Original Image")

    axs[1].imshow(mark_boundaries(temp, mask))
    axs[1].axis('off')
    axs[1].set_title(f"Predicted class: {class_names[class_label]}")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    return fig

def grad_predict(image_data, model, preds, class_idx):
    img_array = img_to_array(image_data)
    x = np.expand_dims(img_array, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)

    # Use VGG16 pretrained model for Grad-CAM
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    last_conv_layer = base_model.get_layer('block5_conv3')
    grad_model = tf.keras.models.Model([base_model.inputs], [last_conv_layer.output, base_model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(x)
        class_channel = preds[:, class_idx]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (x.shape[2], x.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(np.uint8(x[0]), 0.6, heatmap, 0.4, 0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_data)
    ax[0].axis('off')
    ax[0].set_title("Original Image")

    ax[1].imshow(superimposed_img)
    ax[1].axis('off')
    ax[1].set_title(f"Grad-CAM: {class_names[class_idx]}")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    return superimposed_img

def main():
    page = st.sidebar.selectbox("App Selections", ["Homepage", "About"])
    if page == "Homepage":
        st.title("Deepfake Audio Detection using XAI")
        homepage()
    elif page == "About":
        about()

def about():
    st.title("About present work")
    st.markdown("""
**Deepfake audio refers to synthetically created audio by digital or manual means. An emerging field, it is used to not only create legal digital hoaxes, but also fool humans into believing it is a human speaking to them. Through this project, we create our own deep faked audio using Generative Adversarial Neural Networks (GANs) and objectively evaluate generator quality using Fréchet Audio Distance (FAD) metric. We augment a pre-existing dataset of real audio samples with our fake generated samples and classify data as real or fake using MobileNet, Inception, VGG and custom CNN models. MobileNet is the best performing model with an accuracy of 91.5% and precision of 0.507. We further convert our black box deep learning models into white box models, by using explainable AI (XAI) models. We quantitatively evaluate the classification of a MEL Spectrogram through LIME, SHAP and GradCAM models. We compare the features of a spectrogram that an XAI model focuses on to provide a qualitative analysis of frequency distribution in spectrograms.**

The goal of this project is to study features of audio and bridge the gap of explain ability in deep fake audio detection, through our novel system pipeline. The findings of this study are applicable to the fields of phishing audio calls and digital mimicry detection on video streaming platforms. The use of XAI will provide end-users a clear picture of frequencies in audio that are flagged as fake, enabling them to make better decisions in generation of fake samples through GANs.
""")

def homepage():
    st.write('___')
    st.subheader("Choose a wav file")
    uploaded_file = st.file_uploader('Upload .wav audio file', type='wav')

    if uploaded_file is not None:
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')

        save_file(uploaded_file)
        sound = uploaded_file.name

        with st.spinner('Loading model and fetching results...'):
            try:
                model = tf.keras.models.load_model('saved_model/model/saved_model.pb')
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()

            spec = create_spectrogram(sound)

            st.write('### Classification results:')
            class_label, prediction = predictions(spec, model)
            st.write(f"#### The uploaded audio file is **{class_names[class_label]}**")

            if st.button('Show XAI Metrics'):
                st.write('### XAI Metrics using LIME')
                with st.spinner('Generating LIME explanation...'):
                    lime_predict(spec, model)

                st.write('### XAI Metrics using Grad-CAM')
                with st.spinner('Generating Grad-CAM explanation...'):
                    grad_predict(spec, model, prediction, class_label)

    else:
        st.info("Please upload a .wav file")

if __name__ == "__main__":
    main()
