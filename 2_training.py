import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. BUAT FOLDER RESULTS SECARA OTOMATIS
results_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(results_dir, exist_ok=True)

# 2. PERSIAPAN DATASET
dataset_path = os.path.join(os.getcwd(), 'dataset_raw')

datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True
)

print("Memuat Data Training...")
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("Memuat Data Validasi...")
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False # Penting untuk evaluasi confusion matrix
)

# Ambil nama-nama kelas secara otomatis dari folder
class_labels = list(val_generator.class_indices.keys())

# 3. MEMBANGUN MODEL TRANSFER LEARNING
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(class_labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. MELATIH MODEL (TRAINING)
EPOCHS = 15
print(f"Mulai proses training selama {EPOCHS} epoch...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# 5. MENYIMPAN MODEL UTAMA
model_path = os.path.join(os.getcwd(), 'model_modul_elektronik.h5')
model.save(model_path)
print(f"Model berhasil disimpan di: {model_path}")

# 6. MENYIMPAN GRAFIK AKURASI
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'accuracy.png'))
plt.close()

# 7. MENYIMPAN GRAFIK LOSS
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'loss.png'))
plt.close()

# 8. MEMBUAT DAN MENYIMPAN CONFUSION MATRIX
print("Membuat Confusion Matrix...")
val_generator.reset()
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()

# 9. MENYIMPAN LAPORAN METRIK KE FILE CSV (model_comparison.csv)
history_df = pd.DataFrame(history.history)
csv_path = os.path.join(results_dir, 'final_model_comparison.csv')
history_df.to_csv(csv_path, index=False)
print(f"Laporan CSV tersimpan di: {csv_path}")

print("Selesai! Semua file evaluasi lengkap kini tersimpan rapi di dalam folder 'results'.")