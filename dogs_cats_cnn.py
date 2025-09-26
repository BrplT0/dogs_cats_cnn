# Dogs vs Cats Sınıflandırma Projesi
# Akbank Derin Öğrenme Bootcamp

# Bu projede köpek ve kedi resimlerini ayırt etmek için CNN modeli geliştireceğiz.
# Veri seti olarak Kaggle'ın ünlü Dogs vs Cats yarışmasının veri setini kullanacağız.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow versiyonu:", tf.__version__)
print("GPU mevcut mu:", tf.config.list_physical_devices('GPU'))

# ## 1. Veri Seti İncelemesi ve Hazırlık

#Train/Test/Validation zaten ayrılmış
base_dir = "dataset/"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

print("Veri seti yapısını kontrol ediyoruz...")
print(f"Train klasörü: {os.path.exists(train_dir)}")
print(f"Test klasörü: {os.path.exists(test_dir)}")
print(f"Validation klasörü: {os.path.exists(validation_dir)}")

if os.path.exists(train_dir):
    print(f"Train'de bulunan sınıflar: {os.listdir(train_dir)}")
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            print(f"  {class_name}: {len(os.listdir(class_path))} resim")

print(f"Kullanılan veri yolu: {train_dir}")

# ## 2. Veri Önişleme ve Görselleştirme

# Bu veri setinde validation zaten var, manual split yapmaya gerek yok
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Piksel değerlerini 0-1 arasına normalize et
    rotation_range=25,        # Görüntüleri rastgele döndür
    width_shift_range=0.1,    # Genişlik yönünde kaydır
    height_shift_range=0.1,   # Yükseklik yönünde kaydır
    horizontal_flip=True,     # Yatay çevir
    zoom_range=0.2           # Rastgele yakınlaştır/uzaklaştır
    # validation_split KALDIRDIK çünkü ayrı validation klasörü var
)

# Test ve validation verisi için sadece normalizasyon
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Veri yükleyicileri oluştur
img_width, img_height = 150, 150
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',  # İki sınıf: köpek ve kedi
    seed=42  # Tekrarlanabilir sonuçlar için
)

# Artık ayrı validation generator kullanabiliyoruz!
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    seed=42
)

print("Sınıf etiketleri:", train_generator.class_indices)
print("Eğitim örnekleri:", train_generator.samples)
print("Validation örnekleri:", validation_generator.samples)

# Örnek görüntüleri görselleştir
def show_sample_images():
    plt.figure(figsize=(12, 8))
    sample_batch = next(train_generator)
    images, labels = sample_batch

    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i])
        plt.title(f"{'Köpek' if labels[i] == 1 else 'Kedi'}")
        plt.axis('off')

    plt.suptitle('Örnek Eğitim Görüntüleri', fontsize=16)
    plt.tight_layout()
    plt.show()

show_sample_images()

# ## 3. CNN Modelinin Oluşturulması

# Derin öğrenme için Convolutional Neural Network mimarisi tasarlıyoruz
# Her katmanın görevi farklı seviyede özellik çıkarımı yapmak

model = Sequential([
    # İlk konvolüsyon bloğu - temel kenarlар ve şekilleri öğrenir
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    # İkinci blok - daha karmaşık desenler
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Üçüncü blok - yüksek seviye özellikler
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Dördüncü blok - çok detaylı özellikler
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Sınıflandırma katmanları
    Flatten(),                          # 2D'den 1D'ye dönüştür
    Dropout(0.5),                       # Overfitting'i önle
    Dense(512, activation='relu'),      # Tam bağlantılı katman
    Dense(1, activation='sigmoid')      # Binary classification için sigmoid
])

# Modeli derle
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Adam optimizer ile öğrenme hızı
    loss='binary_crossentropy',           # Binary sınıflandırma için loss
    metrics=['accuracy']                  # Başarım metriği
)

# Model mimarisini görüntüle
model.summary()

# Toplam parametre sayısını hesapla
total_params = model.count_params()
print(f"\nToplam parametre sayısı: {total_params:,}")

# ## 4. Modelin Eğitilmesi

# Callback'ler tanımla - eğitimi kontrol etmek için
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,          # 5 epoch boyunca iyileşme yoksa dur
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,          # Learning rate'i %20'ye düşür
        patience=3,
        min_lr=0.0001
    )
]

print("Model eğitimi başlıyor...")
print("Bu işlem birkaç dakika sürebilir...")

# Modeli eğit
epochs = 15
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=callbacks,
    verbose=1
)

print("Eğitim tamamlandı!")

# ## 5. Eğitim Sonuçlarının Analizi

# Eğitim sürecini görselleştir
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy grafiği
    axes[0, 0].plot(history.history['accuracy'], 'bo-', label='Eğitim Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy Değişimi')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss grafiği
    axes[0, 1].plot(history.history['loss'], 'bo-', label='Eğitim Loss')
    axes[0, 1].plot(history.history['val_loss'], 'ro-', label='Validation Loss')
    axes[0, 1].set_title('Model Loss Değişimi')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning Rate (eğer kayıtlı ise)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], 'go-')
        axes[1, 0].set_title('Learning Rate Değişimi')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

    # Overfitting analizi
    train_acc = np.array(history.history['accuracy'])
    val_acc = np.array(history.history['val_accuracy'])
    overfitting = train_acc - val_acc

    axes[1, 1].plot(overfitting, 'mo-')
    axes[1, 1].set_title('Overfitting Analizi (Train_Acc - Val_Acc)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Farkı')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# En iyi sonuçları yazdır
max_val_acc = max(history.history['val_accuracy'])
min_val_loss = min(history.history['val_loss'])
print(f"\nEn iyi validation accuracy: {max_val_acc:.4f}")
print(f"En düşük validation loss: {min_val_loss:.4f}")

# ## 6. Model Değerlendirmesi

# Validation seti üzerinde detaylı değerlendirme yapalım
print("Validation seti üzerinde tahmin yapılıyor...")

# Tüm validation verisi için tahmin
validation_generator.reset()
val_predictions = model.predict(validation_generator, verbose=1)
val_predictions_binary = (val_predictions > 0.5).astype(int).flatten()
val_labels = validation_generator.labels

# Confusion Matrix hesapla ve görselleştir
cm = confusion_matrix(val_labels, val_predictions_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Kedi (0)', 'Köpek (1)'],
            yticklabels=['Kedi (0)', 'Köpek (1)'])
plt.title('Confusion Matrix - Karmaşıklık Matrisi')
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.show()

# Detaylı classification report
print("\n=== DETAYLI DEĞERLENDİRME RAPORU ===")
print(classification_report(val_labels, val_predictions_binary,
                          target_names=['Kedi', 'Köpek'], digits=4))

# Accuracy hesapla
accuracy = np.mean(val_labels == val_predictions_binary)
print(f"\nValidation Accuracy: {accuracy:.4f}")

# Yanlış sınıflandırılan örnekleri göster
def show_misclassified_images():
    plt.figure(figsize=(15, 10))

    # Yanlış tahminleri bul
    wrong_indices = np.where(val_labels != val_predictions_binary)[0]

    if len(wrong_indices) > 0:
        # İlk 8 yanlış tahmin
        for i, idx in enumerate(wrong_indices[:8]):
            plt.subplot(2, 4, i+1)

            # Görüntüyü al
            img_batch, _ = validation_generator[idx // batch_size]
            img_idx = idx % batch_size
            img = img_batch[img_idx]

            plt.imshow(img)
            true_label = 'Köpek' if val_labels[idx] == 1 else 'Kedi'
            pred_label = 'Köpek' if val_predictions_binary[idx] == 1 else 'Kedi'
            confidence = val_predictions[idx][0]

            plt.title(f'Gerçek: {true_label}\nTahmin: {pred_label}\nGüven: {confidence:.2f}',
                     color='red')
            plt.axis('off')

        plt.suptitle('Yanlış Sınıflandırılan Örnekler', fontsize=16)
        plt.tight_layout()
        plt.show()
    else:
        print("Tüm tahminler doğru!")

show_misclassified_images()

# ## 7. Grad-CAM ile Görselleştirme

# Modelin hangi bölgelere odaklandığını görselleştirmek için Grad-CAM kullanacağız
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Grad-CAM heatmap oluşturur"""

    # Gradient modeli oluştur
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.outputs]  # outputs kullandık
    )

    # Gradient hesapla
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array, training=False)
        preds = preds[0]  # batch boyutunu indir
        if pred_index is None:
            pred_index = int(tf.argmax(preds))  # tensor -> int dönüşümü
        class_channel = preds[pred_index]

    # Son konvolüsyon katmanına göre gradient
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Heatmap hesapla
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize et
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Örnek görüntüler üzerinde Grad-CAM uygula
def visualize_gradcam_examples():
    plt.figure(figsize=(15, 12))

    # Bazı örnek görüntüler al
    validation_generator.reset()
    sample_batch = next(validation_generator)
    images, labels = sample_batch

    for i in range(6):  # 6 örnek göster
        img = np.expand_dims(images[i], axis=0)
        pred = model.predict(img, verbose=0)[0][0]

        # Son konvolüsyon katmanından Grad-CAM
        heatmap = make_gradcam_heatmap(img, model, "conv2d_3")  # Son conv katmanı

        # Heatmap'i orijinal boyuta getir
        heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], (150, 150))
        heatmap_resized = tf.squeeze(heatmap_resized)

        # Görselleştir
        plt.subplot(3, 6, i+1)
        plt.imshow(images[i])
        plt.title(f'Orijinal\n{"Köpek" if labels[i] == 1 else "Kedi"}')
        plt.axis('off')

        plt.subplot(3, 6, i+7)
        plt.imshow(heatmap_resized, cmap='jet')
        plt.title(f'Heatmap\nGüven: {pred:.2f}')
        plt.axis('off')

        plt.subplot(3, 6, i+13)
        plt.imshow(images[i])
        plt.imshow(heatmap_resized, cmap='jet', alpha=0.4)
        plt.title(f'Overlay\n{"Köpek" if pred > 0.5 else "Kedi"}')
        plt.axis('off')

    plt.suptitle('Grad-CAM Analizi - Model Nerelere Bakıyor?', fontsize=16)
    plt.tight_layout()
    plt.show()

print("Grad-CAM görselleştirmesi hazırlanıyor...")
visualize_gradcam_examples()

# ## 8. Hiperparametre Optimizasyonu Denemeleri

# Farklı hiperparametreler deneyerek en iyi sonucu bulmaya çalışalım
print("\n=== HİPERPARAMETRE OPTİMİZASYONU ===")

hyperparams_results = {}

# Farklı learning rate'ler dene
learning_rates = [0.001, 0.0001, 0.00001]

for lr in learning_rates:
    print(f"\nLearning Rate: {lr} test ediliyor...")

    # Basit model oluştur
    test_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    test_model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Kısa eğitim (test için)
    hist = test_model.fit(
        train_generator,
        steps_per_epoch=50,  # Hızlı test için
        epochs=5,
        validation_data=validation_generator,
        validation_steps=25,
        verbose=0
    )

    best_val_acc = max(hist.history['val_accuracy'])
    hyperparams_results[f'lr_{lr}'] = best_val_acc
    print(f"En iyi validation accuracy: {best_val_acc:.4f}")

    # Belleği temizle
    del test_model
    tf.keras.backend.clear_session()

# Sonuçları göster
print("\n=== HİPERPARAMETRE SONUÇLARI ===")
for param, score in hyperparams_results.items():
    print(f"{param}: {score:.4f}")

# ## 9. Sonuç ve Yorumlar

print("\n" + "="*50)
print("PROJE SONUÇ ÖZETİ")
print("="*50)

final_val_acc = max(history.history['val_accuracy'])
final_train_acc = max(history.history['accuracy'])

print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Overfitting Durumu: {'Var' if final_train_acc - final_val_acc > 0.1 else 'Kontrol Altında'}")
print(f"Model Parametreleri: {total_params:,}")

print("\nKullanılan Teknikler:")
print("✓ Data Augmentation (Rotation, Flip, Shift, Zoom)")
print("✓ Dropout ile Regularization")
print("✓ Early Stopping")
print("✓ Learning Rate Scheduling")
print("✓ Grad-CAM Görselleştirme")
print("✓ Hiperparametre Optimizasyonu")

print("\nModel Başarı Durumu:")
if final_val_acc > 0.90:
    print("🎉 Mükemmel! Model çok iyi çalışıyor.")
elif final_val_acc > 0.85:
    print("✅ İyi! Model başarılı sayılır.")
elif final_val_acc > 0.80:
    print("⚠️  Orta! Daha fazla eğitim gerekebilir.")
else:
    print("❌ Düşük! Model mimarisi veya hiperparametreler gözden geçirilmeli.")

print("\nProje tamamlandı! 🚀")
