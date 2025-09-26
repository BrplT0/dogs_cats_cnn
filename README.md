# Kedi ve Köpek Sınıflandırması - CNN Projesi

Bu repo, Global AI Hub bootcamp kapsamında geliştirilmiş kedi-köpek görüntü sınıflandırma projesidir. TensorFlow kullanılarak CNN modeli ile %88.56 validation accuracy elde edilmiştir.

## Giriş

Bu projede Kaggle'dan alınan Dogs vs Cats veri seti kullanılarak derin öğrenme tabanlı görüntü sınıflandırması gerçekleştirilmiştir. 

**Kullanılan Veri Seti:** 25,000 kedi ve köpek görüntüsü (20,000 train + 5,000 validation)

**Uygulanan Yöntemler:**
- Convolutional Neural Network (CNN) 
- Data Augmentation
- Dropout Regularization
- Early Stopping
- Hiperparametre Optimizasyonu
- Grad-CAM Görselleştirme

## Metrikler

Proje sonunda elde edilen temel metrikler:

- **Final Validation Accuracy:** %88.56
- **Final Training Accuracy:** %86.35
- **Model Parametreleri:** 3,453,121
- **Overfitting Durumu:** Kontrol altında

### Hiperparametre Optimizasyonu Sonuçları:
- Learning Rate 0.001: %59.87 accuracy
- Learning Rate 0.0001: %67.25 accuracy (en iyi)
- Learning Rate 1e-05: %59.62 accuracy

Bu sonuçlar, modelin başarılı bir şekilde kedi ve köpek görüntülerini ayırt edebildiğini göstermektedir. Validation ve training accuracy arasındaki dengeli fark, overfitting probleminin başarıyla kontrol altına alındığını işaret eder.

## Ekler

Proje kapsamında aşağıdaki ek analizler ve görselleştirmeler yapılmıştır:

- **Confusion Matrix:** Model tahminlerinin detaylı analizi
- **Training History:** Epoch bazında accuracy ve loss değişimleri
- **Grad-CAM Analizi:** Modelin hangi özelliklere odaklandığının görselleştirilmesi
- **Yanlış Sınıflandırılan Örnekler:** Model hatalarının analizi

## Sonuç ve Gelecek Çalışmalar

Bu çalışma ile temel CNN mimarisi kullanarak görüntü sınıflandırmasında başarılı sonuçlar elde edildi. 

**Gelecek geliştirmeler için öneriler:**
- Transfer Learning (VGG16, ResNet) ile performans iyileştirme
- Web arayüzü geliştirme (Streamlit/Flask)
- Mobile deployment için model optimizasyonu
- Gerçek zamanlı kamera ile tahmin sistemi
- Daha büyük veri setleri ile test

Bu proje, bilgisayarla görme alanında daha karmaşık projelerin temelini oluşturmaktadır.

## Linkler

Projeye ait Kaggle notebook:
https://www.kaggle.com/code/beratpolat/dogs-cats-cnn
