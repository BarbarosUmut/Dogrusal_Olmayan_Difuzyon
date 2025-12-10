# Nonlinear Diffusion - Perona-Malik Model


###  Dosya İçeriği

Bu klasörde aşağıdaki dosyalar bulunmaktadır:

1. **Doğrusal_Olmayan_Difüzyon_ile_Görüntü_İyileştirme.pdf**
   - Projenin Türkçe raporu

2. **Doğrusal_Olmayan_Difüzyon_Code Dosya Klasörü**
   - Perona-Malik modeli implementasyonu
   - Üç difüzivite fonksiyonu (PM Type 1, PM Type 2, Charbonnier)
   - Gri tonlamalı ve renkli görüntü desteği
   - Analiz ve görselleştirme fonksiyonları


## Proje Özeti

Bu ödev, doğrusal olmayan difüzyon filtreleme tekniklerini öğrenmeniz ve 
uygulamanız için tasarlanmıştır. Perona-Malik modeli kullanılarak görüntülerdeki 
kenarlar korunurken gürültü giderilmesi amaçlanmaktadır.

### Temel Konular:
- Nonlinear PDE'ler
- Perona-Malik difüzyon modeli
- Difüzivite fonksiyonları
- Görüntü yumuşatma ve kenar koruma
- Gradyan hesaplama ve divergence


## Kurulum ve Kullanım

### Gerekli Kütüphaneler:
```bash
pip install numpy opencv-python matplotlib scipy
```

### Kullanım:
```python
python nonlinear_diffusion_solution.py
```

Program çalıştırıldığında size demo seçenekleri sunulacaktır:
1. Gri tonlamalı görüntü demo
2. Renkli görüntü demo
3. Parametre karşılaştırması
4. Hepsi


## Problemler

### Problem 1.1: Perona-Malik Modeli
Üç farklı difüzivite fonksiyonu implement edilmiştir:

**A. PM Type 1:**
```
g(|x|) = exp(-|x|²/λ²)
```

**B. PM Type 2:**
```
g(|x|) = 1 / (1 + |x|²/λ²)
```

**C. Charbonnier:**
```
g(|x|) = 1 / √(1 + |x|²/λ²)
```

### Problem 1.2: Karşılaştırmalı Analiz
- Linear vs. Nonlinear difüzyon karşılaştırması
- Parametre etkilerinin analizi (λ, σ, T)
- İstatistiksel değişimler (ortalama, varyans, gradyan)

### Problem 1.3: Renkli Görüntü Desteği
- RGB kanalları için difüzyon
- Kanallar arası tutarlılık
- Renk korumalı yumuşatma


## Kod Yapısı

### Ana Sınıflar:

**NonlinearDiffusion**
- Gri tonlamalı görüntüler için
- Üç difüzivite fonksiyonu
- Gradyan ve divergence hesaplama
- İteratif difüzyon süreci

**ColorNonlinearDiffusion**
- Renkli görüntüler için genişletilmiş sınıf
- Kanal bazlı işleme
- Ortak difüzivite hesaplama

### Yardımcı Fonksiyonlar:
- `linear_diffusion()` - Karşılaştırma için linear difüzyon
- `plot_comparison()` - Sonuçların görselleştirilmesi
- `plot_statistics()` - İstatistiklerin grafikleştirilmesi
- `compare_parameters()` - Parametre analizi


## Beklenen Sonuçlar

Kod başarıyla çalıştırıldığında aşağıdaki çıktılar elde edilir:

1. **Görsel Karşılaştırmalar:**
   - Orijinal vs. filtrelenmiş görüntüler
   - Farklı difüzivite fonksiyonlarının sonuçları
   - Parametre varyasyonlarının etkileri

2. **İstatistiksel Grafikler:**
   - Ortalama yoğunluk değişimi
   - Varyans değişimi
   - Gradyan büyüklüğü değişimi

3. **Kaydedilen Dosyalar:**
   - comparison_grayscale.png
   - statistics_pm1.png
   - statistics_pm2.png
   - statistics_charbonnier.png
   - color_diffusion_result.png
   - lambda_comparison_*.png
   - sigma_comparison_*.png


## Kaynaklar

1. **P. Perona and J. Malik** (1990)
   "Scale space and edge detection using anisotropic diffusion"
   IEEE Transactions on Pattern Analysis and Machine Intelligence, 12:629-639

2. **P. Charbonnier et al.** (1994)
   "Two deterministic half-quadratic regularization algorithms for computed imaging"
   Proc. 1994 IEEE International Conference on Image Processing

3. **J. Weickert** (1998)
   "Anisotropic Diffusion in Image Processing"

## İletişim

Sorularınız için:
- E-posta: 230212065@ostimteknik.edu.tr

