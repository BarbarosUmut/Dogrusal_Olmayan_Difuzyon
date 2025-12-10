"""
Yardımcı fonksiyonlar ve demo işlevleri.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from nonlinear_diffusion import NonlinearDiffusion, ColorNonlinearDiffusion
from analysis import linear_diffusion, plot_comparison, plot_statistics, compare_parameters


def create_test_image_grayscale(size=(256, 256)):
    """
    Gri tonlamalı test görüntüsü oluşturur.
    """
    img = np.random.randint(0, 256, size, dtype=np.uint8)
    img[50:80, :] = 200
    img[:, 100:130] = 50
    noise = np.random.normal(0, 25, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def create_test_image_color(size=(256, 256, 3)):
    """
    Renkli test görüntüsü oluşturur.
    """
    img = np.random.randint(0, 256, size, dtype=np.uint8)
    img[50:80, :, 0] = 200
    img[:, 100:130, 1] = 200
    img[150:180, 150:180, 2] = 200
    noise = np.random.normal(0, 20, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def demo_grayscale():
    """
    Gri tonlamalı görüntü için demo.
    """
    print("\n" + "=" * 70)
    print("GRİ TONLAMALI GÖRÜNTÜ İÇİN NONLINEAR DİFÜZYON DEMO")
    print("=" * 70)

    img = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Test görüntüsü bulunamadı. Sentetik görüntü oluşturuluyor...")
        img = create_test_image_grayscale()

    print(f"Görüntü boyutu: {img.shape}")

    print("\nLinear difüzyon uygulanıyor...")
    linear_result = linear_diffusion(img, num_iterations=50)

    results = {}

    for diff_type, name in [('pm1', 'PM Type 1'),
                            ('pm2', 'PM Type 2'),
                            ('charbonnier', 'Charbonnier')]:
        print(f"\n{name} difüzyonu uygulanıyor...")
        diffusion = NonlinearDiffusion(lambda_param=10.0, sigma=1.0,
                                       dt=0.25, num_iterations=50)
        diffusion.set_diffusivity(diff_type)
        result, history = diffusion.apply(img)
        results[diff_type] = (result, history)

    print("\nSonuçlar görselleştiriliyor...")
    plot_comparison(img, linear_result,
                    results['pm1'][0], results['pm2'][0], results['charbonnier'][0],
                    save_path='comparison_grayscale.png')

    for diff_type, name in [('pm1', 'PM Type 1'),
                            ('pm2', 'PM Type 2'),
                            ('charbonnier', 'Charbonnier')]:
        plot_statistics(results[diff_type][1],
                        title=f'{name} - İstatistikler',
                        save_path=f'statistics_{diff_type}.png')


def demo_color():
    """
    Renkli görüntü için demo.
    """
    print("\n" + "=" * 70)
    print("RENKLİ GÖRÜNTÜ İÇİN NONLINEAR DİFÜZYON DEMO")
    print("=" * 70)

    img = cv2.imread('test_image_color.png')

    if img is None:
        print("Renkli test görüntüsü bulunamadı. Sentetik görüntü oluşturuluyor...")
        img = create_test_image_color()
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Görüntü boyutu: {img.shape}")

    print("\nRenkli PM Type 1 difüzyonu uygulanıyor...")
    color_diffusion = ColorNonlinearDiffusion(lambda_param=15.0, sigma=1.0,
                                              dt=0.25, num_iterations=30)
    color_diffusion.set_diffusivity('pm1')
    result, history = color_diffusion.apply_color(img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title('Orijinal')
    axes[0].axis('off')

    axes[1].imshow(result)
    axes[1].set_title('PM Type 1 Difüzyon Sonucu')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('color_diffusion_result.png', dpi=300)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    iterations = range(1, len(history['mean_r']) + 1)

    axes[0].plot(iterations, history['mean_r'], 'r-', label='Red', linewidth=2)
    axes[0].plot(iterations, history['mean_g'], 'g-', label='Green', linewidth=2)
    axes[0].plot(iterations, history['mean_b'], 'b-', label='Blue', linewidth=2)
    axes[0].set_xlabel('İterasyon')
    axes[0].set_ylabel('Ortalama Yoğunluk')
    axes[0].set_title('Kanal Ortalamaları')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, history['variance_r'], 'r-', label='Red', linewidth=2)
    axes[1].plot(iterations, history['variance_g'], 'g-', label='Green', linewidth=2)
    axes[1].plot(iterations, history['variance_b'], 'b-', label='Blue', linewidth=2)
    axes[1].set_xlabel('İterasyon')
    axes[1].set_ylabel('Varyans')
    axes[1].set_title('Kanal Varyansları')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iterations, history['gradient_magnitude'], 'k-', linewidth=2)
    axes[2].set_xlabel('İterasyon')
    axes[2].set_ylabel('Toplam Gradyan')
    axes[2].set_title('Toplam Gradyan Büyüklüğü')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('color_diffusion_statistics.png', dpi=300)
    plt.show()