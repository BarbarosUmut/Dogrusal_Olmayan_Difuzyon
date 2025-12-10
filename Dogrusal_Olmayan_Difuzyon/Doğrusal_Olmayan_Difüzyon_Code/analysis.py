"""
Analiz ve görselleştirme fonksiyonları.
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def linear_diffusion(image, num_iterations=50, dt=0.25):
    """
    Linear (Gaussian) diffusion - karşılaştırma için.
    """
    result = image.astype(np.float64)

    for i in range(num_iterations):
        laplacian = cv2.Laplacian(result, cv2.CV_64F)
        result = result + dt * laplacian
        result = np.clip(result, 0, 255)

    return result.astype(np.uint8)


def plot_comparison(original, linear, pm1, pm2, charbonnier, save_path=None):
    """
    Farklı difüzyon modellerinin sonuçlarını karşılaştırır.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    images = [original, linear, pm1, pm2, charbonnier]
    titles = ['Orijinal', 'Linear Diffusion', 'PM Type 1',
              'PM Type 2', 'Charbonnier']

    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')

    axes[1, 2].axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Karşılaştırma grafiği kaydedildi: {save_path}")

    plt.show()


def plot_statistics(history, title='Statistics Over Iterations', save_path=None):
    """
    İterasyonlar boyunca istatistikleri görselleştirir.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    iterations = range(1, len(history['mean']) + 1)

    axes[0].plot(iterations, history['mean'], 'b-', linewidth=2)
    axes[0].set_xlabel('İterasyon')
    axes[0].set_ylabel('Ortalama Yoğunluk')
    axes[0].set_title('Ortalama Yoğunluk Değişimi')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, history['variance'], 'r-', linewidth=2)
    axes[1].set_xlabel('İterasyon')
    axes[1].set_ylabel('Varyans')
    axes[1].set_title('Yoğunluk Varyansı Değişimi')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iterations, history['gradient_magnitude'], 'g-', linewidth=2)
    axes[2].set_xlabel('İterasyon')
    axes[2].set_ylabel('Toplam Gradyan Büyüklüğü')
    axes[2].set_title('Toplam Gradyan Büyüklüğü Değişimi')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"İstatistik grafiği kaydedildi: {save_path}")

    plt.show()


def compare_parameters(image, diff_type='pm1', lambdas=[5, 10, 20],
                       sigmas=[0.5, 1.0, 2.0], save_dir='results'):
    """
    Farklı parametrelerle sonuçları karşılaştırır.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Lambda karşılaştırması
    print(f"\n{'=' * 60}")
    print(f"Lambda parametresi karşılaştırması ({diff_type})")
    print(f"{'=' * 60}")

    fig, axes = plt.subplots(1, len(lambdas) + 1, figsize=(15, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal')
    axes[0].axis('off')

    for i, lambda_val in enumerate(lambdas, 1):
        from nonlinear_diffusion import NonlinearDiffusion
        diffusion = NonlinearDiffusion(lambda_param=lambda_val, sigma=1.0)
        diffusion.set_diffusivity(diff_type)
        result, _ = diffusion.apply(image)

        axes[i].imshow(result, cmap='gray')
        axes[i].set_title(f'λ = {lambda_val}')
        axes[i].axis('off')

    plt.suptitle(f'{diff_type.upper()} - Lambda Karşılaştırması',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lambda_comparison_{diff_type}.png', dpi=300)
    plt.close()

    # Sigma karşılaştırması
    print(f"\n{'=' * 60}")
    print(f"Sigma parametresi karşılaştırması ({diff_type})")
    print(f"{'=' * 60}")

    fig, axes = plt.subplots(1, len(sigmas) + 1, figsize=(15, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal')
    axes[0].axis('off')

    for i, sigma_val in enumerate(sigmas, 1):
        from nonlinear_diffusion import NonlinearDiffusion
        diffusion = NonlinearDiffusion(lambda_param=10.0, sigma=sigma_val)
        diffusion.set_diffusivity(diff_type)
        result, _ = diffusion.apply(image)

        axes[i].imshow(result, cmap='gray')
        axes[i].set_title(f'σ = {sigma_val}')
        axes[i].axis('off')

    plt.suptitle(f'{diff_type.upper()} - Sigma Karşılaştırması',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sigma_comparison_{diff_type}.png', dpi=300)
    plt.close()

    print(f"\nKarşılaştırma grafikleri '{save_dir}' dizinine kaydedildi.")