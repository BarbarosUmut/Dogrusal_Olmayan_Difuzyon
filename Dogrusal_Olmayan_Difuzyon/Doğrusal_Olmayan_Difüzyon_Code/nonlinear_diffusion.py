"""
Nonlinear Diffusion - Perona-Malik Model Implementation
Ana sınıflar ve temel fonksiyonlar.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class NonlinearDiffusion:
    """
    Nonlinear Diffusion filtreleme için ana sınıf.
    """

    def __init__(self, lambda_param=10.0, sigma=1.0, dt=0.25, num_iterations=50):
        """
        Nonlinear Diffusion filtresini initialize eder.
        """
        self.lambda_param = lambda_param
        self.sigma = sigma
        self.dt = dt
        self.num_iterations = num_iterations
        self.diffusivity_type = 'pm1'  # Varsayılan: PM Type 1

    def set_diffusivity(self, diff_type):
        """
        Difüzivite fonksiyon tipini ayarlar.
        """
        if diff_type not in ['pm1', 'pm2', 'charbonnier']:
            raise ValueError("Geçersiz difüzivite tipi. 'pm1', 'pm2', veya 'charbonnier' olmalı.")
        self.diffusivity_type = diff_type

    def compute_gradients(self, image):
        """
        Görüntünün x ve y yönlerindeki gradyanlarını hesaplar.
        """
        padded = np.pad(image, 1, mode='edge')

        grad_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
        grad_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0

        return grad_x, grad_y

    def compute_gradient_magnitude(self, grad_x, grad_y):
        """
        Gradyan büyüklüğünü hesaplar.
        """
        return np.sqrt(grad_x ** 2 + grad_y ** 2)

    def diffusivity_pm1(self, gradient_magnitude):
        """
        Perona-Malik Difüzivite - Tip 1
        """
        return np.exp(-(gradient_magnitude ** 2) / (self.lambda_param ** 2))

    def diffusivity_pm2(self, gradient_magnitude):
        """
        Perona-Malik Difüzivite - Tip 2
        """
        return 1.0 / (1.0 + (gradient_magnitude ** 2) / (self.lambda_param ** 2))

    def diffusivity_charbonnier(self, gradient_magnitude):
        """
        Charbonnier Difüzivite
        """
        return 1.0 / np.sqrt(1.0 + (gradient_magnitude ** 2) / (self.lambda_param ** 2))

    def compute_diffusivity(self, gradient_magnitude):
        """
        Seçili difüzivite fonksiyonunu hesaplar.
        """
        if self.diffusivity_type == 'pm1':
            return self.diffusivity_pm1(gradient_magnitude)
        elif self.diffusivity_type == 'pm2':
            return self.diffusivity_pm2(gradient_magnitude)
        elif self.diffusivity_type == 'charbonnier':
            return self.diffusivity_charbonnier(gradient_magnitude)

    def diffusion_step(self, image):
        """
        Tek bir difüzyon iterasyon adımı gerçekleştirir.
        """
        if self.sigma > 0:
            smoothed = gaussian_filter(image, sigma=self.sigma)
        else:
            smoothed = image.copy()

        grad_x, grad_y = self.compute_gradients(smoothed)
        gradient_mag = self.compute_gradient_magnitude(grad_x, grad_y)

        g = self.compute_diffusivity(gradient_mag)

        grad_x_orig, grad_y_orig = self.compute_gradients(image)

        diffusion_x = g * grad_x_orig
        diffusion_y = g * grad_y_orig

        div_x, _ = self.compute_gradients(diffusion_x)
        _, div_y = self.compute_gradients(diffusion_y)
        divergence = div_x + div_y

        updated_image = image + self.dt * divergence
        updated_image = np.clip(updated_image, 0, 255)

        return updated_image

    def apply(self, image):
        """
        Görüntüye nonlinear difüzyon filtresi uygular.
        """
        result = image.astype(np.float64)

        history = {
            'mean': [],
            'variance': [],
            'gradient_magnitude': []
        }

        for i in range(self.num_iterations):
            result = self.diffusion_step(result)

            history['mean'].append(np.mean(result))
            history['variance'].append(np.var(result))

            gx, gy = self.compute_gradients(result)
            grad_mag = self.compute_gradient_magnitude(gx, gy)
            history['gradient_magnitude'].append(np.sum(grad_mag))

            if (i + 1) % 10 == 0:
                print(f"  İterasyon {i + 1}/{self.num_iterations} tamamlandı")

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result, history


class ColorNonlinearDiffusion(NonlinearDiffusion):
    """
    Renkli görüntüler için nonlinear diffusion.
    """

    def apply_color(self, color_image):
        """
        Renkli görüntüye nonlinear difüzyon uygular.
        """
        image_float = color_image.astype(np.float64)
        channels = [image_float[:, :, i] for i in range(3)]
        result_channels = [ch.copy() for ch in channels]

        history = {
            'mean_r': [], 'mean_g': [], 'mean_b': [],
            'variance_r': [], 'variance_g': [], 'variance_b': [],
            'gradient_magnitude': []
        }

        print(f"Renkli görüntü difüzyonu başlıyor...")
        print(f"  Difüzivite: {self.diffusivity_type}")
        print(f"  Lambda: {self.lambda_param}, Sigma: {self.sigma}")
        print(f"  İterasyon sayısı: {self.num_iterations}")

        for iteration in range(self.num_iterations):
            if self.sigma > 0:
                smoothed_channels = [gaussian_filter(ch, sigma=self.sigma)
                                     for ch in result_channels]
            else:
                smoothed_channels = [ch.copy() for ch in result_channels]

            gradients = [self.compute_gradients(ch) for ch in smoothed_channels]

            total_gradient_mag = np.zeros_like(result_channels[0])
            for grad_x, grad_y in gradients:
                grad_mag = self.compute_gradient_magnitude(grad_x, grad_y)
                total_gradient_mag += grad_mag

            g = self.compute_diffusivity(total_gradient_mag)

            new_channels = []
            for i, channel in enumerate(result_channels):
                grad_x, grad_y = self.compute_gradients(channel)

                diffusion_x = g * grad_x
                diffusion_y = g * grad_y

                div_x, _ = self.compute_gradients(diffusion_x)
                _, div_y = self.compute_gradients(diffusion_y)
                divergence = div_x + div_y

                updated = channel + self.dt * divergence
                updated = np.clip(updated, 0, 255)
                new_channels.append(updated)

            result_channels = new_channels

            history['mean_r'].append(np.mean(result_channels[0]))
            history['mean_g'].append(np.mean(result_channels[1]))
            history['mean_b'].append(np.mean(result_channels[2]))

            history['variance_r'].append(np.var(result_channels[0]))
            history['variance_g'].append(np.var(result_channels[1]))
            history['variance_b'].append(np.var(result_channels[2]))

            history['gradient_magnitude'].append(np.sum(total_gradient_mag))

            if (iteration + 1) % 10 == 0:
                print(f"  İterasyon {iteration + 1}/{self.num_iterations} tamamlandı")

        result = np.stack(result_channels, axis=2)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result, history