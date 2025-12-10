"""
Difüzivite fonksiyonları.
"""

import numpy as np


def diffusivity_pm1(gradient_magnitude, lambda_param):
    """
    Perona-Malik Difüzivite - Tip 1
    """
    return np.exp(-(gradient_magnitude ** 2) / (lambda_param ** 2))


def diffusivity_pm2(gradient_magnitude, lambda_param):
    """
    Perona-Malik Difüzivite - Tip 2
    """
    return 1.0 / (1.0 + (gradient_magnitude ** 2) / (lambda_param ** 2))


def diffusivity_charbonnier(gradient_magnitude, lambda_param):
    """
    Charbonnier Difüzivite
    """
    return 1.0 / np.sqrt(1.0 + (gradient_magnitude ** 2) / (lambda_param ** 2))


def compute_diffusivity(gradient_magnitude, diffusivity_type, lambda_param):
    """
    Seçili difüzivite fonksiyonunu hesaplar.
    """
    if diffusivity_type == 'pm1':
        return diffusivity_pm1(gradient_magnitude, lambda_param)
    elif diffusivity_type == 'pm2':
        return diffusivity_pm2(gradient_magnitude, lambda_param)
    elif diffusivity_type == 'charbonnier':
        return diffusivity_charbonnier(gradient_magnitude, lambda_param)
    else:
        raise ValueError("Geçersiz difüzivite tipi")