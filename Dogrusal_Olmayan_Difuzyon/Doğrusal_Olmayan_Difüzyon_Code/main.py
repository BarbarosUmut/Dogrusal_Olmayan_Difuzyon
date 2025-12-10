"""
Ana program - Demo seçenekleri.
"""

import matplotlib.pyplot as plt
from utils import demo_grayscale, demo_color, create_test_image_grayscale
from analysis import compare_parameters

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NONLINEAR DIFFUSION - PERONA-MALIK MODEL")
    print("=" * 70)

    print("\nDemo Seçenekleri:")
    print("1. Gri tonlamalı görüntü demo")
    print("2. Renkli görüntü demo")
    print("3. Parametre karşılaştırması")
    print("4. Hepsi")

    choice = input("\nSeçiminiz (1-4): ")

    if choice == '1':
        demo_grayscale()
    elif choice == '2':
        demo_color()
    elif choice == '3':
        img = create_test_image_grayscale((128, 128))
        compare_parameters(img, diff_type='pm1')
        compare_parameters(img, diff_type='pm2')
    elif choice == '4':
        demo_grayscale()
        demo_color()
        img = create_test_image_grayscale((128, 128))
        compare_parameters(img, diff_type='pm1')
    else:
        print("Geçersiz seçim!")

    print("\n" + "=" * 70)
    print("PROGRAM TAMAMLANDI")
    print("=" * 70)