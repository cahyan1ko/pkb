import cv2
import numpy as np

def process_image(image_path):
    try:
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Gambar tidak ditemukan atau format tidak valid.")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

        output_path = image_path.replace('.jpg', '_edges.jpg')
        cv2.imwrite(output_path, edges)

        return output_path
    except Exception as e:
        raise ValueError(f"Kesalahan saat memproses gambar: {str(e)}")
