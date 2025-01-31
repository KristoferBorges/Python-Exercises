from rembg import remove
from PIL import Image
import io
import onnxruntime

input_path = 'img/photo.jpg'
output_path = 'img/result/imagem_sem_fundo.png'

# Carregar a imagem de entrada
with open(input_path, 'rb') as input_file:
    input_image = input_file.read()

# Remover o fundo da imagem
output_image = remove(input_image)

# Salvar a imagem resultante
with open(output_path, 'wb') as output_file:
    output_file.write(output_image)

print(f"Imagem sem fundo salva em: {output_path}")