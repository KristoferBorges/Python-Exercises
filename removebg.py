from rembg import remove
from PIL import Image
import io
import onnxruntime

input_path = 'img/photo.jpg'
output_path = 'img/result/imagem_sem_fundo.png'

# entrada
with open(input_path, 'rb') as input_file:
    input_image = input_file.read()

# remove o fundo da imagem
output_image = remove(input_image)

# salvar a imagem
with open(output_path, 'wb') as output_file:
    output_file.write(output_image)

print(f"Imagem sem fundo salva em: {output_path}")