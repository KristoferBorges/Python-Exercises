"""
Exercício Python 107: Crie um módulo chamado moeda.py que tenha as funções incorporadas aumentar(), diminuir(), dobro()
e metade(). Faça também um programa que importe esse módulo e use algumas dessas funções.
"""

from modulos_k import moeda


def processos(number):
    dobrado = moeda.dobro(number)
    metade = moeda.metade(number)
    aumentado = moeda.aumentar(number)
    diminuido = moeda.diminuir(number)
    print()
    print(f' [!] - Valor digitado [{number}]')
    print(f' [!] - Dobro = {dobrado}')
    print(f' [!] - Metade = {metade}')
    print(f' [!] - Aumentado = {aumentado}')
    print(f' [!] - Diminuido = {diminuido}')


numero = float(input(' [?] - DIGITE UM VALOR E VEJA SEU:\n -> Dobro\n -> Metade\n -> + 10%\n -> - 10%\n [?] - R$ '))
processos(numero)
