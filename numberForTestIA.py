import numpy as np
import random
import os  # Para manipular pastas e arquivos

# Criar pasta para salvar o aprendizado
pasta_treino = "treinoIA"
arquivo_qtable = os.path.join(pasta_treino, "q_table.npy")

# Configurações do ambiente
numeros_possiveis = list(range(101))  # Agora inclui de 0 a 100
indice_mapeado = {num: i for i, num in enumerate(numeros_possiveis)}  # Mapeamento de números para índices

# Inicializa a Q-table (tabela de aprendizado)
if not os.path.exists(pasta_treino):
    os.makedirs(pasta_treino)  # Cria a pasta se não existir

if os.path.exists(arquivo_qtable):
    q_table = np.load(arquivo_qtable)  # Carrega o aprendizado salvo
    print("🚀 Aprendizado carregado com sucesso!")
else:
    q_table = np.zeros((len(numeros_possiveis), len(numeros_possiveis)))  # Cria uma nova Q-table 101x101
    print("📚 Nenhum aprendizado encontrado. Treinando do zero...")

# Parâmetros de aprendizado
learning_rate = 0.2  # Aumentado para acelerar o aprendizado
discount_factor = 0.95  # Fator de desconto ajustado
exploration_rate = 1.0  # Inicialmente, explore bastante
exploration_decay = 0.995  # Decaimento mais suave para exploração
min_exploration_rate = 0.01
episodes = 50000  # Número de tentativas de aprendizado

# Treinamento
for _ in range(episodes):
    estado = random.choice(numeros_possiveis)  # Escolhe um número aleatório como entrada do usuário
    estado_idx = indice_mapeado[estado]  # Converte o número para índice da Q-table

    if random.uniform(0, 1) < exploration_rate:
        acao = random.choice(numeros_possiveis)  # Explora (escolhe um número aleatório)
    else:
        acao_idx = np.argmax(q_table[estado_idx])  # Melhor ação aprendida
        acao = numeros_possiveis[acao_idx]  # Converte índice de volta para número real

    # Calcula recompensa
    soma = estado + acao
    diferenca = abs(100 - soma)
    recompensa = 1 / (1 + diferenca)  # Recompensa mais suave: quanto mais próximo de 100, maior a recompensa

    # Atualiza a tabela Q
    acao_idx = indice_mapeado[acao]  # Converte ação para índice
    q_table[estado_idx, acao_idx] = q_table[estado_idx, acao_idx] + learning_rate * (
        recompensa + discount_factor * np.max(q_table[acao_idx]) - q_table[estado_idx, acao_idx]
    )
    
    # Reduz a taxa de exploração
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

# Salva o aprendizado treinado
np.save(arquivo_qtable, q_table)
print("💾 Aprendizado salvo!")

# Testando a IA
while True:
    try:
        numero_usuario = int(input("\nInforme um número entre 0 e 100: "))
        if numero_usuario not in numeros_possiveis:
            print("Número fora do intervalo! Tente novamente.")
            continue

        numero_usuario_idx = indice_mapeado[numero_usuario]  # Converte número para índice
        escolha_ia_idx = np.argmax(q_table[numero_usuario_idx])  # Melhor ação aprendida
        escolha_ia = numeros_possiveis[escolha_ia_idx]  # Converte índice para número real

        print(f"A IA escolheu: {escolha_ia}")
        print(f"Total: {numero_usuario} + {escolha_ia} = {numero_usuario + escolha_ia}")

    except ValueError:
        print("Por favor, insira um número válido!")
