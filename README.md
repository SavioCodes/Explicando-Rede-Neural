# 🧠 Construindo uma Rede Neural do Zero em Python

> **Por [Savio](https://github.com/SavioCodes)** - Desenvolvedor apaixonado por IA e Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-SavioCodes-blue?style=flat-square&logo=github)](https://github.com/SavioCodes)
[![Python](https://img.shields.io/badge/Python-3.8+-green?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## 📋 Índice

- [🎯 Introdução](#-introdução)
- [🔧 Conceitos Fundamentais](#-conceitos-fundamentais)
- [💻 Implementação do Zero](#-implementação-do-zero)
- [🎯 Exemplo Prático: XOR](#-exemplo-prático-resolvendo-xor)
- [⚡ Comparação com Frameworks](#-comparação-com-frameworks)
- [🚀 Aplicações Reais](#-aplicações-reais)
- [🎓 Conclusão](#-conclusão-e-próximos-passos)
- [📚 Referências](#-referências)

---

Bem-vindos ao meu repositório onde descomplicamos redes neurais! 🚀

Se você já se perguntou como funciona "a mágica" por trás do machine learning e quer entender na prática, sem depender de bibliotecas prontas, chegou ao lugar certo.

Durante minha jornada como desenvolvedor, percebi que a maioria dos programadores usa TensorFlow ou PyTorch sem realmente entender o que acontece por baixo dos panos. Criar uma rede neural do zero mudou completamente minha perspectiva sobre inteligência artificial - e espero que faça o mesmo com você.

## 🎯 Introdução

### O que é uma Rede Neural?

Uma rede neural é basicamente um sistema computacional inspirado no funcionamento do cérebro humano. Imagine milhares de neurônios conectados entre si, processando informações e tomando decisões. Na versão artificial, temos nós (neurônios) organizados em camadas que transformam dados de entrada em resultados úteis.

A beleza está na simplicidade: cada neurônio recebe sinais, processa essas informações usando operações matemáticas simples, e passa o resultado adiante. Quando você multiplica isso por centenas ou milhares de neurônios, surge um comportamento emergente capaz de reconhecer padrões complexos.

### 📜 Um Pouco de História

As redes neurais não são uma invenção recente:

- **1943**: McCulloch e Pitts criam o primeiro modelo matemático de neurônio
- **1958**: Rosenblatt desenvolve o Perceptron, primeiro algoritmo de aprendizado
- **1986**: Rumelhart populariza o algoritmo de backpropagation
- **2000s**: Revolução do deep learning com aumento do poder computacional
- **2012**: AlexNet vence ImageNet, marcando era moderna da IA

### 🌍 Onde São Usadas Hoje?

Essas redes estão por trás de praticamente tudo que consideramos "inteligente" na tecnologia:

| Área | Exemplos |
|------|----------|
| **🖼️ Visão Computacional** | Reconhecimento facial, carros autônomos, diagnóstico médico |
| **💬 Processamento de Linguagem** | ChatGPT, tradutores, assistentes virtuais |
| **🎮 Jogos** | AlphaGo, OpenAI Five, bots inteligentes |
| **💰 Finanças** | Detecção de fraudes, trading algorítmico |
| **🏥 Medicina** | Diagnóstico por imagens, descoberta de medicamentos |

---

## 🔧 Conceitos Fundamentais

### 🧬 Neurônio Artificial vs Biológico

| Neurônio Biológico | Neurônio Artificial |
|-------------------|-------------------|
| Dendritos (recebem sinais) | Inputs (x₁, x₂, x₃...) |
| Corpo celular (processa) | Soma ponderada + Bias |
| Axônio (envia sinal) | Função de ativação → Output |

```
Inputs → Pesos → Soma Ponderada → Função de Ativação → Output
(x₁,x₂,x₃) → (w₁,w₂,w₃) → Σ(xi*wi) + b → f(z) → y
```

### 🏗️ Arquitetura em Camadas

```
Input Layer    Hidden Layer(s)    Output Layer
    x₁  ────────── h₁ ──────────── y₁
    x₂  ────────── h₂ ──────────── y₂
    x₃  ────────── h₃ ──────────── ...
    ...           ...
```

- **Camada de Entrada**: recebe dados brutos
- **Camadas Ocultas**: extraem características e padrões
- **Camada de Saída**: produz resultado final

> 💡 **Dica**: Mais camadas = mais capacidade de aprender padrões complexos, mas também mais risco de overfitting!

### ⚡ Funções de Ativação

#### 1. **Sigmoid** - A Clássica
```python
σ(x) = 1 / (1 + e^(-x))
```
- ✅ Saída entre 0 e 1 (boa para probabilidades)
- ❌ Gradient vanishing em redes profundas

#### 2. **ReLU** - A Mais Popular
```python
f(x) = max(0, x)
```
- ✅ Simples e eficiente computacionalmente
- ✅ Resolve gradient vanishing
- ❌ Neurônios podem "morrer" (sempre zero)

#### 3. **Tanh** - A Centrada
```python
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- ✅ Saída entre -1 e 1
- ✅ Centrada em zero (melhor que sigmoid)

#### 4. **Softmax** - Para Classificação
```python
softmax(xi) = e^xi / Σ(e^xj)
```
- ✅ Converte logits em probabilidades que somam 1
- ✅ Ideal para classificação multi-classe

### ➡️ Forward Propagation

O processo onde dados fluem da entrada para a saída:

```python
# Pseudocódigo simplificado
def forward_pass(input_data):
    current_input = input_data
    
    for layer in neural_network:
        # 1. Multiplica inputs pelos pesos
        weighted_sum = np.dot(current_input, weights) + bias
        
        # 2. Aplica função de ativação
        layer_output = activation_function(weighted_sum)
        
        # 3. Output vira input da próxima camada
        current_input = layer_output
    
    return final_output
```

### ⬅️ Backpropagation: A Mágica do Aprendizado

Aqui está o coração do aprendizado! Backpropagation usa cálculo diferencial para descobrir como cada peso contribuiu para o erro final.

**Como funciona:**
1. 📊 Calcula erro na saída
2. 🔄 Propaga erro para camadas anteriores (regra da cadeia)
3. 📈 Calcula gradientes para cada peso
4. 🔧 Atualiza pesos na direção oposta ao gradiente

```python
# Conceito matemático
∂Error/∂weight = ∂Error/∂output × ∂output/∂weight
```

### 📉 Gradiente Descendente

Imagine que você está numa montanha com vendas nos olhos e quer chegar ao vale (menor erro):

```python
# Fórmula básica
weight_new = weight_old - learning_rate × gradient
```

**Learning Rate é crucial:**
- 🔴 **Muito alto**: você "pula" o mínimo
- 🟡 **Muito baixo**: demora eternidade para convergir  
- 🟢 **Ideal**: converge suavemente para solução ótima

### 🎯 Overfitting vs Underfitting

| Problema | Descrição | Solução |
|----------|-----------|---------|
| **Overfitting** | Rede "decora" dados de treino, não generaliza | Dropout, regularização, mais dados |
| **Underfitting** | Rede muito simples, não aprende padrões | Mais camadas, mais neurônios, treinar mais |

**Técnicas de Regularização:**
- **Dropout**: desliga neurônios aleatoriamente durante treino
- **L1/L2**: penaliza pesos muito grandes
- **Early Stopping**: para quando validação para de melhorar

---

## 💻 Implementação do Zero

Agora vamos sujar as mãos! Nossa implementação usa apenas NumPy - nada de TensorFlow ou PyTorch aqui. 🔥

### 🏗️ Estrutura Básica da Classe

```python
import numpy as np
import matplotlib.pyplot as plt

class RedeNeuralDoZero:
    def __init__(self, arquitetura):
        """
        arquitetura: lista com neurônios por camada
        Ex: [2, 4, 1] = 2 inputs, 4 hidden, 1 output
        """
        self.arquitetura = arquitetura
        self.num_camadas = len(arquitetura)
        
        # Inicializa pesos e bias
        self.pesos = []
        self.bias = []
        
        # Xavier initialization - funciona melhor que random puro
        for i in range(1, self.num_camadas):
            w = np.random.randn(arquitetura[i-1], arquitetura[i]) * np.sqrt(2.0 / arquitetura[i-1])
            b = np.zeros((1, arquitetura[i]))
            
            self.pesos.append(w)
            self.bias.append(b)
        
        print(f"🧠 Rede criada: {' → '.join(map(str, arquitetura))}")
```

### ⚡ Funções de Ativação

```python
def relu(self, x):
    """ReLU: f(x) = max(0, x)"""
    return np.maximum(0, x)

def relu_derivada(self, x):
    """Derivada da ReLU"""
    return (x > 0).astype(float)

def sigmoid(self, x):
    """Sigmoid: σ(x) = 1/(1 + e^(-x))"""
    # Clip para evitar overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(self, x):
    """Derivada da Sigmoid"""
    s = self.sigmoid(x)
    return s * (1 - s)

def softmax(self, x):
    """Softmax para classificação multi-classe"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

### ➡️ Forward Propagation

```python
def forward(self, X):
    """
    Propaga dados pela rede (entrada → saída)
    
    Args:
        X: matriz de inputs (amostras × features)
    
    Returns:
        ativacoes: lista com saídas de cada camada
        z_valores: lista com valores antes da ativação
    """
    ativacoes = [X]  # Guarda ativações de cada camada
    z_valores = []   # Guarda valores antes da ativação
    
    entrada_atual = X
    
    for i in range(len(self.pesos)):
        # 1. Calcula soma ponderada: z = X·W + b
        z = np.dot(entrada_atual, self.pesos[i]) + self.bias[i]
        z_valores.append(z)
        
        # 2. Aplica função de ativação
        if i < len(self.pesos) - 1:  # Camadas ocultas
            ativacao = self.relu(z)
        else:  # Camada de saída
            ativacao = self.sigmoid(z)
            
        ativacoes.append(ativacao)
        entrada_atual = ativacao
    
    return ativacoes, z_valores
```

### ⬅️ Backpropagation - Onde a Mágica Acontece

```python
def backward(self, X, y, ativacoes, z_valores):
    """
    Calcula gradientes usando backpropagation
    
    Esta é a parte mais importante! Aqui calculamos como
    cada peso contribuiu para o erro final.
    """
    m = X.shape[0]  # número de amostras
    
    # Listas para guardar gradientes
    dW = [np.zeros_like(w) for w in self.pesos]
    db = [np.zeros_like(b) for b in self.bias]
    
    # 1. Erro na camada de saída
    delta = ativacoes[-1] - y  # Para MSE
    
    # 2. Propaga erro para trás (backpropagation)
    for i in range(len(self.pesos) - 1, -1, -1):
        # Gradientes para pesos e bias desta camada
        dW[i] = np.dot(ativacoes[i].T, delta) / m
        db[i] = np.mean(delta, axis=0, keepdims=True)
        
        # Se não é a primeira camada, calcula delta para camada anterior
        if i > 0:
            # Propaga erro: delta_anterior = delta_atual · W^T · f'(z)
            delta = np.dot(delta, self.pesos[i].T) * self.relu_derivada(z_valores[i-1])
    
    return dW, db

def atualizar_pesos(self, dW, db, learning_rate):
    """Atualiza pesos e bias usando gradientes calculados"""
    for i in range(len(self.pesos)):
        self.pesos[i] -= learning_rate * dW[i]
        self.bias[i] -= learning_rate * db[i]
```

### 📊 Funções de Custo e Métricas

```python
def calcular_custo(self, y_pred, y_true):
    """Mean Squared Error (MSE)"""
    return np.mean((y_pred - y_true) ** 2)

def calcular_acuracia(self, y_pred, y_true):
    """Acurácia para problemas de classificação"""
    predicoes = (y_pred > 0.5).astype(int)
    return np.mean(predicoes == y_true)

def treinar(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
    """
    Treina a rede neural
    
    Args:
        X: dados de entrada
        y: rótulos verdadeiros
        epochs: número de épocas
        learning_rate: taxa de aprendizado
        verbose: mostrar progresso
    """
    custos = []
    
    for epoch in range(epochs):
        # Forward pass
        ativacoes, z_valores = self.forward(X)
        
        # Calcula custo
        custo = self.calcular_custo(ativacoes[-1], y)
        custos.append(custo)
        
        # Backward pass
        dW, db = self.backward(X, y, ativacoes, z_valores)
        
        # Atualiza pesos
        self.atualizar_pesos(dW, db, learning_rate)
        
        # Mostra progresso
        if verbose and epoch % (epochs // 10) == 0:
            acuracia = self.calcular_acuracia(ativacoes[-1], y)
            print(f"Época {epoch:4d}: Custo = {custo:.6f}, Acurácia = {acuracia:.2%}")
    
    return custos
```

---

## 🎯 Exemplo Prático: Resolvendo XOR

O problema XOR é um clássico! Um perceptron simples não consegue resolvê-lo (não é linearmente separável), mas uma rede com camada oculta sim! 🎯

### 📊 Dataset XOR

```python
def exemplo_xor():
    """
    Problema XOR: saída é 1 quando inputs são diferentes
    
    Tabela verdade:
    0 XOR 0 = 0
    0 XOR 1 = 1  
    1 XOR 0 = 1
    1 XOR 1 = 0
    """
    
    # Dataset
    X = np.array([[0, 0],
                  [0, 1], 
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1], 
                  [0]])
    
    print("📊 Dataset XOR:")
    print("Input → Output")
    for i in range(len(X)):
        print(f"{X[i]} → {y[i][0]}")
    
    return X, y
```

### 🚀 Treinamento Completo

```python
def treinar_xor():
    """Exemplo completo de treinamento para XOR"""
    
    # 1. Prepara dados
    X, y = exemplo_xor()
    
    # 2. Cria rede: 2 inputs → 4 hidden → 1 output
    rede = RedeNeuralDoZero([2, 4, 1])
    
    # 3. Treina
    print("\n🚀 Iniciando treinamento...")
    custos = rede.treinar(X, y, epochs=5000, learning_rate=0.1)
    
    # 4. Testa resultado final
    ativacoes_finais, _ = rede.forward(X)
    predicoes = ativacoes_finais[-1]
    
    print("\n🎯 Resultados finais:")
    print("Input → Predição (Esperado)")
    for i in range(len(X)):
        pred = predicoes[i][0]
        esperado = y[i][0]
        status = "✅" if abs(pred - esperado) < 0.1 else "❌"
        print(f"{X[i]} → {pred:.4f} ({esperado}) {status}")
    
    # 5. Plota curva de aprendizado
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(custos, 'b-', linewidth=2)
    plt.title('📉 Curva de Aprendizado - XOR', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('Custo (MSE)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(predicoes)), predicoes, c='red', s=100, label='Predições', alpha=0.7)
    plt.scatter(range(len(y)), y, c='blue', s=100, label='Esperado', alpha=0.7)
    plt.title('🎯 Predições vs Esperado', fontsize=14)
    plt.xlabel('Amostra')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return rede

# 🚀 Executa o exemplo
if __name__ == "__main__":
    rede_treinada = treinar_xor()
```

### 📈 Resultado Esperado

Após o treinamento, você deve ver algo como:

```
🧠 Rede criada: 2 → 4 → 1

📊 Dataset XOR:
Input → Output
[0 0] → 0
[0 1] → 1
[1 0] → 1
[1 1] → 0

🚀 Iniciando treinamento...
Época    0: Custo = 0.289156, Acurácia = 25.00%
Época  500: Custo = 0.044521, Acurácia = 100.00%
Época 1000: Custo = 0.012334, Acurácia = 100.00%
Época 1500: Custo = 0.006789, Acurácia = 100.00%
Época 2000: Custo = 0.004512, Acurácia = 100.00%

🎯 Resultados finais:
Input → Predição (Esperado)
[0 0] → 0.0123 (0) ✅
[0 1] → 0.9876 (1) ✅
[1 0] → 0.9891 (1) ✅
[1 1] → 0.0109 (0) ✅
```

🎉 **Sucesso!** A rede aprendeu perfeitamente a função XOR!

---

## ⚡ Comparação com Frameworks

Agora vamos ver como a mesma rede ficaria em frameworks populares:

### 🔥 PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.camada1 = nn.Linear(2, 4)
        self.camada2 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = torch.relu(self.camada1(x))
        x = torch.sigmoid(self.camada2(x))
        return x

# Uso
modelo = XORNet()
criterio = nn.MSELoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.01)

# Treinamento em poucas linhas
for epoch in range(5000):
    otimizador.zero_grad()
    saidas = modelo(X_tensor)
    perda = criterio(saidas, y_tensor)
    perda.backward()
    otimizador.step()
```

### 🧠 TensorFlow/Keras

```python
import tensorflow as tf

modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modelo.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
modelo.fit(X, y, epochs=5000, verbose=0)
```

### 📊 Comparação Detalhada

| Aspecto | **Nossa Implementação** | **PyTorch/TensorFlow** |
|---------|------------------------|----------------------|
| **🎓 Aprendizado** | ✅ Entendimento profundo | ❌ Abstração pode esconder detalhes |
| **🔧 Controle** | ✅ Controle total sobre cada operação | ❌ Menos flexibilidade para experimentos |
| **⚡ Performance** | ❌ Mais lento, sem GPU | ✅ Otimizado, GPU automática |
| **🐛 Debugging** | ✅ Fácil debugar cada passo | ❌ Mais difícil debugar internamente |
| **📝 Código** | ❌ Mais verboso | ✅ Mais conciso |
| **🚀 Produção** | ❌ Não recomendado | ✅ Pronto para produção |

> 💡 **Minha recomendação**: Aprenda primeiro do zero (como aqui), depois use frameworks para projetos reais!

---

## 🚀 Aplicações Reais

### 🖼️ Visão Computacional

**Redes Convolucionais (CNNs)** revolucionaram processamento de imagens:

```python
# Conceito de CNN para reconhecimento de dígitos
class CNNSimples:
    def __init__(self):
        # Camadas convolucionais extraem características locais
        self.conv_layers = [
            CamadaConv(filtros=32, kernel=3),
            CamadaPooling(pool_size=2),
            CamadaConv(filtros=64, kernel=3),
            CamadaPooling(pool_size=2)
        ]
        
        # Camadas densas fazem classificação final
        self.dense_layers = [
            CamadaDensa(128, ativacao='relu'),
            CamadaDensa(10, ativacao='softmax')  # 10 classes (0-9)
        ]
```

**🎯 Casos de uso:**
- 🏥 Diagnóstico médico por imagens
- 👤 Reconhecimento facial
- 🚗 Carros autônomos
- 🏭 Controle de qualidade industrial

### 💬 Processamento de Linguagem Natural (NLP)

**Transformers e LSTMs** processam sequências de texto:

```python
# Conceito de LSTM para análise de sentimento
class AnalisadorSentimento:
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=128):
        self.embedding = CamadaEmbedding(vocab_size, embedding_dim)
        self.lstm = CamadaLSTM(hidden_size)
        self.classificador = CamadaDensa(1, ativacao='sigmoid')
    
    def forward(self, sequencia_texto):
        embedded = self.embedding(sequencia_texto)
        lstm_out = self.lstm(embedded)
        sentimento = self.classificador(lstm_out[-1])  # Última saída
        return sentimento
```

**🎯 Aplicações:**
- 🌐 Tradução automática (Google Translate)
- 🤖 Chatbots e assistentes virtuais
- 📱 Análise de sentimentos em redes sociais
- 📄 Sumarização automática de textos

### 📈 Séries Temporais

**Previsão de valores futuros** baseado em histórico:

```python
# Rede para prever preço de ações
class PrevisaoAcoes:
    def __init__(self):
        # LSTM para capturar padrões temporais
        self.lstm_layers = [
            CamadaLSTM(50, return_sequences=True),
            CamadaLSTM(50),
            CamadaDensa(25),
            CamadaDensa(1)  # Preço previsto
        ]
    
    def prever_proximo_dia(self, historico_precos):
        # historico_precos: últimos 60 dias
        return self.forward(historico_precos)
```

**🎯 Casos de uso:**
- 💰 Previsão financeira
- ⚡ Demanda de energia
- 🌤️ Previsão do tempo
- 🔧 Manutenção preditiva

### 🎮 Jogos e Reinforcement Learning

**AlphaGo e OpenAI Five** usam redes neurais para jogar:

```python
# Conceito de rede para jogo da velha
class JogadorVelha:
    def __init__(self):
        # Input: estado do tabuleiro 3x3 = 9 posições
        # Output: valor de cada posição possível
        self.rede = RedeNeuralDoZero([9, 128, 64, 9])
    
    def escolher_jogada(self, estado_tabuleiro):
        valores_jogadas = self.forward(estado_tabuleiro)
        jogadas_validas = self.obter_jogadas_validas(estado_tabuleiro)
        return jogadas_validas[np.argmax(valores_jogadas[jogadas_validas])]
```

---

## 🎓 Conclusão e Próximos Passos

Parabéns! 🎉 Se chegou até aqui, agora você entende como funciona o coração da inteligência artificial moderna. 

### 🧠 O que Você Aprendeu

✅ **Fundamentos sólidos**: neurônios, camadas, ativações, gradientes  
✅ **Matemática por trás**: forward pass, backpropagation, otimização  
✅ **Implementação prática**: código funcional sem bibliotecas mágicas  
✅ **Intuição**: por que as coisas funcionam (ou não funcionam)  

### ⚠️ Limitações da Nossa Implementação

Nossa rede é educacional, mas tem limitações para uso real:
- ❌ Sem otimizações de performance (GPU, vectorização avançada)
- ❌ Funções de ativação limitadas
- ❌ Sem técnicas modernas (batch normalization, residual connections)
- ❌ Sem regularização avançada

### 🚀 Próximos Desafios

1. **🖼️ CNNs**: Implemente redes convolucionais do zero
2. **📝 RNNs/LSTMs**: Crie redes recorrentes para sequências
3. **🎨 GANs**: Desenvolva redes adversárias para gerar imagens
4. **🎮 RL**: Experimente com Reinforcement Learning

### 🤝 Contribua!

Encontrou algum bug? Tem sugestões? Quer adicionar exemplos?

[![Contribuir](https://img.shields.io/badge/Contribuir-GitHub-green?style=for-the-badge&logo=github)](https://github.com/SavioCodes/Explicando-Rede-Neural)

---

## 📚 Referências

### 📖 Livros Essenciais
- **"Deep Learning"** - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- **"Neural Networks and Deep Learning"** - Michael Nielsen (online, gratuito)
- **"Hands-On Machine Learning"** - Aurélien Géron
- **"Pattern Recognition and Machine Learning"** - Christopher Bishop

### 📄 Papers Fundamentais
- **"A Learning Algorithm for Continually Running Fully Recurrent Neural Networks"** - Williams & Zipser (1989)
- **"Attention Is All You Need"** - Vaswani et al. (2017)
- **"Deep Residual Learning for Image Recognition"** - He et al. (2016)
- **"ImageNet Classification with Deep Convolutional Neural Networks"** - Krizhevsky et al. (2012)

### 🎓 Cursos Online
- **CS231n (Stanford)** - Computer Vision
- **CS224n (Stanford)** - Natural Language Processing  
- **Deep Learning Specialization (Coursera)** - Andrew Ng
- **Fast.ai** - Practical Deep Learning

### 🔗 Links Úteis
- [Neural Networks and Deep Learning (online book)](http://neuralnetworksanddeeplearning.com/)
- [Distill.pub - Visual explanations](https://distill.pub/)
- [Papers With Code](https://paperswithcode.com/)
- [Towards Data Science](https://towardsdatascience.com/)

---

## 👨‍💻 Sobre o Autor

**Savio** - Desenvolvedor apaixonado por IA e Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-SavioCodes-blue?style=flat-square&logo=github)](https://github.com/SavioCodes)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/savio)

> *"A melhor forma de aprender é ensinando, e a melhor forma de entender é implementando."*

---

<div align="center">

### ⭐ Se este projeto te ajudou, deixe uma estrela!

[![Star](https://img.shields.io/github/stars/SavioCodes/Explicando-Rede-Neural?style=social)](https://github.com/SavioCodes/Explicando-Rede-Neural)

**Feito com ❤️ por [Savio](https://github.com/SavioCodes)**

</div>
