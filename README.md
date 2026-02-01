# PyTorch Image Classification with Pretrained Models

## 📋 Descrição

Este projeto implementa classificação de imagens utilizando PyTorch e modelos pré-treinados. O foco é demonstrar como utilizar transfer learning para classificação de lesões em folhas de feijão.

## 🎯 Objetivo

O projeto tem como objetivo classificar imagens de folhas de feijão em três categorias:
- **Angular Leaf Spot** (Mancha Angular)
- **Bean Rust** (Ferrugem)
- **Healthy** (Saudável)

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **PyTorch** - Framework de Deep Learning
- **torchvision** - Modelos pré-treinados e transformações
- **scikit-learn** - Pré-processamento de dados
- **Matplotlib** - Visualização
- **Pandas** - Manipulação de dados
- **PIL (Pillow)** - Processamento de imagens

## 📊 Dataset

O dataset utilizado é o **Bean Leaf Lesions Classification** disponível no Kaggle:
- Total de imagens: 1,167
- Classes: 3 (balanceadas)
- Divisão: 70% treino / 30% teste

## 🏗️ Estrutura do Projeto

```
Image_classification/
│
├── pytorch_image.ipynb          # Notebook principal
├── README.md                     # Este arquivo
├── requirements.txt              # Dependências
└── data/                        # Diretório de dados (não incluído)
```

## 🚀 Como Usar

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/DaniloBlancoMotta/Pytorch_Image_Classification_Pretrained.ipynb.git
cd Pytorch_Image_Classification_Pretrained.ipynb
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Execução

1. Abra o notebook no Jupyter:
```bash
jupyter notebook pytorch_image.ipynb
```

2. Execute as células sequencialmente

## 📝 Workflow do Notebook

1. **Inicialização e Download do Dataset**
   - Download do dataset do Kaggle usando `opendatasets`

2. **Imports**
   - Importação de bibliotecas necessárias
   - Configuração do dispositivo (GPU/CPU)

3. **Leitura dos Dados**
   - Carregamento dos CSVs de treino e validação
   - Concatenação dos dados

4. **Inspeção dos Dados**
   - Análise das classes
   - Distribuição dos dados

5. **Divisão dos Dados**
   - Split 70/30 para treino/teste

6. **Pré-processamento**
   - Criação do LabelEncoder
   - Definição de transformações (resize, normalização)

7. **Dataset Customizado**
   - Implementação de classe CustomImageDataset

8. **Visualização**
   - Exibição de amostras de imagens

9. **Treinamento do Modelo**
   - Utilização de modelos pré-treinados (Transfer Learning)

## 🎓 Conceitos Aprendidos

- **Transfer Learning**: Utilização de modelos pré-treinados
- **Data Augmentation**: Transformações para aumentar dados
- **Custom Datasets**: Criação de datasets personalizados no PyTorch
- **GPU Acceleration**: Uso de CUDA para acelerar treinamento

## 📈 Resultados

Os resultados variam de acordo com o modelo utilizado e hiperparâmetros. O notebook demonstra o processo completo de treinamento e avaliação.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Adicionar novos modelos
- Melhorar documentação

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 👤 Autor

**Danilo Blanco Motta**

- GitHub: [@DaniloBlancoMotta](https://github.com/DaniloBlancoMotta)

## 🙏 Agradecimentos

- Dataset disponibilizado no Kaggle
- Comunidade PyTorch
- Documentação oficial do torchvision

---

⭐ Se este projeto foi útil para você, considere dar uma estrela!
