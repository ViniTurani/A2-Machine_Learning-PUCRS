# Aprendizado de máquina - T2

Opção escolhida: Visão computacional

## Alunos:
Kristen Karsburg Arguello - 22103087  
Ramiro Nilson Barros - 22111163  
Vinícius Conte Turani - 22106859

## Como reproduzir os resultados:
1) Instalar dependências:
```sh
pip install -r requirements.txt
```

2) Treinar o modelo:
```sh
python3 model_training.py
```
| Isso gera o arquivo checkpoint.pth com os pesos finais

3) Executar a aplicação de demonstração:
```sh
python3 camera.py
```
*Atenção*: na primeira execução o sistema solicitará permissão de acesso à webcam.
Caso negue ou ainda não tenha concedido, o programa encerrará. Basta conceder o acesso e reexecutar o comando acima.

4) Usar a demo: 
- A janela da webcam será exibida.
- Posicione o rosto da pessoa dentro do quadro e pressione BARRA DE ESPAÇO para capturar o frame.
- O modelo classificará a imagem como “woman” ou “man” e mostrará o resultado na tela.
- Para encerrar a aplicação, pressione ESC.

### Extras — como reproduzir os gráficos
1. **Saídas automáticas do treinamento**  
   Ao treinar o modelo, três arquivos são gerados na pasta `apresentacao`:
   - `confusion_matrix.png` – matriz de confusão normalizada.  
   - `loss_accuracy.png` – curvas de perda e acurácia por época.  
   - `feature_maps_demo.png` – visualização dos *feature maps* das duas primeiras amostras.
2. **Inspeção das transformações de dados**  
   Para ver exemplos antes/depois do pipeline de *data augmentation*, execute:  
   ```bash
   python documenting_dataset.py
   ```
   
#### Transformações aplicadas, em ordem:
Geometria
• Rotação ±30 ° (50 %)
• Escala ±20 % (50 %)
• Shift ±10 % (50 %)
• Flip horizontal (50 %)

Oclusão e desfoque
• CoarseDropout (1 furo, 20 × 20 px, valor 128, 50 %)
• Desfoque gaussiano (3 px, 30 %)

Cor e brilho (um dos itens abaixo, 80 % de probabilidade)
• Brilho/contraste ±20 % (70 %)
• Hue ±15 °, Saturação ±25 %, Valor ±15 % (70 %)
• Channel shuffle (30 %)

Ruído
• Ruído gaussiano σ² entre 10 e 50 (30 %)

Pós-processamento
• Redimensionamento para 64 × 64
• Normalização (média = 0.5, desvio = 0.5)
• Conversão para tensor (ToTensorV2)
   

## Visão-geral rápida do projeto
- **Objetivo:** classificar imagens 64 × 64 em *woman* ou *man* com uma CNN enxuta (~2 M parâmetros).  
- **Arquitetura (`SimpleCNN`):** 4 blocos `Conv → BN → ReLU → MaxPool`, saindo de 3×64×64 até 128×4×4; depois `Flatten → Dropout → Linear → ReLU → Dropout → Linear` para logits.  
- **Augmentação:** rotações, *affine* (translate/scale/shear), flips H/V, `ColorJitter`, ruído gaussiano, *RandomErasing*; tudo seguido de `Normalize([0.5], [0.5])`.  
- **Treino:** Adam (LR = 1e-3) + `StepLR` (γ = 0.1 a cada 3 épocas), `CrossEntropyLoss`, 10 épocas padrão, *checkpoint* em `checkpoint.pth`.  
- **Avaliação:** acurácia em *test set* + matriz de confusão normalizada (salva em `apresentacao/confusion_matrix.png`).  
- **Extras:** *forward hook* salva *feature maps* do último bloco conv; script imprime média/σ do dataset e, opcionalmente, das capturas de webcam para análise de domínio.
