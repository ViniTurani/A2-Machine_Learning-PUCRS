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