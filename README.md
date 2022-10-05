# codigo-photopletismografia

## Descrição

Este repositório contém o código para estimação da variabilidade da frequência cardíaca de um sujeito a partir do vídeo da face do mesmo. O código executa a detecção de face e olhos e, a partir destes, estima a região da testa, que é a região selecionada para a fotopletismografia.

O código ainda está em fase de implementação.

## Requisitos

* [OpenCV](https://opencv.org/)
* [Numpy](https://numpy.org/)
* [EasyGUI](http://easygui.sourceforge.net/)

Os requisitos podem ser instalados via PIP usando o arquivo [requirements.txt](requirements.txt).

## Uso

A execução padrão do código, sem parâmetros, abre uma janela para seleção do vídeo onde será feita a fotopletismografia. Em sistemas Unix ele pode ser executado como script.

### Parâmetros

* `-c` ou `--captura`: realiza a captura do vídeo usando a webcam. O vídeo será armazenado na mesma pasta do código e terá a duração de 20 segundos.
