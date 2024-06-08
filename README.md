# prueba-conteo-malcon
Como harías para resolver este problema, puedes usar el modelo que desees y el lenguaje debe ser python. Según un video cualquiera de carreteras por donde pasen coches tienes que contar el numero de coches que pasan y presetarlo visualmente la inferencia de como cuenta los coches, el input seria un video que quieras y output una ventana donde muestre el video y como cuenta, al mismo tiempo tambien puedes mostrar logs en consola de como va contando por consola.


# Propuesta de Solución - Contador de Coches

Este es un proyecto en Python que utiliza OpenCV para contar el número de coches que pasan por una franja en un video. Utiliza técnicas de detección de objetos para identificar los coches en cada fotograma del video y realiza un conteo de los bounding boxes cuyo centroide se ubica en la franja considerando un salto de frames determinado.

## Funcionalidades

- Detecta coches en un video.
- Cuenta el número de coches que pasan por una línea definida en el video.
- Proporciona visualización en tiempo real del conteo de coches en el video.

## Requisitos

- Python 3.6 o superior
- OpenCV
- NumPy

## Uso

1. Instalar los requerimientos.
2. Ejecuta el script principal "main.py".
3. Para detener la ejecución, presionar la tecla 'Enter'.

### Autor
Malcon Mora