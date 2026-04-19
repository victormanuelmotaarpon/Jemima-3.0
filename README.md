# Jemima AI v3.1 - Motor de NLP con NumPy

Jemima es una arquitectura de red neuronal densa desarrollada desde cero para el procesamiento de lenguaje natural, optimizada para hardware con recursos limitados (Intel HD Graphics 620).

## 🛠️ Interfaz de Comandos (CLI)

El sistema opera mediante una consola interactiva que permite la gestión dinámica del conocimiento:

| Comando | Acción Técnica |
| :--- | :--- |
| `enseñar:[pregunta]` | Fuerza el registro de una nueva entrada y dispara el re-entrenamiento supervisado. |
| `modificar:[pregunta]` | Sobreescribe la respuesta de una entrada existente sin generar nuevos IDs. |
| `actualizar:[pregunta]` | Concatena información adicional a una respuesta ya grabada. |
| `salir` | Serializa los pesos en `cerebro_ia.npz` y finaliza la ejecución. |

## 📂 Archivos del Proyecto

* `datos.csv`: Dataset local de interacciones.
* `cerebro_ia.npz`: Archivo de pesos entrenados (Memoria de la red).
* `historial_preguntas.txt`: Log de auditoría de preguntas del usuario.

## ⚙️ Uso

1. Instala las dependencias: `pip install numpy pandas`
2. Ejecuta el motor: `python Jemima3.3.py`

------------------------------------------------------------
   SECCIÓN DE CONTRIBUCIÓN (OPEN SOURCE)
------------------------------------------------------------
¡El desarrollo de Jemima es evolutivo! Se invita a la 
comunidad a colaborar en las siguientes áreas:

A. Optimización de Gradiente: Implementar optimizadores como 
   Adam o RMSprop para acelerar la convergencia.
B. Tokenización Avanzada: Mejorar el procesamiento de texto 
   para manejar sinónimos y lematización.
C. Capas Dropout: Añadir regularización para evitar el 
   sobreajuste (overfitting) en datasets grandes.
D. Interfaz: Desarrollo de una GUI o integración con API Flask.

Para contribuir:
1. Realiza un Fork del repositorio.
2. Crea una rama para tu mejora (git checkout -b feature/Mejora).
3. Envía un Pull Request detallando los cambios.

------------------------------------------------------------
   LICENCIA
------------------------------------------------------------
Este software es de código abierto bajo la licencia MIT. 
Puedes usarlo, modificarlo y distribuirlo siempre que se 
mantenga el crédito al autor original.
