------------------------------------------------------------
JEMIMA NEURAL ENGINE v3.0 - DOCUMENTACIÓN
------------------------------------------------------------

DESARROLLADOR PRINCIPAL:
Victor Manuel Mota Arpon. 
Estudiante de Ingenieria en Sistemas.

------------------------------------------------------------
1. DESCRIPCIÓN DEL PROYECTO
------------------------------------------------------------
Jemima es un motor de inteligencia artificial basado en una 
red neuronal profunda (Deep Learning) programada desde cero. 
A diferencia de otros modelos que dependen de frameworks 
pesados, Jemima utiliza exclusivamente NumPy para el álgebra 
lineal y Pandas para la gestión de datos.

El proyecto nació bajo la filosofía de "Live Coding", buscando 
la máxima transparencia en el flujo de datos y la optimización 
de recursos en hardware de gama media (como GPUs integradas 
Intel HD 620).

------------------------------------------------------------
2. ARQUITECTURA TÉCNICA
------------------------------------------------------------
* Red Neuronal: Multicapa (Input, 2 Capas Ocultas, Output).
* Activación: Función Sigmoide con Backpropagation.
* Inicialización: Pesos Xavier/Glorot para estabilidad.
* Clasificación: Multietiqueta con umbral de confianza > 0.6.
* Optimización: Sistema de sobreescritura de etiquetas para 
  mantener la red compacta y eficiente.

------------------------------------------------------------
3. ARCHIVOS DEL SISTEMA
------------------------------------------------------------
- Jemima3.0.py: Lógica de la red y bucle de interacción.
- datos.csv: Dataset dinámico (Preguntas/Respuestas/IDs).
- cerebro_ia.npz: Almacenamiento binario de pesos sinápticos.
- historial_preguntas.txt: Log único de consultas del usuario.

------------------------------------------------------------
4. COMANDOS DE CONTROL (CLI)
------------------------------------------------------------
Jemima permite la manipulación del conocimiento en tiempo real:

- actualizar: [pregunta] -> Añade información a la respuesta 
  existente (concatenación).
- modificar: [pregunta] -> Sobreescribe la respuesta actual. 
  Ideal para corregir errores sin aumentar el tamaño del modelo.
- salir/exit -> Guarda el estado y cierra el programa.

------------------------------------------------------------
5. SECCIÓN DE CONTRIBUCIÓN (OPEN SOURCE)
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
6. LICENCIA
------------------------------------------------------------
Este software es de código abierto bajo la licencia MIT. 
Puedes usarlo, modificarlo y distribuirlo siempre que se 
mantenga el crédito al autor original.
