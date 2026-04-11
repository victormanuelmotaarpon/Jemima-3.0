import numpy as np
import pandas as pd
import re
import os
from datetime import datetime

# --- 1. CONFIGURACIÓN DE ARCHIVOS ---
CSV_FILE = 'datos.csv'
WEIGHTS_FILE = 'cerebro_ia.npz'
LOG_FILE = 'historial_preguntas.txt'

# --- 2. UTILIDADES DE PROCESAMIENTO Y LOGS ---
def limpiar_texto(texto):
    """Limpia el texto para estandarizar la entrada del usuario."""
    texto = str(texto).lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto.strip()

def registrar_pregunta_log(pregunta):
    """Almacena la pregunta en un .txt. Si ya existe, sobreescribe su registro."""
    preguntas_dict = {}
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for linea in f:
                if " -> " in linea:
                    partes = linea.split(" -> ", 1)
                    meta = partes[0].split("] ", 1)
                    if len(meta) > 1:
                        preguntas_dict[meta[1].strip()] = meta[0] + "]"

    fecha_hoy = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    preguntas_dict[pregunta.strip()] = f"[{fecha_hoy}]"

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        for preg, fecha in preguntas_dict.items():
            f.write(f"{fecha} {preg} -> Registrada\n")

def modificar_respuesta_directa(pregunta_objetivo, nueva_respuesta):
    """Sobreescribe la respuesta en el dataset. Optimiza recursos al no crear IDs nuevos."""
    if not os.path.exists(CSV_FILE): return False
    
    df = pd.read_csv(CSV_FILE)
    pregunta_busqueda = limpiar_texto(pregunta_objetivo)
    mask = df['pregunta'].apply(limpiar_texto) == pregunta_busqueda
    
    if mask.any():
        df.loc[mask, 'respuesta_texto'] = nueva_respuesta.strip()
        df.to_csv(CSV_FILE, index=False)
        return True
    return False

def actualizar_conocimiento_existente(pregunta_objetivo, nueva_info):
    """Concatena información a la respuesta existente."""
    if not os.path.exists(CSV_FILE): return False
    df = pd.read_csv(CSV_FILE)
    pregunta_busqueda = limpiar_texto(pregunta_objetivo)
    mask = df['pregunta'].apply(limpiar_texto) == pregunta_busqueda
    
    if mask.any():
        texto_anterior = df.loc[mask, 'respuesta_texto'].values[0]
        if nueva_info not in str(texto_anterior):
            df.loc[mask, 'respuesta_texto'] = f"{texto_anterior} {nueva_info}".strip()
            df.to_csv(CSV_FILE, index=False)
            return True
    return False

# --- 3. ARQUITECTURA DE LA RED NEURONAL ---
class RedNeuronalConciencia:
    def __init__(self, capas, lr=0.1):
        self.capas = capas
        self.lr = lr
        self.pesos = [np.random.randn(capas[i], capas[i+1]) * np.sqrt(2/capas[i]) for i in range(len(capas)-1)]
        self.biases = [np.zeros((1, capas[i+1])) for i in range(len(capas)-1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_der(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        activaciones = [X]
        for i in range(len(self.pesos)):
            net = np.dot(activaciones[-1], self.pesos[i]) + self.biases[i]
            activaciones.append(self.sigmoid(net))
        return activaciones

    def entrenar(self, X, y, epocas=20000):
        print(f">> Entrenando Jemima por {epocas} épocas...")
        for epoca in range(epocas):
            activaciones = self.feedforward(X)
            error = y - activaciones[-1]
            delta = error * self.sigmoid_der(activaciones[-1])
            for i in reversed(range(len(self.pesos))):
                self.pesos[i] += np.dot(activaciones[i].T, delta) * self.lr
                self.biases[i] += np.sum(delta, axis=0, keepdims=True) * self.lr
                delta = np.dot(delta, self.pesos[i].T) * self.sigmoid_der(activaciones[i])
        self.guardar_pesos()

    def guardar_pesos(self):
        dict_guardar = {f'p_{i}': self.pesos[i] for i in range(len(self.pesos))}
        dict_guardar.update({f'b_{i}': self.biases[i] for i in range(len(self.biases))})
        np.savez(WEIGHTS_FILE, **dict_guardar)

    def cargar_pesos(self):
        if os.path.exists(WEIGHTS_FILE):
            data = np.load(WEIGHTS_FILE)
            try:
                self.pesos = [data[f'p_{i}'] for i in range(len(self.capas) - 1)]
                self.biases = [data[f'b_{i}'] for i in range(len(self.capas) - 1)]
                return True
            except: return False
        return False

    def predecir(self, X):
        return self.feedforward(X)[-1]

# --- 4. GESTIÓN DE DATOS ---
def cargar_y_reparar_datos():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=['pregunta', 'respuesta_texto', 'id'])
        df.loc[0] = ["hola", "¡Hola! Soy Jemima.", 0]
        df.to_csv(CSV_FILE, index=False)
        return df
    return pd.read_csv(CSV_FILE)

def procesar_matrices(df):
    preguntas_limpias = [limpiar_texto(p) for p in df['pregunta']]
    vocab = sorted(list(set(" ".join(preguntas_limpias).split())))
    X = [[1 if p in frase.split() else 0 for p in vocab] for frase in preguntas_limpias]
    num_respuestas = int(df['id'].max() + 1)
    y = np.zeros((len(df), num_respuestas))
    for i, row in df.iterrows():
        y[i, int(row['id'])] = 1
    return np.array(X), y, vocab, num_respuestas

# --- 5. INICIALIZACIÓN ---
df = cargar_y_reparar_datos()
respuestas_texto = {int(row['id']): row['respuesta_texto'] for _, row in df.iterrows()}
X, y, vocabulario, total_ids = procesar_matrices(df)

ia = RedNeuronalConciencia(capas=[len(vocabulario), 32, 16, total_ids])
if not ia.cargar_pesos() or ia.pesos[0].shape[0] != len(vocabulario):
    ia.entrenar(X, y, epocas=30000)

# --- 6. BUCLE PRINCIPAL ---
print("\n" + "="*45)
print(" JEMIMA AI v3.0: Optimización de Recursos ")
print(" 'actualizar:' para sumar | 'modificar:' para cambiar ")
print("="*45)

while True:
    user_input = input("\nTú: ").strip()
    if user_input.lower() in ['salir', 'exit', 'quit']: break
    
    registrar_pregunta_log(user_input)

    # NUEVO COMANDO: MODIFICAR (Sobreescribe para ahorrar recursos)
    if user_input.lower().startswith("modificar:"):
        target_pregunta = user_input.split(":", 1)[1].strip()
        nueva_respuesta = input(f"Nueva respuesta para '{target_pregunta}': ").strip()
        
        if modificar_respuesta_directa(target_pregunta, nueva_respuesta):
            print(">> Modificando dataset y re-sincronizando...")
            df = cargar_y_reparar_datos()
            respuestas_texto = {int(row['id']): row['respuesta_texto'] for _, row in df.iterrows()}
            # No hace falta re-entrenar 30k épocas si solo cambió el texto de la etiqueta
            print(">> Respuesta modificada. Jemima responderá lo nuevo ahora.")
        else:
            print("IA: No encontré esa pregunta para modificar.")
        continue

    # COMANDO: ACTUALIZAR (Sumar información)
    if user_input.lower().startswith("actualizar:"):
        target_pregunta = user_input.split(":", 1)[1].strip()
        extra_info = input("Añadir info extra: ").strip()
        if actualizar_conocimiento_existente(target_pregunta, extra_info):
            df = cargar_y_reparar_datos()
            respuestas_texto = {int(row['id']): row['respuesta_texto'] for _, row in df.iterrows()}
            print(">> Información enriquecida exitosamente.")
        continue

    # PREDICCIÓN NORMAL
    entrada_limpia = limpiar_texto(user_input)
    vector_test = np.array([1 if p in entrada_limpia.split() else 0 for p in vocabulario]).reshape(1, -1)
    
    if np.sum(vector_test) == 0:
        print("IA: No comprendo esas palabras nuevas. ¿Qué respondo?")
        nueva_res = input("Respuesta: ")
        nuevo_id = int(df['id'].max() + 1)
        nueva_fila = pd.DataFrame([[user_input, nueva_res, nuevo_id]], columns=['pregunta', 'respuesta_texto', 'id'])
        nueva_fila.to_csv(CSV_FILE, mode='a', header=False, index=False)
        
        # Re-entrenamiento necesario por nuevo vocabulario e ID
        df = cargar_y_reparar_datos()
        respuestas_texto = {int(row['id']): row['respuesta_texto'] for _, row in df.iterrows()}
        X, y, vocabulario, total_ids = procesar_matrices(df)
        ia = RedNeuronalConciencia(capas=[len(vocabulario), 32, 16, total_ids])
        ia.entrenar(X, y, epocas=20000)
        continue

    predicciones = ia.predecir(vector_test)[0]
    finales = [respuestas_texto[i] for i, prob in enumerate(predicciones) if prob > 0.6]
    
    if finales:
        print(f"IA: {' '.join(finales)}")
    else:
        print("IA: No estoy segura. Usa 'modificar:[pregunta]' si quieres que cambie mi respuesta.")