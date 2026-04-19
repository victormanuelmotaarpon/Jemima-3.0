import numpy as np
import pandas as pd
import re
import os
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
CSV_FILE = 'datos.csv'
WEIGHTS_FILE = 'cerebro_ia.npz'
LOG_FILE = 'historial_preguntas.txt'

# --- 2. UTILIDADES DE TEXTO ---
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-zñáéíóú\s]', '', texto)
    return texto.strip()

def registrar_pregunta_log(pregunta):
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

def actualizar_conocimiento_existente(pregunta_objetivo, nueva_info):
    """Agrega información extra a una respuesta existente sin borrar la anterior."""
    if not os.path.exists(CSV_FILE): return False
    df = pd.read_csv(CSV_FILE)
    mask = df['pregunta'].apply(limpiar_texto) == limpiar_texto(pregunta_objetivo)
    if mask.any():
        texto_anterior = df.loc[mask, 'respuesta_texto'].values[0]
        df.loc[mask, 'respuesta_texto'] = f"{texto_anterior} {nueva_info}".strip()
        df.to_csv(CSV_FILE, index=False)
        return True
    return False

# --- 3. ARQUITECTURA DE LA RED NEURONAL ---
class RedNeuronalConciencia:
    def __init__(self, capas, lr=0.01):
        self.capas = capas
        self.lr = lr
        self.pesos = [np.random.randn(capas[i], capas[i+1]) * np.sqrt(1/capas[i]) for i in range(len(capas)-1)]
        self.biases = [np.zeros((1, capas[i+1])) for i in range(len(capas)-1)]

    def relu(self, x): return np.maximum(0, x)
    def relu_der(self, x): return (x > 0).astype(float)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def feedforward(self, X):
        activaciones = [X]
        for i in range(len(self.pesos) - 1):
            net = np.dot(activaciones[-1], self.pesos[i]) + self.biases[i]
            activaciones.append(self.relu(net))
        net_final = np.dot(activaciones[-1], self.pesos[-1]) + self.biases[-1]
        activaciones.append(self.softmax(net_final))
        return activaciones

    def entrenar(self, X, y, epocas=50000):
        print(f">> Calibrando cerebro ({X.shape[1]} neuronas de entrada)...")
        for epoca in range(epocas):
            activaciones = self.feedforward(X)
            error = y - activaciones[-1]
            delta = error 
            for i in reversed(range(len(self.pesos))):
                gradiente = np.dot(activaciones[i].T, delta)
                self.pesos[i] += gradiente * self.lr
                self.biases[i] += np.sum(delta, axis=0, keepdims=True) * self.lr
                if i > 0:
                    delta = np.dot(delta, self.pesos[i].T) * self.relu_der(activaciones[i])
        self.guardar_pesos()

    def guardar_pesos(self):
        dict_guardar = {f'p_{i}': self.pesos[i] for i in range(len(self.pesos))}
        dict_guardar.update({f'b_{i}': self.biases[i] for i in range(len(self.biases))})
        np.savez(WEIGHTS_FILE, **dict_guardar)

    def cargar_pesos(self, esperado_in):
        if os.path.exists(WEIGHTS_FILE):
            data = np.load(WEIGHTS_FILE)
            if data['p_0'].shape[0] == esperado_in:
                self.pesos = [data[f'p_{i}'] for i in range(len(self.capas) - 1)]
                self.biases = [data[f'b_{i}'] for i in range(len(self.capas) - 1)]
                return True
        return False

    def predecir(self, X):
        return self.feedforward(X)[-1]

# --- 4. GESTIÓN DE MATRICES Y DATOS ---
def generar_embedding(frase, vocabulario, id_contexto, total_ids):
    palabras = frase.split()
    vector_palabras = [palabras.count(p) for p in vocabulario]
    vector_contexto = [0] * total_ids
    if id_contexto is not None and id_contexto < total_ids:
        vector_contexto[id_contexto] = 2.5 
    return vector_palabras + vector_contexto

def cargar_y_preparar():
    if not os.path.exists(CSV_FILE):
        pd.DataFrame([["hola", "¡Hola! Soy Jemima.", 0]], columns=['pregunta', 'respuesta_texto', 'id']).to_csv(CSV_FILE, index=False)
    
    df = pd.read_csv(CSV_FILE)
    preguntas_limpias = [limpiar_texto(p) for p in df['pregunta']]
    vocab = sorted(list(set(" ".join(preguntas_limpias).split())))
    total_ids = int(df['id'].max() + 1)
    
    X = [generar_embedding(p, vocab, None, total_ids) for p in preguntas_limpias]
    y = np.zeros((len(df), total_ids))
    for i, row in df.iterrows():
        y[i, int(row['id'])] = 1
        
    respuestas = {int(row['id']): row['respuesta_texto'] for _, row in df.iterrows()}
    return df, np.array(X), y, vocab, total_ids, respuestas

# --- 5. INICIALIZACIÓN ---
df, X, y, vocabulario, total_ids, respuestas_texto = cargar_y_preparar()
n_entrada = len(vocabulario) + total_ids
ia = RedNeuronalConciencia(capas=[n_entrada, 128, 64, total_ids])

if not ia.cargar_pesos(n_entrada):
    ia.entrenar(X, y)

# --- 6. BUCLE PRINCIPAL ---
print("\n" + "="*60)
print(" JEMIMA AI v3.3: SISTEMA DE DIFERENCIACIÓN Y ACTUALIZACIÓN ")
print("-" * 60)
print(" COMANDOS DISPONIBLES:")
print("  • enseñar: [pregunta]    -> Crea una respuesta nueva (ID nuevo).")
print("  • modificar: [pregunta]  -> Sobrescribe una respuesta existente.")
print("  • actualizar: [pregunta] -> Añade información a la respuesta actual.")
print("  • salir                  -> Cierra la aplicación.")
print("="*60)

ultimo_id = 0

while True:
    user_input = input("\nTú: ").strip()
    if user_input.lower() in ['salir', 'exit']: break
    
    # COMANDO ENSEÑAR
    if user_input.lower().startswith("enseñar:"):
        frase_novedad = user_input.split(":", 1)[1].strip()
        print(f"IA: Entendido. ¿Qué debo responder a '{frase_novedad}'?")
        nueva_res = input("Respuesta: ")
        nuevo_id = int(df['id'].max() + 1)
        pd.DataFrame([[frase_novedad, nueva_res, nuevo_id]], columns=['pregunta', 'respuesta_texto', 'id']).to_csv(CSV_FILE, mode='a', header=False, index=False)
        df, X, y, vocabulario, total_ids, respuestas_texto = cargar_y_preparar()
        ia = RedNeuronalConciencia(capas=[len(vocabulario) + total_ids, 128, 64, total_ids])
        ia.entrenar(X, y, epocas=40000)
        continue

    # COMANDO MODIFICAR
    if user_input.lower().startswith("modificar:"):
        target = user_input.split(":", 1)[1].strip()
        nueva = input(f"Nueva respuesta: ")
        df_mod = pd.read_csv(CSV_FILE)
        mask = df_mod['pregunta'].apply(limpiar_texto) == limpiar_texto(target)
        if mask.any():
            df_mod.loc[mask, 'respuesta_texto'] = nueva
            df_mod.to_csv(CSV_FILE, index=False)
            _, _, _, _, _, respuestas_texto = cargar_y_preparar()
            print(">> Respuesta sobrescrita correctamente.")
        continue

    # COMANDO ACTUALIZAR
    if user_input.lower().startswith("actualizar:"):
        target = user_input.split(":", 1)[1].strip()
        info_extra = input(f"Información adicional para '{target}': ")
        if actualizar_conocimiento_existente(target, info_extra):
            _, _, _, _, _, respuestas_texto = cargar_y_preparar()
            print(">> Conocimiento enriquecido con éxito.")
        else:
            print(">> No se encontró esa pregunta exacta para actualizar.")
        continue

    # PROCESAMIENTO ESTÁNDAR
    registrar_pregunta_log(user_input)
    entrada_limpia = limpiar_texto(user_input)
    
    if not any(p in vocabulario for p in entrada_limpia.split()):
        print("IA: No reconozco esas palabras. ¿Qué debería responder?")
        res = input("Respuesta: ")
        nid = int(df['id'].max() + 1)
        pd.DataFrame([[user_input, res, nid]], columns=['pregunta', 'respuesta_texto', 'id']).to_csv(CSV_FILE, mode='a', header=False, index=False)
        df, X, y, vocabulario, total_ids, respuestas_texto = cargar_y_preparar()
        ia = RedNeuronalConciencia(capas=[len(vocabulario) + total_ids, 128, 64, total_ids])
        ia.entrenar(X, y)
        continue

    v_final = np.array(generar_embedding(entrada_limpia, vocabulario, ultimo_id, total_ids)).reshape(1, -1)
    predicciones = ia.predecir(v_final)[0]
    id_ganador = np.argmax(predicciones)
    confianza = predicciones[id_ganador]

    if confianza > 0.45:
        print(f"IA: {respuestas_texto[id_ganador]}")
        ultimo_id = id_ganador
    else:
        print(f"IA: Lo siento, me confundo. (Confianza: {confianza*100:.1f}%)")
        print("Sugerencias:")
        print("  - Usar 'enseñar: [pregunta]' para una respuesta nueva.")
        print("  - Usar 'actualizar: [pregunta]' para añadir datos a una existente.")
        ultimo_id = None