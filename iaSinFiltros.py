import json
import random
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from transformers import pipeline
from nltk.corpus import stopwords

# Descargar datos necesarios para NLTK (solo una vez)
def descargar_recursos_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Descargando recursos de NLTK...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')

descargar_recursos_nltk()

# Cargar el chat desde el archivo JSON o texto plano
def cargar_chat(ruta_archivo):
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as file:
            # Intenta cargar como JSON
            chat_historial = json.load(file)
            print("✅ Archivo cargado como JSON.")
            return chat_historial
    except json.JSONDecodeError:
        # Si falla, carga como texto plano
        with open(ruta_archivo, "r", encoding="utf-8") as file:
            chat_historial = file.readlines()
            chat_historial = [line.strip() for line in chat_historial if line.strip()]
            print("✅ Archivo cargado como texto plano.")
            return chat_historial
    except Exception as e:
        print(f"❌ Error al cargar el archivo: {e}")
        return []

# Filtrar mensajes de Panecito
def filtrar_mensajes_panecito(chat):
    mensajes_panecito = []
    for mensaje in chat:
        if "Panecito ♥️:" in mensaje:
            # Extraer solo el texto del mensaje (eliminar el nombre y la hora)
            texto_mensaje = mensaje.split("Panecito ♥️:")[1].strip()
            mensajes_panecito.append(texto_mensaje)
    return mensajes_panecito

# 1️⃣ Análisis del chat
def analizar_chat(chat):
    palabras = []
    stop_words = set(stopwords.words('spanish'))  # Filtra stopwords en español
    for mensaje in chat:
        palabras.extend([palabra for palabra in word_tokenize(mensaje.lower()) if palabra.isalnum() and palabra not in stop_words])
    
    # Contar frecuencia de palabras
    frecuencia = Counter(palabras)
    print("📊 Palabras más usadas en el chat:", frecuencia.most_common(5))
    
    return frecuencia

# 2️⃣ Generar una respuesta aleatoria basada en patrones del chat
def respuesta_similar(chat, pregunta):
    pregunta_palabras = [palabra for palabra in word_tokenize(pregunta.lower()) if palabra.isalnum()]
    respuestas_candidatas = [mensaje for mensaje in chat if any(palabra in mensaje.lower() for palabra in pregunta_palabras)]
    
    if respuestas_candidatas:
        return random.choice(respuestas_candidatas)
    else:
        return random.choice(chat)  # Si no hay coincidencias, elige una respuesta aleatoria

# 3️⃣ Usar IA para generar una respuesta más avanzada
try:
    chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")  # Usar un modelo más pequeño para desarrollo
except Exception as e:
    print("Error al cargar el modelo de IA:", e)
    chatbot = None

def respuesta_ia(mensaje):
    if chatbot:
        try:
            respuesta = chatbot(
                mensaje,
                max_length=50,
                num_return_sequences=1,
                temperature=0.9,  # Más creatividad
                top_k=50,         # Más diversidad
                top_p=0.9,        # Más diversidad
                do_sample=True    # Muestreo aleatorio
            )
            return respuesta[0]["generated_text"]
        except Exception as e:
            return f"Error al generar respuesta: {e}"
    else:
        return "El modelo de IA no está disponible."

# 💬 Simulación del bot
ruta_archivo = "chat.json"  # Cambia esto si el archivo tiene otro nombre
chat_historial = cargar_chat(ruta_archivo)

if chat_historial:
    # Filtrar solo los mensajes de Panecito
    mensajes_panecito = filtrar_mensajes_panecito(chat_historial)
    if not mensajes_panecito:
        print("❌ No se encontraron mensajes de Panecito en el chat.")
    else:
        analizar_chat(mensajes_panecito)
        while True:
            pregunta = input("\nTú: ")  # Usuario ingresa una pregunta
            if pregunta.lower() in ["salir", "exit", "adiós"]:
                print("Bot: ¡Adiós! 👋")
                break
            print("\nMiamor (simple):", respuesta_similar(mensajes_panecito, pregunta))
            print("Bot (IA):", respuesta_ia(pregunta))
else:
    print("No se pudo cargar el historial de chat.")