import os
import re
import cv2
import pytesseract
import pandas as pd
from decimal import Decimal, InvalidOperation

# ==============================================================================
#  CONFIGURACIÓN GLOBAL Y CONSTANTES
# ==============================================================================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

DIRECTORIO_ENTRADA = 'facturas'
DIRECTORIO_SALIDA = 'reporte'

# ==============================================================================
#  FUNCIONES DE PROCESAMIENTO DE IMAGEN Y DATOS
# ==============================================================================

def optimizar_imagen_para_ocr(path_imagen):
    """
    Carga una imagen, la convierte a escala de grises y aplica un umbral
    adaptativo para maximizar la precisión del reconocimiento de texto.
    """
    imagen_original = cv2.imread(path_imagen)
    if imagen_original is None:
        return None
    
    imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
    
    imagen_binarizada = cv2.adaptiveThreshold(
        imagen_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    return imagen_binarizada

def convertir_texto_a_decimal(cadena_valor):
    """
    Convierte un string de monto (con posibles comas y puntos) a un objeto Decimal.
    Ej: "1.234,56" -> Decimal('1234.56'), "78.90" -> Decimal('78.90')
    """
    # Reemplaza coma por punto para el decimal y elimina puntos de miles
    monto_str_normalizado = cadena_valor.replace('.', '').replace(',', '.')
    partes = monto_str_normalizado.split('.')
    if len(partes) > 2: # Maneja casos como "1.234.56"
        valor_normalizado = "".join(partes[:-1]) + "." + partes[-1]
    else:
        valor_normalizado = monto_str_normalizado
    try:
        return Decimal(valor_normalizado)
    except (InvalidOperation, TypeError):
        return None

def analizar_estructura_factura(imagen_optimizada):
    """
    Realiza el OCR para obtener texto y su posición, reconstruye la estructura
    de la factura (tabla de ítems y total) y extrae los datos relevantes.
    """
    # Configuración para Tesseract
    config_tesseract = r'--oem 3 --psm 6 -l spa'
    
    datos_ocr = pytesseract.image_to_data(imagen_optimizada, config=config_tesseract, output_type=pytesseract.Output.DICT)

    # 1. Filtrar y estructurar los resultados del OCR
    elementos_texto = []
    for i in range(len(datos_ocr['text'])):
        text = datos_ocr['text'][i].strip()
        confianza = int(datos_ocr['conf'][i])
        if text and confianza > 60:
            elementos_texto.append({
                'text': text,
                'left': datos_ocr['left'][i],
                'top': datos_ocr['top'][i],
                'width': datos_ocr['width'][i],
                'height': datos_ocr['height'][i]
            })

    # 2. Reconstruir líneas de texto a partir de palabras sueltas
    elementos_texto.sort(key=lambda p: (p['top'], p['left']))
    filas_texto = []
    if elementos_texto:
        fila_en_proceso = [elementos_texto[0]]
        for i in range(1, len(elementos_texto)):
            palabra_anterior = fila_en_proceso[-1]
            palabra_actual = elementos_texto[i]
            
            if abs(palabra_actual['top'] - palabra_anterior['top']) < 20: # Umbral de proximidad vertical
                fila_en_proceso.append(palabra_actual)
            else:
                filas_texto.append(sorted(fila_en_proceso, key=lambda p: p['left']))
                fila_en_proceso = [palabra_actual]
        filas_texto.append(sorted(fila_en_proceso, key=lambda p: p['left']))

    # 3. Analizar las líneas reconstruidas para extraer datos
    items_factura = []
    total_extraido = None
    cabeceras_tabla = {}
    fase_analisis = 'buscando_cabeceras'  # Fases: buscando_cabeceras, extrayendo_items, finalizado

    DICCIONARIO_CABECERAS = {
        'Cant': ['cant', 'cantidad', 'qty'],
        'Descripción': ['descripción', 'descripcion', 'producto', 'servicio', 'concepto'],
        'P.Unit': ['p.unit', 'unitario', 'precio', 'p/u'],
        'Importe': ['importe', 'total', 'subtotal', 'valor']
    }

    for fila in filas_texto:
        texto_fila = ' '.join(p['text'] for p in fila).lower()

        # --- Fase 1: Identificar las cabeceras de la tabla de ítems ---
        if fase_analisis == 'buscando_cabeceras':
            cabeceras_encontradas = False
            for nombre_col, alias_list in DICCIONARIO_CABECERAS.items():
                for alias in alias_list:
                    if alias in texto_fila:
                        for palabra in fila:
                            if alias in palabra['text'].lower():
                                cabeceras_tabla[nombre_col] = palabra['left']
                                cabeceras_encontradas = True
                                break
            if cabeceras_encontradas:
                print(f"Cabeceras detectadas (pos x): {cabeceras_tabla}")
                fase_analisis = 'extrayendo_items'
                continue

        # --- Fase de búsqueda del total (se puede encontrar antes o después de los ítems) ---
        if 'total' in texto_fila and fase_analisis != 'buscando_cabeceras':
            montos_en_linea = [convertir_texto_a_decimal(p['text']) for p in fila if convertir_texto_a_decimal(p['text']) is not None]
            if montos_en_linea:
                total_extraido = montos_en_linea[-1]
                fase_analisis = 'finalizado' # Una vez encontrado el total, se asume que no hay más ítems
                continue

        # --- Fase 2: Extraer las filas de ítems ---
        if fase_analisis == 'extrayendo_items' and cabeceras_tabla:
            # Ignorar si la línea parece ser otra cabecera
            if any(alias in texto_fila for col_aliases in DICCIONARIO_CABECERAS.values() for alias in col_aliases):
                continue

            # Asignar cada palabra de la fila a una cabecera por proximidad horizontal
            item_actual = {nombre_col: [] for nombre_col in cabeceras_tabla}
            for palabra in fila:
                distancias = {nombre_col: abs(palabra['left'] - pos_col) for nombre_col, pos_col in cabeceras_tabla.items()}
                columna_cercana = min(distancias, key=distancias.get)
                item_actual[columna_cercana].append(palabra['text'])

            # Consolidar y validar el ítem extraído
            try:
                desc = ' '.join(item_actual.get('Descripción', []))
                cant_str = item_actual.get('Cant', [None])[0]
                punit_str = item_actual.get('P.Unit', [None])[0]
                importe_str = item_actual.get('Importe', [None])[0]

                # Un ítem es válido si tiene descripción y un importe
                if importe_str and desc:
                    importe = convertir_texto_a_decimal(importe_str)
                    if importe is not None:
                        items_factura.append({
                            'Cant': int(float(cant_str.replace(',', '.'))) if cant_str else 1,
                            'Descripción': desc,
                            'P.Unit': convertir_texto_a_decimal(punit_str) if punit_str else importe,
                            'Importe': importe
                        })
            except (ValueError, InvalidOperation, IndexError):
                pass # Ignorar filas que no se puedan convertir a un ítem válido

    # 4. Calcular la suma de los ítems para validación
    suma_items = sum(d['Importe'] for d in items_factura) if items_factura else Decimal('0')
    
    return items_factura, total_extraido, suma_items

# ==============================================================================
#  ORQUESTADOR PRINCIPAL
# ==============================================================================

def ejecutar_procesamiento_masivo():
    """
    Orquesta el proceso completo: lee imágenes de un directorio, las procesa
    una por una y genera un reporte final con los resultados.
    """
    print(">>> Iniciando el motor de reconocimiento de facturas <<<")
    
    if not os.path.exists(DIRECTORIO_ENTRADA):
        print(f"Error: El directorio de entrada '{DIRECTORIO_ENTRADA}' no fue encontrado.")
        return

    if not os.path.exists(DIRECTORIO_SALIDA):
        os.makedirs(DIRECTORIO_SALIDA)

    resumen_procesamiento = []
    lista_imagenes = [f for f in os.listdir(DIRECTORIO_ENTRADA) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

    for nombre_fichero in lista_imagenes:
        path_completo_imagen = os.path.join(DIRECTORIO_ENTRADA, nombre_fichero)
        print(f"\n--- Procesando archivo: {nombre_fichero} ---")

        imagen_optimizada = optimizar_imagen_para_ocr(path_completo_imagen)
        if imagen_optimizada is None:
            print(f"Error al leer el archivo de imagen.")
            resumen_procesamiento.append({
                'Archivo': nombre_fichero, 'Total Calculado': 'N/A', 'Total Factura': 'N/A',
                'Coherente': 'No', 'Error': 'No se pudo leer la imagen'
            })
            continue

        items_factura, total_extraido, suma_items = analizar_estructura_factura(imagen_optimizada)
        
        if not items_factura or total_extraido is None:
            print("Análisis fallido: No se extrajeron suficientes datos estructurados.")
            resumen_procesamiento.append({
                'Archivo': nombre_fichero, 'Total Calculado': 'N/A', 'Total Factura': 'N/A',
                'Coherente': 'No', 'Error': 'Sin detalles'
            })
            continue

        # Validación de consistencia
        es_coherente = (suma_items == total_extraido)

        print("Ítems extraídos de la factura:")
        if items_factura:
            tabla_items = pd.DataFrame(items_factura)
            print(tabla_items.to_string(index=False))
        else:
            print("No se encontraron ítems.")
        
        print("\n--- Resumen de Validación ---")
        print(f"Suma de ítems: {suma_items:.2f}")
        print(f"Total en factura: {total_extraido:.2f}")
        print(f"Consistencia: {'VÁLIDA' if es_coherente else 'INVÁLIDA'}")

        resumen_procesamiento.append({
            'Archivo': nombre_fichero,
            'Total Calculado': f"{suma_items:.2f}",
            'Total Factura': f"{total_extraido:.2f}",
            'Coherente': 'Sí' if es_coherente else 'No',
            'Error': ''
        })

    # Generación del reporte final
    if resumen_procesamiento:
        reporte_final_df = pd.DataFrame(resumen_procesamiento)
        path_reporte_csv = os.path.join(DIRECTORIO_SALIDA, 'reporte_facturas.csv')
        reporte_final_df.to_csv(path_reporte_csv, index=False)
        print(f"\n>>> Proceso finalizado. Reporte guardado en: {path_reporte_csv} <<<")
    else:
        print("\nNo se encontraron imágenes en el directorio de entrada.")

if __name__ == '__main__':
    ejecutar_procesamiento_masivo()
