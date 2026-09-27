import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# --- Carpeta de resultados ----------------------------------------------------
# ------------------------------------------------------------------------------

carpeta_resultados = 'resultados/problema2'
os.makedirs(
    carpeta_resultados,
    exist_ok=True
)


# ------------------------------------------------------------------------------
# --- Respuestas correctas ------------------------------------------------------
# ------------------------------------------------------------------------------

respuestas_correctas = ['C', 'B', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']


# ------------------------------------------------------------------------------
# --- Análisis inicial sobre examen_1 -------------------------------------------
# ------------------------------------------------------------------------------

img = cv2.imread(
    'imagenes/examen_1.png',
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    print("Error al cargar la imagen")
    exit()


print("Tipo de dato:", img.dtype)
print("Dimensiones:", img.shape)
print("Valor mínimo:", img.min())
print("Valor máximo:", img.max())
print("Cantidad de niveles de intensidad:", len(np.unique(img)))


# Visualizo la imagen original

plt.figure()

plt.imshow(
    img,
    cmap='gray',
    vmin=0,
    vmax=255
)

plt.title('Examen Original')
plt.colorbar()
plt.xticks([])
plt.yticks([])

plt.savefig(
    os.path.join(
        carpeta_resultados,
        '01_examen_original.png'
    ),
    dpi=150,
    bbox_inches='tight'
)
plt.show(block=False)


# Histograma

hist, bins = np.histogram(
    img.flatten(),
    256,
    [0, 256]
)

plt.figure()

plt.subplot(121)

plt.imshow(
    img,
    cmap='gray',
    vmin=0,
    vmax=255
)

plt.title('Imagen Original')
plt.xticks([])
plt.yticks([])


plt.subplot(122)

plt.plot(
    bins[:-1],
    hist
)

plt.title('Histograma')

plt.tight_layout()
plt.savefig(
    os.path.join(
        carpeta_resultados,
        '02_histograma_examen_1.png'
    ),
    dpi=150,
    bbox_inches='tight'
)
plt.show(block=False)


# Umbralización con Otsu

T_otsu, img_th_otsu = cv2.threshold(
    img,
    thresh=127,
    maxval=255,
    type=cv2.THRESH_OTSU
)

print("Umbral de Otsu:", T_otsu)


# Fondo = 0 / tinta = 1

img_binaria = np.uint8(
    img < T_otsu
)


plt.figure()

plt.subplot(121)

plt.imshow(
    img,
    cmap='gray',
    vmin=0,
    vmax=255
)

plt.title('Imagen Original')
plt.xticks([])
plt.yticks([])


plt.subplot(122)

plt.imshow(
    img_binaria,
    cmap='gray_r',
    vmin=0,
    vmax=1
)

plt.title('Umbral Otsu: ' + str(T_otsu))
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig(
    os.path.join(
        carpeta_resultados,
        '03_umbral_otsu.png'
    ),
    dpi=150,
    bbox_inches='tight'
)
plt.show(block=False)


# Sumas por filas y columnas

suma_filas = np.sum(
    img_binaria,
    1
)

suma_columnas = np.sum(
    img_binaria,
    0
)


plt.figure()

plt.subplot(211)

plt.plot(
    suma_filas
)

plt.title('Suma por filas')


plt.subplot(212)

plt.plot(
    suma_columnas
)

plt.title('Suma por columnas')

plt.tight_layout()
plt.savefig(
    os.path.join(
        carpeta_resultados,
        '04_sumas_filas_columnas.png'
    ),
    dpi=150,
    bbox_inches='tight'
)
plt.show(block=False)


# ------------------------------------------------------------------------------
# --- Función auxiliar: buscar intervalos consecutivos --------------------------
# ------------------------------------------------------------------------------

def buscar_intervalos(vector):

    intervalos = []
    inicio = -1

    for i in range(len(vector)):

        if vector[i] and inicio == -1:
            inicio = i

        if inicio != -1:

            if not vector[i]:

                intervalos.append(
                    (inicio, i)
                )

                inicio = -1

            elif i == len(vector) - 1:

                intervalos.append(
                    (inicio, i + 1)
                )

                inicio = -1

    return intervalos


# ------------------------------------------------------------------------------
# --- Detecto las diez celdas --------------------------------------------------
# ------------------------------------------------------------------------------

def detectar_celdas(img_binaria):

    # Detecto las cuatro líneas verticales

    suma_columnas = np.sum(
        img_binaria,
        0
    )

    columnas_linea = (
        suma_columnas
        > 0.80 * suma_columnas.max()
    )

    bandas_verticales = buscar_intervalos(
        columnas_linea
    )

    if len(bandas_verticales) != 4:

        print(
            "No se detectaron correctamente "
            "los cuatro bordes verticales"
        )

        return None, None


    cajas_celdas = []
    separadores_tablas = []


    # Analizo las dos tablas

    for tabla in range(2):

        borde_izquierdo = bandas_verticales[
            2 * tabla
        ]

        borde_derecho = bandas_verticales[
            2 * tabla + 1
        ]


        # Interior de la tabla

        x0 = borde_izquierdo[1]
        x1 = borde_derecho[0]


        # Detecto las seis líneas horizontales

        suma_filas = np.sum(
            img_binaria[
                :,
                x0:x1
            ],
            1
        )

        ancho_tabla = x1 - x0

        filas_linea = (
            suma_filas
            > 0.95 * ancho_tabla
        )

        separadores = buscar_intervalos(
            filas_linea
        )

        if len(separadores) != 6:

            print(
                "No se detectaron correctamente "
                "los seis separadores horizontales"
            )

            return None, None


        separadores_tablas.append(
            separadores
        )


        # Entre seis líneas quedan cinco preguntas

        for i in range(5):

            y0 = separadores[i][1]
            y1 = separadores[i + 1][0]

            cajas_celdas.append(
                (
                    x0,
                    y0,
                    x1,
                    y1
                )
            )


    return cajas_celdas, separadores_tablas


# ------------------------------------------------------------------------------
# --- Verifico visualmente la detección de celdas -------------------------------
# ------------------------------------------------------------------------------

cajas_prueba, separadores_prueba = detectar_celdas(
    img_binaria
)

if cajas_prueba is not None:

    plt.figure()

    plt.imshow(
        img,
        cmap='gray',
        vmin=0,
        vmax=255
    )

    for x0, y0, x1, y1 in cajas_prueba:

        plt.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0]
        )

    plt.title('Celdas Detectadas')
    plt.xticks([])
    plt.yticks([])

    plt.savefig(
        os.path.join(
            carpeta_resultados,
            '05_celdas_detectadas.png'
        ),
        dpi=150,
        bbox_inches='tight'
    )
    plt.show(block=False)


# ------------------------------------------------------------------------------
# --- Visualizo las diez preguntas recortadas ----------------------------------
# ------------------------------------------------------------------------------

if cajas_prueba is not None:

    preguntas_prueba = []

    for x0, y0, x1, y1 in cajas_prueba:

        pregunta = img[
            y0:y1,
            x0:x1
        ].copy()

        preguntas_prueba.append(
            pregunta
        )


    plt.figure(figsize=(8, 10))


    for i in range(5):

        # Preguntas 1 a 5

        plt.subplot(
            5,
            2,
            2 * i + 1
        )

        plt.imshow(
            preguntas_prueba[i],
            cmap='gray',
            vmin=0,
            vmax=255
        )

        plt.title(
            'Pregunta ' + str(i + 1)
        )

        plt.xticks([])
        plt.yticks([])


        # Preguntas 6 a 10

        plt.subplot(
            5,
            2,
            2 * i + 2
        )

        plt.imshow(
            preguntas_prueba[i + 5],
            cmap='gray',
            vmin=0,
            vmax=255
        )

        plt.title(
            'Pregunta ' + str(i + 6)
        )

        plt.xticks([])
        plt.yticks([])


    plt.tight_layout()
    plt.savefig(
        os.path.join(
            carpeta_resultados,
            '06_preguntas_recortadas.png'
        ),
        dpi=150,
        bbox_inches='tight'
    )
    plt.show(block=False)


# ------------------------------------------------------------------------------
# --- Detecto la línea donde se escribe cada respuesta --------------------------
# ------------------------------------------------------------------------------

def detectar_linea_respuesta(celda):

    alto, ancho = celda.shape

    mejor_longitud = 0
    mejor_x0 = 0
    mejor_x1 = 0
    mejor_y = 0


    # La línea de respuesta está en la mitad superior de la celda

    limite = alto // 2


    for y in range(limite):

        intervalos = buscar_intervalos(
            celda[y, :] > 0
        )

        for x0, x1 in intervalos:

            longitud = x1 - x0

            if longitud > mejor_longitud:

                mejor_longitud = longitud
                mejor_x0 = x0
                mejor_x1 = x1
                mejor_y = y


    if mejor_longitud == 0:
        return None


    # Busco el espesor completo de la línea

    y0 = mejor_y
    y1 = mejor_y + 1


    while y0 > 0:

        cantidad = np.sum(
            celda[
                y0 - 1,
                mejor_x0:mejor_x1
            ]
        )

        if cantidad > 0.80 * mejor_longitud:
            y0 -= 1

        else:
            break


    while y1 < alto:

        cantidad = np.sum(
            celda[
                y1,
                mejor_x0:mejor_x1
            ]
        )

        if cantidad > 0.80 * mejor_longitud:
            y1 += 1

        else:
            break


    return (
        mejor_x0,
        y0,
        mejor_x1,
        y1
    )


# ------------------------------------------------------------------------------
# --- Extraigo las letras marcadas ---------------------------------------------
# ------------------------------------------------------------------------------

def extraer_letras_respuesta(celda):

    linea = detectar_linea_respuesta(
        celda
    )

    if linea is None:
        return []


    x0, y0, x1, y1 = linea


    # Tomo una región inmediatamente por encima de la línea

    alto_roi = (
        celda.shape[0] // 4
    )

    inicio_y = (
        y0 - alto_roi
    )

    if inicio_y < 0:
        inicio_y = 0


    roi = celda[
        inicio_y:y0,
        x0:x1
    ]

    alto_roi_real = roi.shape[0]


    # Componentes conectadas

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        roi.copy(),
        8,
        cv2.CV_32S
    )


    letras = []


    # La componente 0 corresponde al fondo

    for i in range(1, num_labels):

        x, y, w, h, area = stats[i]


        # Elimino componentes de área muy chica

        if area > 3:

            # La respuesta queda próxima a la línea.
            # El texto de la pregunta queda más arriba.

            distancia = (
                alto_roi_real
                - (y + h)
            )

            if distancia <= 0.5 * h:

                letra = (
                    labels[
                        y:y+h,
                        x:x+w
                    ] == i
                ).astype(np.uint8)

                letras.append(
                    letra
                )


    return letras


# ------------------------------------------------------------------------------
# --- Clasifico una letra como A, B, C o D -------------------------------------
# ------------------------------------------------------------------------------

def clasificar_letra(letra):

    letra_255 = (
        letra * 255
    )


    contours, hierarchy = cv2.findContours(
        letra_255,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
    )


    if hierarchy is None:
        return None


    # Cuento huecos internos

    huecos = 0


    for i in range(len(contours)):

        if hierarchy[0][i][3] != -1:
            huecos += 1


    # B tiene dos huecos

    if huecos >= 2:
        return 'B'


    # C no tiene huecos

    if huecos == 0:
        return 'C'


    # A y D tienen un hueco

    alto = letra.shape[0]

    inicio = alto // 3
    fin = (2 * alto) // 3


    # En A existe una barra horizontal que une ambos lados.
    # En D quedan dos trazos separados en la zona central.

    for y in range(
        inicio,
        fin + 1
    ):

        tramos = buscar_intervalos(
            letra[y, :] > 0
        )

        if len(tramos) == 1:
            return 'A'


    return 'D'


# ------------------------------------------------------------------------------
# --- Detecto las líneas de los campos del encabezado ---------------------------
# ------------------------------------------------------------------------------

def detectar_encabezado(img_binaria, limite_header):

    header = img_binaria[
        :limite_header,
        :
    ]


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        header,
        8,
        cv2.CV_32S
    )


    alto, ancho = img_binaria.shape

    lineas = []


    # Busco componentes largas y horizontales:
    # Name - Date - Class

    for i in range(
        1,
        num_labels
    ):

        x, y, w, h, area = stats[i]

        if (
            w > 0.08 * ancho
            and h <= 4
        ):

            lineas.append(
                (x, y, w, h)
            )


    lineas.sort()


    if len(lineas) != 3:

        print(
            "No se detectaron correctamente "
            "los campos del encabezado"
        )

        return None


    return lineas


# ------------------------------------------------------------------------------
# --- Analizo caracteres y palabras del encabezado -----------------------------
# ------------------------------------------------------------------------------

def analizar_campo(campo, factor_espacio):

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        campo,
        8,
        cv2.CV_32S
    )


    componentes = []


    for i in range(
        1,
        num_labels
    ):

        x, y, w, h, area = stats[i]


        # Elimino componentes pequeñas que puedan corresponder a ruido

        if (
            area > 2
            and h > 3
        ):

            componentes.append(
                (x, w, h)
            )


    componentes.sort()


    cantidad_caracteres = len(
        componentes
    )


    if cantidad_caracteres == 0:
        return 0, 0


    # Calculo altura promedio de los caracteres

    suma_altos = 0


    for x, w, h in componentes:
        suma_altos += h


    alto_promedio = (
        suma_altos
        / cantidad_caracteres
    )


    # Determino cantidad de palabras

    cantidad_palabras = 1


    for i in range(
        1,
        len(componentes)
    ):

        x_anterior = componentes[i - 1][0]
        w_anterior = componentes[i - 1][1]

        fin_anterior = (
            x_anterior
            + w_anterior
        )

        x_actual = componentes[i][0]

        espacio = (
            x_actual
            - fin_anterior
        )


        if espacio > factor_espacio * alto_promedio:
            cantidad_palabras += 1


    return (
        cantidad_caracteres,
        cantidad_palabras
    )


# ------------------------------------------------------------------------------
# --- Verifico visualmente los campos del encabezado ----------------------------
# ------------------------------------------------------------------------------

if separadores_prueba is not None:

    limite_header_prueba = (
        separadores_prueba[0][0][0]
    )

    if (
        separadores_prueba[1][0][0]
        < limite_header_prueba
    ):

        limite_header_prueba = (
            separadores_prueba[1][0][0]
        )


    lineas_header_prueba = detectar_encabezado(
        img_binaria,
        limite_header_prueba
    )


    if lineas_header_prueba is not None:

        nombres_campos = [
            'Name',
            'Date',
            'Class'
        ]

        plt.figure(figsize=(10, 3))


        for i in range(3):

            x, y, w, h = lineas_header_prueba[i]

            campo = img[
                0:y,
                x:x+w
            ].copy()


            plt.subplot(
                1,
                3,
                i + 1
            )

            plt.imshow(
                campo,
                cmap='gray',
                vmin=0,
                vmax=255
            )

            plt.title(
                nombres_campos[i]
            )

            plt.xticks([])
            plt.yticks([])


        plt.tight_layout()
        plt.savefig(
            os.path.join(
                carpeta_resultados,
                '07_campos_encabezado.png'
            ),
            dpi=150,
            bbox_inches='tight'
        )
        plt.show(block=False)


# ------------------------------------------------------------------------------
# --- Corrijo un examen --------------------------------------------------------
# ------------------------------------------------------------------------------

def corregir_examen(ruta):

    img = cv2.imread(
        ruta,
        cv2.IMREAD_GRAYSCALE
    )


    if img is None:

        print(
            "Error al cargar:",
            ruta
        )

        return None


    # Umbralización automática con Otsu

    T_otsu, img_th_otsu = cv2.threshold(
        img,
        thresh=127,
        maxval=255,
        type=cv2.THRESH_OTSU
    )


    # Fondo = 0
    # Tinta = 1

    img_binaria = np.uint8(
        img < T_otsu
    )


    # Detecto las celdas

    cajas_celdas, separadores_tablas = detectar_celdas(
        img_binaria
    )


    if cajas_celdas is None:
        return None


    celdas = []


    for x0, y0, x1, y1 in cajas_celdas:

        celda = img_binaria[
            y0:y1,
            x0:x1
        ].copy()

        celdas.append(
            celda
        )


    # Corrijo las diez respuestas

    cantidad_correctas = 0


    for i in range(10):

        componentes = extraer_letras_respuesta(
            celdas[i]
        )


        letras_detectadas = []


        for componente in componentes:

            letra = clasificar_letra(
                componente
            )


            if letra is not None:

                letras_detectadas.append(
                    letra
                )


        # Una respuesta es válida solamente si hay una única opción marcada

        if len(letras_detectadas) == 1:

            if (
                letras_detectadas[0]
                == respuestas_correctas[i]
            ):

                estado = "OK"
                cantidad_correctas += 1

            else:
                estado = "MAL"

        else:
            estado = "MAL"


        print(
            "Pregunta " + str(i + 1) + ": " + estado
        )


    # Detecto el encabezado

    limite_header = (
        separadores_tablas[0][0][0]
    )


    if (
        separadores_tablas[1][0][0]
        < limite_header
    ):

        limite_header = (
            separadores_tablas[1][0][0]
        )


    lineas_header = detectar_encabezado(
        img_binaria,
        limite_header
    )


    if lineas_header is None:
        return None


    campos_binarios = []
    campos_grises = []


    for x, y, w, h in lineas_header:

        campos_binarios.append(
            img_binaria[
                0:y,
                x:x+w
            ].copy()
        )


        campos_grises.append(
            img[
                0:y,
                x:x+w
            ].copy()
        )


    # Name

    # Para Name uso un factor menor porque debo detectar espacios entre palabras.
    caracteres_name, palabras_name = analizar_campo(
        campos_binarios[0],
        0.35
    )

    # Para Name también cuento los espacios entre palabras
    caracteres_name_total = (
        caracteres_name
        + palabras_name
        - 1
    )

    name_ok = (
        palabras_name >= 2
        and caracteres_name_total <= 25
    )


    # Date

    # Para Date uso un factor mayor para no interpretar las separaciones
    # alrededor de "/" como espacios entre palabras.
    caracteres_date, palabras_date = analizar_campo(
        campos_binarios[1],
        0.50
    )

    date_ok = (
        palabras_date == 1
        and caracteres_date == 8
    )


    # Class

    caracteres_class, palabras_class = analizar_campo(
        campos_binarios[2],
        0.50
    )

    class_ok = (
        caracteres_class == 1
    )


    # Muestro cantidad de caracteres y palabras

    print(
        "Name - caracteres:",
        caracteres_name_total,
        "- palabras:",
        palabras_name
    )

    print(
        "Date - caracteres:",
        caracteres_date,
        "- palabras:",
        palabras_date
    )

    print(
        "Class - caracteres:",
        caracteres_class
    )


    # Muestro estado de los campos

    if name_ok:
        print("Name: OK")
    else:
        print("Name: MAL")


    if date_ok:
        print("Date: OK")
    else:
        print("Date: MAL")


    if class_ok:
        print("Class: OK")
    else:
        print("Class: MAL")


    # Resultado final

    aprobado = (
        cantidad_correctas >= 6
    )


    if aprobado:
        resultado_texto = "APROBADO"
    else:
        resultado_texto = "DESAPROBADO"


    print(
        "Respuestas correctas:",
        cantidad_correctas
    )

    print(
        "Resultado:",
        resultado_texto
    )


    return (
        cantidad_correctas,
        aprobado,
        campos_grises[0]
    )


# ------------------------------------------------------------------------------
# --- Corrijo todos los exámenes -----------------------------------------------
# ------------------------------------------------------------------------------

resultados = []


for i in range(
    1,
    6
):

    ruta = (
        'imagenes/examen_'
        + str(i)
        + '.png'
    )


    print()
    print("----------------------------------------")
    print("EXAMEN", i)
    print("----------------------------------------")


    resultado = corregir_examen(
        ruta
    )


    if resultado is None:

        print(
            "No se pudo procesar el examen",
            i
        )

        exit()


    resultados.append(
        resultado
    )


# ------------------------------------------------------------------------------
# --- Genero imagen final con los nombres --------------------------------------
# ------------------------------------------------------------------------------

alto_total = 0
ancho_maximo = 0


for resultado in resultados:

    nombre = resultado[2]

    alto_total += (
        nombre.shape[0]
        + 10
    )


    if nombre.shape[1] > ancho_maximo:

        ancho_maximo = (
            nombre.shape[1]
        )


# Creo una imagen blanca

salida = np.ones(
    (
        alto_total,
        ancho_maximo + 10,
        3
    ),
    dtype=np.uint8
) * 255


fila = 0


for resultado in resultados:

    aprobado = resultado[1]

    nombre = resultado[2]


    nombre_color = cv2.cvtColor(
        nombre,
        cv2.COLOR_GRAY2BGR
    )


    h, w = nombre.shape


    salida[
        fila:fila+h,
        0:w
    ] = nombre_color


    # Verde = aprobado
    # Rojo = desaprobado

    if aprobado:
        color = (0, 255, 0)

    else:
        color = (0, 0, 255)


    cv2.rectangle(
        salida,
        (0, fila),
        (w - 1, fila + h - 1),
        color,
        3
    )


    fila += (
        h + 10
    )


# ------------------------------------------------------------------------------
# --- Muestro y guardo resultado final -----------------------------------------
# ------------------------------------------------------------------------------

cv2.imwrite(
    os.path.join(
        carpeta_resultados,
        'resultado_examenes.png'
    ),
    salida
)


plt.figure()

plt.imshow(
    cv2.cvtColor(
        salida,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    'Verde: aprobado - Rojo: desaprobado'
)

plt.xticks([])
plt.yticks([])

plt.savefig(
    os.path.join(
        carpeta_resultados,
        '08_resultado_examenes.png'
    ),
    dpi=150,
    bbox_inches='tight'
)
plt.show()

print("Resultados guardados en:", carpeta_resultados)
