import cv2
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# --- Cargo imagen -------------------------------------------------------------
# ------------------------------------------------------------------------------

img = cv2.imread(
    'imagenes/Imagen_con_detalles_escondidos.tif',
    cv2.IMREAD_GRAYSCALE
)

if img is None:
    print("Error al cargar la imagen")
    exit()


# ------------------------------------------------------------------------------
# --- Analizo imagen -----------------------------------------------------------
# ------------------------------------------------------------------------------

print("Tipo de dato:", img.dtype)
print("Dimensiones:", img.shape)
print("Valor mínimo:", img.min())
print("Valor máximo:", img.max())


# Muestro imagen original

plt.figure()
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen Original')
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show(block=False)


# ------------------------------------------------------------------------------
# --- Histograma de la imagen original -----------------------------------------
# ------------------------------------------------------------------------------

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

plt.figure()

plt.subplot(121)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.plot(bins[:-1], hist)
plt.title('Histograma')

plt.tight_layout()
plt.show(block=False)


# ------------------------------------------------------------------------------
# --- Ecualización global ------------------------------------------------------
# ------------------------------------------------------------------------------

img_heq = cv2.equalizeHist(img)

plt.figure()

ax1 = plt.subplot(221)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen Original')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.hist(img.flatten(), 256, [0, 256])
plt.title('Histograma Original')

plt.subplot(223, sharex=ax1, sharey=ax1)
plt.imshow(img_heq, cmap='gray', vmin=0, vmax=255)
plt.title('Ecualización Global')
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.hist(img_heq.flatten(), 256, [0, 256])
plt.title('Histograma Ecualizado')

plt.tight_layout()
plt.show(block=False)


# ------------------------------------------------------------------------------
# --- Función de ecualización local --------------------------------------------
# ------------------------------------------------------------------------------

def ecualizacion_local(img, M, N):

    # La ventana debe tener un pixel central
    if M % 2 == 0 or N % 2 == 0:
        print("El tamaño de la ventana debe ser impar")
        return None

    # Cantidad de pixels a agregar alrededor de la imagen
    borde_filas = M // 2
    borde_columnas = N // 2

    # Agrego borde para poder procesar los pixels de los extremos
    img_borde = cv2.copyMakeBorder(
        img,
        borde_filas,
        borde_filas,
        borde_columnas,
        borde_columnas,
        cv2.BORDER_REPLICATE
    )

    # Creo imagen de salida
    img_salida = np.zeros(
        img.shape,
        dtype=np.uint8
    )

    # Recorro todos los pixels de la imagen original
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            # Obtengo la ventana M x N alrededor del pixel
            ventana = img_borde[
                i:i+M,
                j:j+N
            ]

            # Calculo el histograma de la ventana
            hist, _ = np.histogram(
                ventana.flatten(),
                256,
                [0, 256]
            )

            # Normalizo el histograma
            histn = hist.astype(np.double) / ventana.size

            # Calculo la CDF
            cdf = histn.cumsum()

            # Obtengo el valor del pixel central
            pixel_central = ventana[
                borde_filas,
                borde_columnas
            ]

            # Aplico la transformación de ecualización
            img_salida[i, j] = np.uint8(
                np.round(255 * cdf[pixel_central])
            )

    return img_salida


# ------------------------------------------------------------------------------
# --- Analizo influencia del tamaño de la ventana -------------------------------
# ------------------------------------------------------------------------------

img_local_7 = ecualizacion_local(img, 7, 7)
img_local_15 = ecualizacion_local(img, 15, 15)
img_local_31 = ecualizacion_local(img, 31, 31)
img_local_63 = ecualizacion_local(img, 63, 63)


# ------------------------------------------------------------------------------
# --- Comparo los diferentes tamaños de ventana --------------------------------
# ------------------------------------------------------------------------------

plt.figure()

plt.subplot(221)
plt.imshow(img_local_7, cmap='gray', vmin=0, vmax=255)
plt.title('Ventana 7 x 7')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.imshow(img_local_15, cmap='gray', vmin=0, vmax=255)
plt.title('Ventana 15 x 15')
plt.xticks([])
plt.yticks([])

plt.subplot(223)
plt.imshow(img_local_31, cmap='gray', vmin=0, vmax=255)
plt.title('Ventana 31 x 31')
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.imshow(img_local_63, cmap='gray', vmin=0, vmax=255)
plt.title('Ventana 63 x 63')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show(block=False)


# ------------------------------------------------------------------------------
# --- Comparación ecualización global vs local ---------------------------------
# ------------------------------------------------------------------------------

plt.figure()

ax1 = plt.subplot(131)

plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(132, sharex=ax1, sharey=ax1)

plt.imshow(img_heq, cmap='gray', vmin=0, vmax=255)
plt.title('Ecualización Global')
plt.xticks([])
plt.yticks([])

plt.subplot(133, sharex=ax1, sharey=ax1)

plt.imshow(img_local_31, cmap='gray', vmin=0, vmax=255)
plt.title('Ecualización Local 31 x 31')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# --- Guardo resultados ---------------------------------------------------------
# ------------------------------------------------------------------------------

cv2.imwrite(
    'resultado_ecualizacion_local_7x7.png',
    img_local_7
)

cv2.imwrite(
    'resultado_ecualizacion_local_15x15.png',
    img_local_15
)

cv2.imwrite(
    'resultado_ecualizacion_local_31x31.png',
    img_local_31
)

cv2.imwrite(
    'resultado_ecualizacion_local_63x63.png',
    img_local_63
)