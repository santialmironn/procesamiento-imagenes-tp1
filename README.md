# Procesamiento de Imágenes — Trabajo Práctico N°1

Universidad Austral — LCD — Año 2026

**Integrantes:** *Almiron Nanni, Santino - Borrás, Justo - Tejerina, Wenceslao*

Resolución de los dos problemas del TP N°1:

- Ecualización local de histograma.
- Corrección automática de exámenes multiple choice.

---

## Contenido del repositorio

```text
.
├── README.md
├── requirements.txt
├── informe.pdf
├── Problema1.py
├── Problema2.py
├── imagenes/
│   ├── Imagen_con_detalles_escondidos.tif
│   ├── examen_1.png
│   ├── examen_2.png
│   ├── examen_3.png
│   ├── examen_4.png
│   └── examen_5.png
└── resultados/
    ├── problema1/
    └── problema2/
```

La carpeta `resultados/` se crea automáticamente al ejecutar los scripts.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/santialmironn/procesamiento-imagenes-tp1.git
```

Ingresar a la carpeta del proyecto:

```bash
cd procesamiento-imagenes-tp1
```

### 2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecución

Los scripts deben ejecutarse desde la raíz del repositorio, ya que utilizan rutas relativas para acceder a la carpeta `imagenes/`.

### Problema 1 — Ecualización local de histograma

Ejecutar:

```bash
python Problema1.py
```

El script procesa:

```text
imagenes/Imagen_con_detalles_escondidos.tif
```

Se realiza el análisis de la imagen, la ecualización global y la ecualización local utilizando distintos tamaños de ventana:

```text
7 x 7
15 x 15
31 x 31
63 x 63
```

Los resultados se guardan automáticamente en:

```text
resultados/problema1/
```

La ecualización local recorre la imagen pixel a pixel, por lo que la ejecución puede tardar algunos segundos.

---

### Problema 2 — Corrección automática de multiple choice

Ejecutar:

```bash
python Problema2.py
```

El script procesa automáticamente:

```text
imagenes/examen_1.png
imagenes/examen_2.png
imagenes/examen_3.png
imagenes/examen_4.png
imagenes/examen_5.png
```

Por consola se informa, para cada examen:

- estado de cada una de las 10 preguntas;
- validación de los campos `Name`, `Date` y `Class`;
- cantidad de respuestas correctas;
- resultado final.

Los resultados y figuras generadas se guardan automáticamente en:

```text
resultados/problema2/
```

La imagen final muestra los nombres de los alumnos con:

- borde verde: aprobado;
- borde rojo: desaprobado.

El criterio de aprobación es de al menos 6 respuestas correctas.
