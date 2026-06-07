# Machine-Learning (Sistema Predictivo de Caries Dental)

Aplicación de escritorio con interfaz gráfica para predecir la propensión a padecer caries dental usando modelos de Machine Learning entrenados con scikit-learn.

Machine-Learning incluye:
- Un dataset incorporado (dataset_caries.csv).
- Un script con interfaz gráfica (Tkinter) que compara varios modelos, selecciona el mejor y permite diagnosticar pacientes mediante inputs clínicos.
- Visualización de métricas: accuracy, precision, recall, F1-score y matriz de confusión.

---

## Tabla de Contenidos

- [Características](#características)
- [Tecnologías](#tecnologías)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Arquitectura](#arquitectura)
- [Validaciones](#validaciones)
- [Notas](#notas)
- [Autor](#autor)

---

## Características

### Funcionalidades Principales

- Interfaz gráfica con Tkinter para ingresar datos clínicos del paciente.
- Comparación automática de varios modelos de clasificación (Random Forest, Decision Tree, SVM, Naive Bayes, KNN, Regresión Logística).
- Selección del mejor modelo en función del accuracy sobre el conjunto de prueba.
- Predicción en tiempo real para un paciente a partir de los valores ingresados.
- Visualización de métricas: Accuracy, Precision, Recall, F1-Score y matriz de confusión.
- Validación de entradas con mensajes de error claros y feedback vía diálogos.
- Dataset incluido (dataset_caries.csv) para entrenamiento y pruebas.

---

## Tecnologías

| Categoría | Detalles |
|-----------|----------|
| **Lenguaje** | Python 3.8+ |
| **Librerías** | pandas, numpy, scikit-learn, tkinter |
| **Modelos ML** | Random Forest, Decision Tree, SVM, Naive Bayes, KNN, Regresión Logística |
| **Preprocesamiento** | MinMaxScaler (escalado) |
| **Almacenamiento** | Dataset CSV incluido en el repositorio (`dataset_caries.csv`) |

---

## Requisitos

### Sistema / Software

- Python 3.8 o superior
- pip (gestor de paquetes)

### Paquetes Python

- pandas
- numpy
- scikit-learn

tkinter suele venir preinstalado con Python en la mayoría de distribuciones; en caso contrario, instala el paquete correspondiente de tu sistema.

---

## Instalación

Clona el repositorio:

```bash
git clone https://github.com/FedeJGM/Machine-Learning.git
```

Entra al directorio:

```bash
cd Machine-Learning
```

---

## Ejecutar localmente (Recomendado)

Ejecutar el script:

```bash
python caries_tkinter.py
```

El script entrenará los modelos usando `dataset_caries.csv`, seleccionará el mejor basado en accuracy y abrirá la interfaz gráfica para diagnósticos.

---

## Uso

### Flujo básico

1. Ejecuta `python caries_tkinter.py`.
2. Se entrenan y comparan los modelos con el dataset incluido.
3. En la ventana:
   - Ingresa el nombre y los datos clínicos del paciente en los campos correspondientes.
   - Presiona "Diagnosticar" para obtener la predicción y ver las métricas del modelo.
   - Presiona "Limpiar" para restaurar los placeholders y borrar campos.
   - Presiona "Ayuda" para ver instrucciones rápidas.
   - Presiona "Salir" para cerrar la aplicación.

### Qué muestra la aplicación

- Resultado del diagnóstico: indica si el paciente es propenso a caries según la predicción.
- Modelo usado: nombre del modelo con mejor desempeño.
- Accuracy, Precisión, Recall, F1-Score del modelo seleccionado.
- Matriz de confusión (VN, FP, FN, VP).

---

## Arquitectura

| Capa | Responsabilidad |
|------|-----------------|
| Data | Carga y preparación del dataset (`dataset_caries.csv`) |
| Preprocessing | Escalado (MinMaxScaler), separación train/test |
| Models | Entrenamiento y comparación de clasificadores (scikit-learn) |
| UI | Interfaz Tkinter para entrada de datos y visualización de resultados |
| Utils | Validaciones de campos, formato de mensajes, helpers para UI |

---

## Validaciones

La aplicación valida los campos de entrada y muestra errores específicos si faltan o son inválidos.

### Reglas generales

- Campos numéricos deben contener valores numéricos válidos.
- Campos obligatorios: todos los campos clínicos excepto el nombre (el nombre se usa solo para etiquetar el diagnóstico).
- Se muestran mensajes de error con el/los campos inválidos (ej.: "Edad y Altura").

---

## Notas

### Dataset

- Archivo: `dataset_caries.csv`
- El entrenamiento se realiza localmente en el script al iniciarse; no se suben modelos a un servicio externo.
- Si deseas re-entrenar con otros datos, reemplaza el CSV manteniendo la misma estructura de columnas.

### Limitaciones

- El dataset incluido determina el rendimiento; la calidad y representatividad de los datos afectan las métricas.
- Entrenamiento y evaluación se hacen en memoria con scikit-learn; datasets muy grandes pueden requerir recursos mayores.
- La interfaz es local (Tkinter) y no está diseñada para uso multiusuario ni para producción en servidor.

---

## Autor

FedeJGM

---

``` 
