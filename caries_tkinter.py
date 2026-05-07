# ============================
# LIBRERÍAS
# ============================
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================
# CARGAR DATOS
# ============================
base_dir = os.path.dirname(__file__)
ruta_csv = os.path.join(base_dir, "dataset_caries.csv")

df = pd.read_csv(ruta_csv)

x = df.drop(['caries'], axis=1)
y = df['caries']

# Guardamos los nombres de columnas para usarlos al predecir
columnas = x.columns.tolist()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# ============================
# COMPARAR MODELOS
# ============================
modelos = {
    "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
    "Árbol de Decisión":  DecisionTreeClassifier(),
    "SVM":                SVC(),
    "Naive Bayes":        GaussianNB(),
    "KNN":                KNeighborsClassifier(),
    "Regresión Logística": LogisticRegression(max_iter=1000)
}

mejor_modelo  = None
modelo_usado  = ""
mejor_accuracy = 0
mejor_pred    = None

for nombre, modelo in modelos.items():
    modelo.fit(x_train, y_train)
    pred = modelo.predict(x_test)
    acc  = accuracy_score(y_test, pred)

    if acc > mejor_accuracy:
        mejor_accuracy = acc
        mejor_modelo   = modelo
        modelo_usado   = nombre
        mejor_pred     = pred

rf     = mejor_modelo
y_pred = mejor_pred

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

matrix = confusion_matrix(y_test, y_pred)
print("MATRIZ DE CONFUSIÓN")
print(matrix)

# ============================
# FUNCIONES AUXILIARES
# ============================

def formatear_campos_invalidos(campos):
    """
    Recibe una lista de nombres de campos inválidos:
      - 1 campo  → "Edad"
      - 2 campos → "Edad y Altura"
      - 3+ campos → "Edad, Altura y Peso"
    """
    if len(campos) == 1:
        return campos[0]
    elif len(campos) == 2:
        return f"{campos[0]} y {campos[1]}"
    else:
        return ", ".join(campos[:-1]) + f" y {campos[-1]}"

def actualizar_matriz():
    """Muestra los valores de la matriz de confusión global en los labels."""
    tn_label.config(text=str(matrix[0, 0]))
    fp_label.config(text=str(matrix[0, 1]))
    fn_label.config(text=str(matrix[1, 0]))
    vp_label.config(text=str(matrix[1, 1]))

def borrar_campos():
    for entry in entradas:
        entry.delete(0, tk.END)
        entry.insert(0, placeholders[entry])
        entry.config(fg="gray")

    diagnostico_label.config(text="")

    # Limpia las celdas de la matriz de confusión
    tn_label.config(text="")
    fp_label.config(text="")
    fn_label.config(text="")
    vp_label.config(text="")

def obtener_diagnostico():
    """
    Valida cada campo numérico individualmente.
    Si hay errores, muestra exactamente qué campos son inválidos.
    """
    # Mapeo entry → nombre legible del campo
    nombres_campos = {
        edad_entry:               "Edad",
        altura_entry:             "Altura",
        peso_entry:               "Peso",
        cintura_entry:            "Cintura",
        vista_izquierda_entry:    "Vista izquierda",
        vista_derecha_entry:      "Vista derecha",
        audicion_izquierda_entry: "Audición izquierda",
        audicion_derecha_entry:   "Audición derecha",
        sistolica_entry:          "Sistólica",
        presion_arterial_entry:   "Presión arterial",
        glucosa_en_ayunas_entry:  "Glucosa",
        colesterol_entry:         "Colesterol",
        triglicerido_entry:       "Triglicéridos",
        hdl_entry:                "HDL",
        ldl_entry:                "LDL",
        hemoglobina_entry:        "Hemoglobina",
        proteina_urinaria_entry:  "Proteína urinaria",
        creatinina_serica_entry:  "Creatinina sérica",
        ast_entry:                "AST",
        alt_entry:                "ALT",
        gtp_entry:                "GTP",
    }

    campos_invalidos = []
    valores = []

    # Validar cada campo numérico (se omite nombre_entry, índice 0)
    for entry in entradas[1:]:
        valor = entry.get()
        nombre = nombres_campos[entry]

        # Campo vacío o con placeholder
        if valor == "" or valor == placeholders[entry]:
            campos_invalidos.append(nombre)
            continue

        # Campo con valor no numérico
        try:
            valores.append(float(valor))
        except ValueError:
            campos_invalidos.append(nombre)

    # Si hay campos inválidos → mostrar mensaje específico
    if campos_invalidos:
        lista = formatear_campos_invalidos(campos_invalidos)
        messagebox.showerror(
            "Error",
            f"Complete todos los campos correctamente.\n\n"
            f"Datos no válidos en el campo: {lista}."
        )
        return

    # Todo correcto → predecir
    nombre_paciente = nombre_entry.get()

    data_df    = pd.DataFrame([valores], columns=columnas)
    data_scaled = scaler.transform(data_df)
    pred       = rf.predict(data_scaled)

    resultado = (
        "NO es propenso a padecer caries dental."
        if pred[0] == 0
        else "SI es propenso a padecer caries dental."
    )

    diagnostico_label.config(
        text=f"Paciente: {nombre_paciente}\n\n"
             f"{resultado}\n\n"
             f"Modelo usado: {modelo_usado}\n\n"
             f"Accuracy: {accuracy:.2%}\n"
             f"Precisión: {precision:.2%}\n"
             f"Recall: {recall:.2%}\n"
             f"F1-Score: {f1:.2%}"
    )

    actualizar_matriz()

def mostrar_instrucciones():
    mensaje = (
        "1. Ingrese en cada campo con todos los datos clínicos del paciente.\n\n"
        "2. Presione 'Diagnosticar' para obtener el resultado y las métricas del modelo.\n\n"
        "3. Presione 'Limpiar' para borrar los campos e iniciar un nuevo diagnóstico.\n\n"
        "4. Presione 'Salir' para cerrar el programa.\n\n\n"
        "Presione 'OK' para cerrar esta ventana.\n\n"
    )
    messagebox.showinfo("Instrucciones", mensaje)

def salir():
    window.destroy()

# ============================
# INTERFAZ
# ============================
window = tk.Tk()
window.title("Sistema Predictivo de Caries Dental")
window.geometry("1180x820")
window.configure(bg="#D6EAF8")

titulo = tk.Label(
    window,
    text="Sistema Predictivo de Caries Dental",
    font=("Segoe UI", 22, "bold"),
    bg="#D6EAF8",
    fg="#2C3E50"
)
titulo.pack(pady=20)

main_frame = tk.Frame(window, bg="#D6EAF8")
main_frame.pack(padx=20, pady=10)


# ============================
# COLUMNA IZQ: DATOS DEL PACIENTE
# ============================
datos_frame = tk.LabelFrame(
    main_frame,
    text="Datos del Paciente",
    font=("Segoe UI", 12, "bold"),
    bg="#F8F9F9",
    fg="#34495E",
    padx=20,
    pady=20
)
datos_frame.grid(row=0, column=0, padx=20, sticky="n")

# =================================================
# COLUMNA DECHA: DIAGNÓSTICO + MATRIZ DE CONFUSIÓN
# =================================================
right_frame = tk.Frame(main_frame, bg="#D6EAF8")
right_frame.grid(row=0, column=1, sticky="n", padx=20)

diagnostico_frame = tk.LabelFrame(
    right_frame,
    text="Resultado del Diagnóstico",
    font=("Segoe UI", 12, "bold"),
    bg="#F8F9F9",
    fg="#34495E",
    padx=20,
    pady=20,
    width=320,
    height=350
)
diagnostico_frame.grid(row=0, column=0, sticky="n")
diagnostico_frame.grid_propagate(False)

matriz_frame = tk.LabelFrame(
    right_frame,
    text="Matriz de Confusión",
    font=("Segoe UI", 12, "bold"),
    bg="#F8F9F9",
    fg="#34495E",
    padx=15,
    pady=10
)
matriz_frame.grid(row=0, column=1, sticky="n", padx=(10, 0))

# Encabezados de columna
tk.Label(matriz_frame, text="",              bg="#F8F9F9", width=10).grid(row=0, column=0)
tk.Label(matriz_frame, text="Predicho: NO", bg="#F8F9F9", fg="#2C3E50",
         font=("Segoe UI", 9, "bold"), width=11).grid(row=0, column=1)
tk.Label(matriz_frame, text="Predicho: SÍ", bg="#F8F9F9", fg="#2C3E50",
         font=("Segoe UI", 9, "bold"), width=11).grid(row=0, column=2)

# Encabezados de fila
tk.Label(matriz_frame, text="Real: NO", bg="#F8F9F9", fg="#2C3E50",
         font=("Segoe UI", 9, "bold")).grid(row=1, column=0, pady=4)
tk.Label(matriz_frame, text="Real: SÍ", bg="#F8F9F9", fg="#2C3E50",
         font=("Segoe UI", 9, "bold")).grid(row=2, column=0, pady=4)

# Celdas de valor — inician vacías
tn_label = tk.Label(matriz_frame, text="",
                    font=("Segoe UI", 13, "bold"),
                    bg="#A9DFBF", fg="#1E8449", width=7, relief="groove")
tn_label.grid(row=1, column=1, padx=4, pady=4)

fp_label = tk.Label(matriz_frame, text="",
                    font=("Segoe UI", 13, "bold"),
                    bg="#F1948A", fg="#922B21", width=7, relief="groove")
fp_label.grid(row=1, column=2, padx=4, pady=4)

fn_label = tk.Label(matriz_frame, text="",
                    font=("Segoe UI", 13, "bold"),
                    bg="#F1948A", fg="#922B21", width=7, relief="groove")
fn_label.grid(row=2, column=1, padx=4, pady=4)

vp_label = tk.Label(matriz_frame, text="",
                    font=("Segoe UI", 13, "bold"),
                    bg="#A9DFBF", fg="#1E8449", width=7, relief="groove")
vp_label.grid(row=2, column=2, padx=4, pady=4)

# Leyendas
tk.Label(
    matriz_frame,
    text="VN |  FP\n"
         "FN |  VP",
    bg="#F8F9F9", fg="#7F8C8D",
    font=("Segoe UI", 8), justify="center"
).grid(row=3, column=0, columnspan=3, pady=(8, 4))

# ============================
# CAMPOS DEL FORMULARIO
# ============================
placeholders = {}

def crear_campo(frame, texto, fila, columna, placeholder=""):
    tk.Label(
        frame,
        text=texto,
        font=("Segoe UI", 10),
        bg="#F8F9F9",
        fg="#2C3E50"
    ).grid(row=fila, column=columna, sticky="w", pady=7)

    entry = tk.Entry(frame, font=("Segoe UI", 10), width=18, fg="gray")
    entry.grid(row=fila, column=columna + 1, padx=8, pady=7)
    entry.insert(0, placeholder)
    placeholders[entry] = placeholder

    def entrar(event):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg="black")

    def salir_entry(event):
        if entry.get() == "":
            entry.insert(0, placeholder)
            entry.config(fg="gray")

    entry.bind("<FocusIn>",  entrar)
    entry.bind("<FocusOut>", salir_entry)
    return entry

# ============================
# CAMPOS
# ============================
entradas = []

nombre_entry              = crear_campo(datos_frame, "Nombre:",             0,  0, "Ej: Juan Pérez")
edad_entry                = crear_campo(datos_frame, "Edad:",               1,  0, "18 - 80")
altura_entry              = crear_campo(datos_frame, "Altura:",             2,  0, "140 - 200")
peso_entry                = crear_campo(datos_frame, "Peso:",               3,  0, "40 - 150")
cintura_entry             = crear_campo(datos_frame, "Cintura:",            4,  0, "50 - 130")
vista_izquierda_entry     = crear_campo(datos_frame, "Vista izquierda:",    5,  0, "0.1 - 2.0")
vista_derecha_entry       = crear_campo(datos_frame, "Vista derecha:",      6,  0, "0.1 - 2.0")
audicion_izquierda_entry  = crear_campo(datos_frame, "Audición izquierda:", 7,  0, "1 - 2")
audicion_derecha_entry    = crear_campo(datos_frame, "Audición derecha:",   8,  0, "1 - 2")
sistolica_entry           = crear_campo(datos_frame, "Sistólica:",          9,  0, "80 - 200")
presion_arterial_entry    = crear_campo(datos_frame, "Presión arterial:",   10, 0, "50 - 130")

glucosa_en_ayunas_entry   = crear_campo(datos_frame, "Glucosa:",            0,  2, "70 - 200")
colesterol_entry          = crear_campo(datos_frame, "Colesterol:",         1,  2, "100 - 350")
triglicerido_entry        = crear_campo(datos_frame, "Triglicéridos:",      2,  2, "30 - 500")
hdl_entry                 = crear_campo(datos_frame, "HDL:",                3,  2, "20 - 100")
ldl_entry                 = crear_campo(datos_frame, "LDL:",                4,  2, "50 - 250")
hemoglobina_entry         = crear_campo(datos_frame, "Hemoglobina:",        5,  2, "8 - 20")
proteina_urinaria_entry   = crear_campo(datos_frame, "Proteína urinaria:",  6,  2, "1 - 6")
creatinina_serica_entry   = crear_campo(datos_frame, "Creatinina sérica:",  7,  2, "0.5 - 2.0")
ast_entry                 = crear_campo(datos_frame, "AST:",                8,  2, "10 - 200")
alt_entry                 = crear_campo(datos_frame, "ALT:",                9,  2, "10 - 200")
gtp_entry                 = crear_campo(datos_frame, "GTP:",                10, 2, "10 - 300")

entradas = [
    nombre_entry, edad_entry, altura_entry, peso_entry, cintura_entry,
    vista_izquierda_entry, vista_derecha_entry, audicion_izquierda_entry,
    audicion_derecha_entry, sistolica_entry, presion_arterial_entry,
    glucosa_en_ayunas_entry, colesterol_entry, triglicerido_entry,
    hdl_entry, ldl_entry, hemoglobina_entry, proteina_urinaria_entry,
    creatinina_serica_entry, ast_entry, alt_entry, gtp_entry
]

# ============================
# LABEL DEL RESULTADO
# ============================
diagnostico_label = tk.Label(
    diagnostico_frame,
    text="Aquí aparecerá el diagnóstico del paciente",
    font=("Segoe UI", 11),
    bg="#F8F9F9",
    fg="#2C3E50",
    wraplength=260,
    justify="left"
)
diagnostico_label.pack(pady=20)

# ============================
# BOTONES
# ============================
botones_frame = tk.Frame(window, bg="#D6EAF8")
botones_frame.pack(side="bottom", pady=20)

btn_style = {
    "font":   ("Segoe UI", 10, "bold"),
    "width":  12,
    "bd":     0,
    "cursor": "hand2",
    "pady":   8
}

tk.Button(botones_frame, text="Diagnosticar", bg="#3498DB", fg="white",
          command=obtener_diagnostico,   **btn_style).grid(row=0, column=0, padx=8)
tk.Button(botones_frame, text="Limpiar",      bg="#95A5A6", fg="white",
          command=borrar_campos,         **btn_style).grid(row=0, column=1, padx=8)
tk.Button(botones_frame, text="Ayuda",        bg="#27AE60", fg="white",
          command=mostrar_instrucciones, **btn_style).grid(row=0, column=2, padx=8)
tk.Button(botones_frame, text="Salir",        bg="#E74C3C", fg="white",
          command=salir,                 **btn_style).grid(row=0, column=3, padx=8)

window.mainloop()