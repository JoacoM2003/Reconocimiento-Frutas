import os
import csv

def generar_csv_desde_directorio(base_dir, salida_csv):
    datos = []

    for clase in os.listdir(base_dir):
        clase_path = os.path.join(base_dir, clase)
        if not os.path.isdir(clase_path):
            continue

        for archivo in os.listdir(clase_path):
            if archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                ruta_archivo = os.path.join(clase, archivo)
                datos.append([ruta_archivo, clase])

    with open(salida_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['archivo', 'clase'])
        writer.writerows(datos)

    print(f"✅ CSV generado: {salida_csv} (total: {len(datos)} filas)")

# Usar con tus carpetas reales
generar_csv_desde_directorio('frutas/Training', 'train_labels.csv')
generar_csv_desde_directorio('frutas/Test', 'test_labels.csv')
