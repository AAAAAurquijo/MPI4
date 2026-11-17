import os
import time
from collections import Counter
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def cargar_consulta(consulta_path, case_sensitive=False):
    """
    Lee consulta.txt y devuelve un conjunto de palabras objetivo.
    """
    if not os.path.isfile(consulta_path):
        raise FileNotFoundError(f"No se encontró el archivo de consulta: {consulta_path}")

    with open(consulta_path, "r", encoding="utf-8") as f:
        palabras = [line.strip() for line in f if line.strip()]

    if not case_sensitive:
        palabras = [w.lower() for w in palabras]

    return set(palabras)

def listar_archivos(dataset_dir):
    """
    Devuelve la lista ordenada de rutas de los archivos file_*.txt
    """
    rutas = []
    for fname in os.listdir(dataset_dir):
        if not fname.startswith("file_") or not fname.endswith(".txt"):
            continue
        ruta = os.path.join(dataset_dir, fname)
        if os.path.isfile(ruta):
            rutas.append(ruta)
    return sorted(rutas)

def contar_palabras_en_archivos(rutas, palabras_objetivo, case_sensitive=False):
    """
    Cuenta cuántas veces aparecen las palabras de consulta.txt
    en archivos file_*.txt asignados a un proceso
    """

    freq_local = Counter()
    total_tokens_leidos = 0

    for ruta in rutas:
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                palabras = linea.split()
    
                if not case_sensitive:
                    palabras = [w.lower() for w in palabras]
    
                total_tokens_leidos += len(palabras)
    
                for w in palabras:
                    if w in palabras_objetivo:
                        freq_local[w] += 1
    
    return freq_local, len(rutas), total_tokens_leidos

def guardar_resultados_csv(resultados_path, freq_global):
    """
    Guarda el conteo global en CSV para comparar con baseline y Dask.
    """
    with open(resultados_path, "w", encoding="utf-8") as f:
        f.write("palabra,conteo\n")
        for palabra in sorted(freq_global):
            f.write(f"{palabra},{freq_global[palabra]}\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")

    consulta_name = "consulta.txt"
    case_sensitive = False
    top_n = 10
    output_file = "mpi1_results.csv"

    consulta_path = os.path.join(dataset_dir, consulta_name)

    if not os.path.isdir(dataset_dir):
            if rank == 0:
                print(f"Error: no existe la carpeta dataset en {dataset_dir}")
            return
    if not os.path.isfile(consulta_path):
        if rank == 0:
            print(f"Error: no se encontró el archivo de consulta: {consulta_path}")
        return

    comm.Barrier()
    t0 = time.perf_counter()

    #1.	rank 0 reads consulta.txt
    if rank == 0:
        palabras_objetivo = cargar_consulta(consulta_path, case_sensitive=case_sensitive)
    else:
        palabras_objetivo = None

    #2.	rank 0 broadcasts the query words to all processes (comm.bcast)
    palabras_objetivo = comm.bcast(palabras_objetivo, root=0)

    #3. rank 0 obtains the list of file_*.txt files y 4. the files are distributed statically among the processes
    if rank == 0:
        files = listar_archivos(dataset_dir)
        chunks = [files[i::size] for i in range(size)]
    else:
        chunks = None
    
    mis_files = comm.scatter(chunks, root=0)

    #5.	each process locally counts the occurrences of the query words in its assigned files
    t_local0 = time.perf_counter()
    freq_local, n_archivos, n_tokens = contar_palabras_en_archivos(
        mis_files, palabras_objetivo, case_sensitive=case_sensitive
    )
    t_local = time.perf_counter() - t_local0

    #6.	partial results are gathered in rank 0
    resultados = comm.gather((rank, freq_local, n_archivos, n_tokens, t_local), root=0)
    
    
    #7.	rank 0 builds the global result and prints the top 10, which must match the output of the sequential baseline
    if rank == 0:
        freq_global = Counter()
        archivos_procesados = 0
        total_tokens_leidos = 0
        tiempos_locales = []

        print("Reporte por proceso:")
        for r, freq_r, n_arch_r, n_tok_r, t_r in sorted(resultados, key=lambda x: x[0]):
            freq_global.update(freq_r)
            archivos_procesados += n_arch_r
            total_tokens_leidos += n_tok_r
            tiempos_locales.append(t_r)
            print(f"  rank {r}: {n_arch_r} archivos, tiempo local {t_r:.6f} s")

        elapsed = time.perf_counter() - t0

        
        # Load imbalance
        promedio = sum(tiempos_locales) / len(tiempos_locales)
        li = max(tiempos_locales) / promedio if promedio > 0 else 1.0

        # Relative spread
        relative_spread = (max(tiempos_locales)-min(tiempos_locales)) / promedio if promedio > 0 else 0.0
        

        total_ocurrencias = sum(freq_global.values())
        top_words = freq_global.most_common(top_n)

        resultados_path = os.path.join(dataset_dir, output_file)
        guardar_resultados_csv(resultados_path, freq_global)

        print(f"\nProcesos: {size}")
        print(f"Tiempo de ejecución: {elapsed:.6f} segundos")
        print(f"Desbalance de carga (LI): {li:.3f}")
        print(f"Dispersión relativa (RS): {relative_spread:.3f}")
        print(f"Dataset procesado: {dataset_dir}")
        print(f"Archivo de consulta: {consulta_name}")
        print(f"Archivos procesados: {archivos_procesados}")
        print(f"Total de tokens leídos: {total_tokens_leidos}")
        print(f"Total de ocurrencias encontradas: {total_ocurrencias}")
        print(f"Resultados guardados en: {resultados_path}\n")

        print(f"Top {top_n} palabras de consulta en el corpus:")
        for palabra, cuenta in top_words:
            print(f"  {palabra}: {cuenta}")

if __name__ == "__main__":
    main()

# docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 mpiexec --allow-run-as-root --use-hwthread-cpus --oversubscribe -n 13 python /app/mpi1.py