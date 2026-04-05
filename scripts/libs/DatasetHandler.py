from . import CameraProcessor as cam
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt     # Gráficos

class DatasetHandler:
    def __init__(self, corner_ids, color_filter, map_size, pixels_per_mm, angle_step, calib_matrix_path):
        self.corner_ids = corner_ids
        self.color_filter = color_filter
        self.map_size = map_size
        self.pixels_per_mm = pixels_per_mm
        self.angle_step = angle_step

        self.cam = cam.CameraProcessor()
        self.cam.loadCalibMatrix(calib_matrix_path)

    def merge(self, img_dir, csv_dir, out_dir):
        '''
        Añade a los .CSV del microcontrolador la información de su imagen asociada. Lo guarda todo en un CSV en out_dir
        '''
        os.makedirs(out_dir, exist_ok=True)

        # Process the images
        distances, angles, img_names = self.cam.processDistances(img_dir, self.corner_ids, 
                                                            color_filter=self.color_filter,
                                                            plane_size=self.map_size, 
                                                            pixels_per_mm=self.pixels_per_mm, 
                                                            angle_step=self.angle_step)

        # img_map["name"] = idx
        img_map = {}
        for idx, img in enumerate(img_names):
            name = os.path.splitext(img)[0].lower()
            img_map[f"{name}"] = idx

        # List and sort the CSV name files [csv_name, csv_name, ...]
        csv_files = sorted(
            [f for f in os.listdir(csv_dir) if f.endswith(".csv")],
            key=lambda x: int(x.replace(".csv", "")) )

        for csv in csv_files:
            name = csv.replace(".csv", "")
            df = pd.read_csv(os.path.join(csv_dir, csv))

            if name not in img_map.keys():
                df.loc[0, "angulos_mapa"] = ""
                df.loc[0, "distancias_mapa"] = ""
                print(f"[WARN] Img {name} no encontrada")
                continue

            idx = img_map[name]
            df.loc[0, "angulos_mapa"] = json.dumps(angles[idx])
            df.loc[0, "distancias_mapa"] = json.dumps(distances[idx])
            df.loc[0, "image_path"] = img_names[idx]

            out_path = os.path.join(out_dir, csv)
            df.to_csv(out_path, index=False)

        print("FUSION COMPLETADA")

    def load_dataset(self, dataset_dir) -> pd.DataFrame:
        """
        Carga todos los CSV del directorio dataset_dir.
        Retorna un DataFrame consolidado con columnas:
        id, fecha, imagen_path (si existe), angulo (lista),
        ultrasonico (lista), infrarrojo (lista), kalman (lista),
        distancias_mapa (lista, si existe), angulos_mapa (lista, si existe)
        """
        records = []
        for fname in os.listdir(dataset_dir):
            if not fname.lower().endswith(".csv"):
                continue
            path = os.path.join(dataset_dir, fname)
            df = pd.read_csv(path)

            row = df.iloc[0].to_dict()
            for col in ["angulo", "ultrasonico", "infrarrojo", "kalman", 
                        "distancias_mapa", "angulos_mapa"]:
                if col in row and pd.notna(row[col]) and row[col] != "":
                    try:
                        row[col] = json.loads(row[col])
                    except json.JSONDecodeError:
                        row[col] = None
                else:
                    row[col] = None

            records.append(row)
        df = pd.DataFrame(records)
        
        df.set_index('id', inplace=True)

        return df

    def plot_polar(self, angles_deg, values, title=None, ax:plt.Axes = None, 
                threshold=None, threshold_color="red", **plot_kwargs):
        if ax is None:
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111, projection="polar")
        else:
            fig = ax.figure

        angles_rad = np.deg2rad(angles_deg)
        ax.plot(angles_rad, values, **plot_kwargs)

        # Limite
        if threshold is not None:
            full_circle = np.linspace(0, 2*np.pi, 400)
            r_max = max(values.max(), threshold * 1.2)
            ax.fill_between(full_circle, threshold, r_max, color=threshold_color, alpha=0.2)
            ax.plot(full_circle, np.full_like(full_circle, threshold), color=threshold_color, linestyle="--", linewidth=1.5)

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)

        if title:
            ax.set_title(title)

        ax.grid(True)
        return fig, ax

    def plot_idx(self, df: pd.DataFrame, idx: int, mode="side"):
        """
        mode="side": 3 subplots polares, lado a lado
        mode="overlay": 3 curvas superpuestas en 1 solo polar
        """

        row = df.iloc[idx]
        ang = row["angulo"]
        us  = row["ultrasonico"]
        ir  = row["infrarrojo"]
        ang_map = row.get("angulos_mapa")
        dist_map = row.get("distancias_mapa")
        
        scan_id = row.get("id", idx)

        # MODO SUBPLOTS LATERALES 
        if mode == "side":
            fig = plt.figure(figsize=(18,6))
            
            # Ultrasonido
            ax1 = fig.add_subplot(131, projection="polar")
            self.plot_polar(ang, us, title="Ultrasonido", ax=ax1)

            # Infrarrojo
            ax2 = fig.add_subplot(132, projection="polar")
            self.plot_polar(ang, ir, title="Infrarrojo", ax=ax2)

            # Imagen procesada
            ax3 = fig.add_subplot(133, projection="polar")
            self.plot_polar(ang_map, dist_map, title="Foto procesada", ax=ax3)

            fig.suptitle(f"Medición ID {scan_id}", fontsize=14)
            plt.tight_layout()
            plt.show()
            return fig

        # MODO SUPERPUESTO 
        elif mode == "overlay":
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111, projection="polar")

            self.plot_polar(ang, us,  ax=ax, label="Ultrasonido")
            self.plot_polar(ang, ir,  ax=ax, label="Infrarrojo")
            self.plot_polar(ang_map, dist_map, ax=ax, label="Distancia mapa")

            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
            ax.set_title(f"Medición ID {scan_id} — Comparación")

            plt.show()
            return fig

        else:
            raise ValueError('mode debe ser "side" o "overlay"')

    def _is_image(self, file_name: str) -> bool:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        return any(file_name.lower().endswith(ext) for ext in valid_extensions)
    
    def _is_image(self, file_name: str) -> bool:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        return any(file_name.lower().endswith(ext) for ext in valid_extensions)


if __name__ == "__main__":
    a = DatasetHandler([], [], [80,80], 3, 3, "..")
    a.merge("img_dir", "csv_dir", "out_dir")
    dataset = a.load_dataset("out_dir")

