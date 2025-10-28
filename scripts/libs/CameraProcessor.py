import cv2
import numpy as np
import glob
import os

#######################################
### Calibracion de imagen para eliminar distorsiones externas al sensor.
###     - Estimar la pose relativa cámara–plano de referencia.
###     - Rectificar y centrar las imágenes para que todas compartan la misma orientación y escala.
### Autor: Martinez Agustin
### Fecha: 28/10/2025
### Version: 2.0
#######################################

class CameraProcessor:
    def __init__(self, outpath = 'images'):
        # Variables Calibracion
        self._mtx = []
        self._dist = []
        self._image_res = []

        # Path de saldia de imágenes
        self._outpath = outpath

        # Variables ArUco
        self._center_coord = []
        self._homografy = None
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        parameters = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    def isCalibred(self):
        '''
        Indica si el objeto está calibrado o no.
        bool: True si esta calibrado, False sino.
        '''
        return (
            self._mtx is not None and self._dist is not None
            and hasattr(self._mtx, "shape") and self._mtx.size > 0
            and hasattr(self._dist, "shape") and self._dist.size > 0
        )

    def ImageFileName(self, img_path):
        '''
        Devuelve una lista con todas las imágenes dentro de una carpeta (jpg, png, jpeg).
        Parameters:
            img_path (str): path a la carpeta donde se encuentran las fotos.
        Returns:
            list: Lista de imágenes encontradas.
        '''
        img_fname = []

        self.checkPath(img_path)
        for ext in ('*.jpg', '*.png', '*.jpeg'):
            img_fname.extend(glob.glob(os.path.join(img_path, ext)))
        return img_fname
 
    def calib(self, calib_img_path, chess_dim = (9,6), chess_square_lenght = 30.0):
        '''
        Crea la matriz de distorsión de las fotos que se encuentren en el path indicado. Calibra al objeto cameraProcessor.
        Parameters:
            calib_img_path (str): path a la carpeta donde se encuentran todas las imágenes de calibración.
            chess_dim (tuple): dimensiones del tablero a ajedrez a detectar.
            chess_square_lenght (float): largo de un cuadrado del tablero de ajedrez en mm.
        Returns:
            Matlike: Matriz de la cámara.
            Matlike: Coeficientes de distorsión.
            list: Vector de rotación.
            list: Vector de traslación.
            list: Resolución de imágenes de calibración.
        '''
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corner_dim = (chess_dim[0]-1, chess_dim[1]-1)    #Las funciones utilizan las esquinas internas, no la cant de cuadrados

        objp = np.zeros((corner_dim[0]*corner_dim[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:corner_dim[0],0:corner_dim[1]].T.reshape(-1,2)
        objp *= chess_square_lenght     # Grilla con todas las posiciones de las esquinas

        objpoints = [] # puntos 3D en el mundo real
        imgpoints = [] # puntos 2D en la imagen

        # Lista de imagenes
        images = self.ImageFileName(calib_img_path)

        if not images:
            print("No hay imagenes.")
            return None
        if len(images) < 10:
            print("Advertencia. Pocas imágenes. Se recomiendan 10 o más en distintos ángulos")

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)		# Lo pasa a escala de grises para mejorar
            ret, corners = cv2.findChessboardCornersSB(gray, corner_dim, None)
            
            if ret == True:
                print(f"Encontrado en {fname}")
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)

        if len(objpoints) == 0:
            print("No se detectaron esquinas en ninguna imagen.")
            return None

        # Resolucion utilizada para calibrar
        img = cv2.imread(images[0])                  # Nota = usa solo la primer imagen !!! 
        self._image_res = (img.shape[1], img.shape[0])    # calib_res = (WIDHT, HEIGHT) 

        # Calibración
        ret, self._mtx, self._dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        return self._mtx, self._dist, rvecs, tvecs, self._image_res
    
    def saveCalibMatrix(self, file_name='calib_matrix'):
        '''
        Guarda los valores de calibración en un archivo .npz
        Parameters:
            file_name (str): Nombre del archivo a guardar.
        '''
        out_path = os.path.join(self._outpath, file_name)
        self.checkPath(self._outpath)

        try:
            np.savez(out_path, mtx=self._mtx, dist=self._dist, _image_res=self._image_res )
            print(f"Matrices guardadas en {file_name}.npz")
        except Exception as e:
            print(f"Error al guardar: {e}")

    def loadCalibMatrix(self, file_name='calib_matrix'):
        '''
        Lee los valores de calibración de un archivo .npz
        Parameters:
            file_name (str): Nombre del archivo a leer
        Returns:
            Matlike: Matriz de la cámara.
            Matlike: Coeficientes de distorsión de la cámara.
            list: Resolución de imágenes de calibración.
        '''
        out_path = os.path.join(self._outpath, file_name)
        if self.checkPath(self._outpath, create=False):
            try:
                calib = np.load(f"{out_path}.npz")
                self._mtx, self._dist, self._image_res = calib["mtx"], calib["dist"], calib['_image_res']
                self._image_res = tuple(self._image_res)
            except Exception as e:
                print('Error al cargar el archivo')
                return None, None, None

        return self._mtx, self._dist, self._image_res

    def undistort(self, img):
        '''
        Elimina la distorsión intrínseca en una imagen.
        Parameters:
            img (Matlike): archivo del tipo Matlike de cv2.
        Returns:
            Matlike: Imagen sin distorción.
            Matlike: Matriz de cámara nueva.
            Matlike: Roi.
        '''
        if not self.checkImage(img):
            return None, None, None

        if (img.shape[1], img.shape[0]) != tuple(self._image_res):
            print("WARNING: Resolucion de imagen diferente a la de calibracion\n")

        try:
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self._mtx, self._dist, (w,h), 1, (w,h))

            # Elimina distorcion
            undistorted = cv2.undistort(img, self._mtx, self._dist, None, newcameramtx)
    
            # Recorte de la imagen
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]
            return undistorted, newcameramtx, roi
        except Exception as e:
            print(f'Error al corregir distorcion {e}')
            return None, None, None
        
    def warp(self, img, corner_ids, plane_size, pixels_per_mm = 2):
        '''
        Realiza Homografía con 4 esquinas ArUco.
        Parameters:
            calib_value_file: Nombre de archivo (sin extensión) de las matrices de distorsión.
            images_path: path a carpeta de imágenes que van a realizar el path.
            corner_ids: IDs de ArUco a identificar como esquinas.
            plane_size: Tamaño del plano de 4 esquinas.
            pixels_per_mm: Resolución pixeles --> mm.
        '''
        if not self.isCalibred():
            print('No hay matriz intrinseca y/o coeficientes de distorsion')
            return None

        if not self.checkImage(img):
            print('No hay imagen')
            return None
        
        # Cálculo de puntos de esquinas (ejemplo: ([0 0], [0 100], [24 0], [24 100]) mm)
        image_size = (plane_size[0]*pixels_per_mm,
                    plane_size[1]*pixels_per_mm) 

        real_points = np.array([
                        [0, 0],                             # arriba izq
                        [image_size[0]-1, 0],               # arriba der
                        [image_size[0]-1, image_size[1]-1], # abajo der
                        [0, image_size[1]-1]                # abajo izq
                    ], dtype=np.float32)

       
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray_img)
        corner_coords = self.get_corners(corner_ids, corners, ids)
        if corner_coords is None or len(corner_coords) != 4:
            print('no hay coordenadas')
            return None

        # Corrección de la imagen: 
        H, _ = cv2.findHomography(corner_coords, real_points)
        self._homografy = H
        aligned_img = cv2.warpPerspective(img, H, image_size)

        return aligned_img

    def get_corners(self, corner_ids, corners, ids):
        '''
        Devuelve un vector de 4 posiciones con los centros de cada esquina. 
        El resultado está ordenado [TOP LEFT, TOP RIGHT, BOTTOM RIGHT, BOTTOM LEFT]
        Parameters:
            corner_ids: Id's a utilizar como esquina.
            corners: Vectos de esquinas de todos los ArUco obtenidos.
            ids: Id de cada ArUco enviado en corners.
        '''
        detected_centers = []
        corner_coords = []

        if ids is not None:
            # Busqueda de corners:
            ids = ids.flatten()

            for i, marker_id in enumerate(ids):
                # Obtengo centros solo de los corners ID:
                if marker_id in corner_ids:
                    c = corners[i][0]
                    center = c.mean(axis=0)
                    detected_centers.append(center)

            if len(detected_centers) == 4:
                detected_centers = np.array(detected_centers, dtype=np.float32)

                s = detected_centers.sum(axis=1)
                diff = np.diff(detected_centers, axis=1)

                top_left = detected_centers[np.argmin(s)]
                bottom_right = detected_centers[np.argmax(s)]
                top_right = detected_centers[np.argmin(diff)]
                bottom_left = detected_centers[np.argmax(diff)]

                corner_coords = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

        return corner_coords

    def colorFilter(self, img, color_range):
        '''
        Detecta regiones de un color específico en todas las imágenes dentro de 'images_path/warped'
        y guarda los resultados en 'images_path/color'.

        Parameters:
            images_path (str): Carpeta base donde están las imágenes originales.
            color_range (tuple): (lower_HSV, upper_HSV) con los límites del color a detectar.
        '''

        if not self.checkImage(img):
            print("No hay imágenes.")
            return None, None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_range[0], color_range[1])

        # Limpieza de ruido
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # resultado a color
        # masked_img_color = cv2.bitwise_and(img, img, mask=mask)
        white_background = np.full_like(img, 255)  # fondo blanco
        masked_img_color = np.where(mask[..., None].astype(bool), img, white_background)

        return mask, masked_img_color

    def loadImage(self, img_path):
        '''
        Carga una imagen a partir de su path.
        Parameters:
            img_path: Path de la imagen.
        '''
        if self.checkPath(img_path, create=False):
            img = cv2.imread(img_path)
            return img
        else:
            return None
    
    def saveImage(self, img, file_name='img.jpeg', img_rel_path=''):
        '''
        Salva una imagen en la carpeta indicada.
        Parameters:
            img: Imagen a guardar.
            file_name: Nombre del archivo imagen a guardar.
            img_rel_path: Path relativo a donde se va a guardar.
        '''
        out_path = os.path.join(self._outpath, img_rel_path)
        self.checkPath(out_path)

        out_path = os.path.join(out_path, file_name)
        cv2.imwrite(out_path, img)

    def processImages(self, img_folder_path, color_filter, corner_ids, plane_size, pixels_per_mm = 2, center_id = 50, save = False, out_path='processed'):
        '''
        Realiza el proceso completo de una imagen. Requiere que la clase ya esté calibrada.
        Parameters:
            img_folder_path (str): Path a carpeta con fotos a procesar.
            color_filter (tuple): filtro de colores a utiliziar.
            corner_ids (list): Id de los ArUco que definen las esquinas.
            plane_size (tuple): Tamaño del plano en mm.
            pixels_per_mm (float): pixeles equivalentes a un mm en el resultado final.
            save (bool): Salva los resultados o no.
            out_path (str): Path donde guarda los resultados.
        returns:
            Matlike: Imagen procesada.
            Matlike: Imagen procesada a color.
        '''
        images = self.ImageFileName(img_folder_path)
        img_res = []
        img_color_res = []

        out = os.path.join(self._outpath, out_path)
        if save:
            self.checkPath(out)
        
        for fname in images:
            print(f'Procesando: {fname}\n')
            img = self.loadImage(fname)
            img, _, _ = self.undistort(img)
            
            self._center_coord = None
            self.getOrigin(img, center_id)

            img = self.warp(img, corner_ids, plane_size, pixels_per_mm)
            if self._center_coord is not None:
                self._center_coord = self.transformPoint(self._center_coord)

            # falta detectar centro!!
            if color_filter:
                img, img_color = self.colorFilter(img, color_filter)
            else:
                img_color = img

            img_color = self.drawOrigin(img_color)

            # Guardado
            if not self.checkImage(img):
                continue
            img_res.append(img)
            img_color_res.append(img_color)

            if save:
                name, ext = os.path.splitext(os.path.basename(fname))

                out_bw = os.path.join(out, f"{name}{ext}")
                out_color = os.path.join(out, f"{name}_color{ext}")
                cv2.imwrite(out_bw, img)
                cv2.imwrite(out_color, img_color)
        
        return img_res, img_color_res

    def transformPoint(self, points, inverse=False):
        '''
        Transforma uno o varios puntos utilizando la homografía almacenada en la clase.
        Parameters:
            points (tuple | list | np.ndarray): Punto o lista de puntos en formato (x, y).
            inverse (bool): Si True, aplica la homografía inversa (de plano a imagen original).
        Returns:
            np.ndarray: Puntos transformados en formato (N, 2).
        '''
        if self._homografy is None:
            print("No hay homografía almacenada. Ejecute warp primero.")
            return None

        pts = np.array(points, dtype=np.float32)

        # Forzar a formato (N, 1, 2)
        if pts.ndim == 1:  
            if pts.shape[0] != 2:
                raise ValueError("Un punto debe tener formato (x, y)")
            pts = pts.reshape((1, 1, 2))
        elif pts.ndim == 2:
            if pts.shape[1] != 2:
                raise ValueError("Cada punto debe tener formato (x, y)")
            pts = pts.reshape((-1, 1, 2))
        else:
            raise ValueError("Formato de puntos inválido. Esperado (N,2) o (2,)")

        # Seleccionar homografía directa o inversa
        H = np.linalg.inv(self._homografy) if inverse else self._homografy

        # Aplicar la transformación
        transformed = cv2.perspectiveTransform(pts, H)

        # Salida (N, 2)
        return transformed[:, 0, :]

    def getOrigin(self, img, origin_Id):
        '''
        Define al centro de coordenadas en el centro del ArUco con la ID correspondiente.
        Parameters:
            img: Imagen a analizar.
        origin_Id: Id del ArUco origen de coordenadas.
        '''
        if not self.checkImage(img):
            print("Imagen inválida para buscar origen.")
            return []
        corners, ids, _ = self._aruco_detector.detectMarkers(img)
        if ids is not None:
            for i, corner in enumerate(corners):
                if ids[i] == origin_Id:
                    c = corner[0]
                    self._center_coord = c.mean(axis=0)
                    break
        
        return self._center_coord
    
    def drawOrigin(self, img, radius=10, color=(0, 0, 255), thickness=-1):
        '''
        Dibuja un punto en la imagen en la posición del origen detectado.
        Parameters:
            img: Imagen donde se va a dibujar.
            radius (int): Radio del círculo.
            color (tuple): Color BGR (por defecto rojo).
            thickness (int): Grosor del círculo (-1 para relleno).
        '''
        if not self.checkImage(img):
            print("Imagen inválida para dibujar origen.")
            return img

        if self._center_coord is None or len(self._center_coord) == 0:
            print("No se detectó origen. Ejecute getOrigin primero.")
            return img
        
        coord = np.array(self._center_coord, dtype=np.float32).reshape(-1)
        x, y = int(coord[0]), int(coord[1])

        cv2.circle(img, (x, y), radius, color, thickness)
        return img

    def checkPath(self, path, create = True):
        '''
        Verifica que el Path exista. Lo crea sino.
        Parameters:
            path (str): Path a verificar.
            create (bool): Lo crea si no existe.
        '''
        exist = os.path.exists(path)
        if not exist and create:
            os.makedirs(path, exist_ok=True)
            exist = True
        return exist
    
    def checkImage(self, img):
        '''
        Verifica que la imagen exista y sea válida.

        Parameters:
            img (Matlike | None): Imagen a verificar.

        Returns:
            bool: True si la imagen es válida, False si no lo es.
        '''
        if img is None:
            return False

        if not isinstance(img, np.ndarray):
            return False

        if img.size == 0 or len(img.shape) < 2:
            return False

        return True
    
