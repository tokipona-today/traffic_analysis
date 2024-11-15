import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import deque
import tempfile
import os
from PIL import Image
import urllib.request
from io import BytesIO


@dataclass
class ROI:
    points: np.ndarray
    mask: np.ndarray
    name: str
    color: Tuple[int, int, int]
    bbox: Tuple[int, int, int, int]
    background_subtractor: cv2.BackgroundSubtractor = None
    motion_frames: deque = None
    params: Dict = None

    def __post_init__(self):
        """Initialise les attributs spécifiques à chaque ROI"""
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=50,
            detectShadows=False
        )
        self.motion_frames = deque(maxlen=10)
        self.params = {
            'min_area': 5,
            'history': 50,
            'threshold': 10,
            'learning_rate': 0.01
        }


class TrafficAnalyzerApp:
    def __init__(self):
        self.rois: List[ROI] = []
        self.motion_threshold = 30
        self.current_frame = None
        self.mask = None

    def load_mask(self, upload_file, min_region_area: int = 1000):
        """Charge un masque depuis un fichier uploadé"""
        if upload_file is not None:
            # Convertir le fichier uploadé en image
            file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
            mask = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                st.error("Erreur lors du chargement du masque")
                return None

                # Binariser si nécessaire
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            self.mask = binary_mask

            # Trouver les composantes connectées
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )

            # Créer un ROI pour chaque région
            self.rois = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_region_area:
                    continue

                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                region_mask = np.zeros_like(binary_mask)
                region_mask[labels == i] = 255

                contours, _ = cv2.findContours(
                    region_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                epsilon = 0.005 * cv2.arcLength(contours[0], True)
                approx = cv2.approxPolyDP(contours[0], epsilon, True)

                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )

                self.rois.append(ROI(
                    points=approx,
                    mask=region_mask[y:y + h, x:x + w],
                    name=f"ROI_{i}",
                    color=color,
                    bbox=(x, y, w, h)
                ))

            return binary_mask
        return None

    def analyze_frame(self, frame):
        """Analyse un frame uniquement dans les zones ROI"""
        results = {}
        all_motion_viz = np.zeros_like(frame)

        for roi in self.rois:
            x, y, w, h = roi.bbox
            roi_frame = frame[y:y + h, x:x + w]

            masked_roi = cv2.bitwise_and(roi_frame, roi_frame, mask=roi.mask)

            # Utiliser les paramètres spécifiques au ROI
            fg_mask = roi.background_subtractor.apply(
                masked_roi,
                learningRate=roi.params['learning_rate']
            )

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                fg_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            vehicles = []
            for cnt in contours:
                if cv2.contourArea(cnt) > roi.params['min_area']:
                    cnt_adjusted = cnt.copy()
                    cnt_adjusted[:, :, 0] += x
                    cnt_adjusted[:, :, 1] += y
                    vehicles.append(cnt_adjusted)

            total_roi_area = cv2.countNonZero(roi.mask)
            vehicle_area = sum(cv2.contourArea(c) for c in vehicles)
            occupation_rate = (vehicle_area / total_roi_area
                               if total_roi_area > 0 else 0)

            # Mise à jour de la visualisation du mouvement
            roi.motion_frames.append(fg_mask.copy())
            motion_visualization = np.zeros_like(fg_mask, dtype=np.float32)

            for i, mask in enumerate(roi.motion_frames):
                weight = (i + 1) / len(roi.motion_frames)
                motion_visualization += mask * weight

            motion_visualization = np.clip(
                motion_visualization, 0, 255
            ).astype(np.uint8)
            motion_viz_roi = cv2.applyColorMap(
                motion_visualization,
                cv2.COLORMAP_JET
            )

            all_motion_viz[y:y + h, x:x + w] = cv2.bitwise_and(
                motion_viz_roi,
                motion_viz_roi,
                mask=roi.mask
            )

            results[roi.name] = {
                'vehicles_count': len(vehicles),
                'occupation_rate': occupation_rate,
                'traffic_status': self.get_traffic_status(occupation_rate),
                'vehicles': vehicles,
                'color': roi.color
            }

        return results, all_motion_viz

    @staticmethod
    def get_traffic_status(occupation_rate):
        if occupation_rate < 0.1:
            return "Fluide"
        elif occupation_rate < 0.3:
            return "Modéré"
        elif occupation_rate < 0.5:
            return "Dense"
        else:
            return "Embouteillage"

    def draw_analysis(self, frame, results, motion_visualization):
        output = frame.copy()

        alpha = 0.4
        mask = (motion_visualization > self.motion_threshold).any(axis=2)
        output[mask] = cv2.addWeighted(
            output[mask], 1 - alpha,
            motion_visualization[mask], alpha, 0
        )

        # Dessiner les ROIs et leurs noms
        for roi in self.rois:
            cv2.polylines(output, [roi.points], True, roi.color, 2)

            if roi.name in results:
                result = results[roi.name]
                for vehicle in result['vehicles']:
                    cv2.drawContours(output, [vehicle], -1, roi.color, 2)

                    # Calculer le centre du ROI pour le texte
                x, y, w, h = roi.bbox
                center_x = x + w // 2
                center_y = y + h // 2

                # Paramètres du texte
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3
                text = roi.name

                # Obtenir la taille du texte
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                # Calculer la position du texte pour le centrer
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2

                # Ajouter un contour noir par coquetterie
                cv2.putText(output, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
                # Texte blanc
                cv2.putText(output, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

                # Ajouter le timestamp  pour faire sérieux
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Position en bas au centre
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(timestamp, font, font_scale, thickness)[0]
        text_x = (output.shape[1] - text_size[0]) // 2
        text_y = output.shape[0] - 20

        # Ajouter un contour noir pour les mêmes raisons que précédemment
        cv2.putText(output, timestamp, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
        # Texte blanc
        cv2.putText(output, timestamp, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        return output


def main():
    st.set_page_config(layout="wide", page_title="Analyse de Trafic")

    analyzer = TrafficAnalyzerApp()

    st.title("Analyse de Trafic en Temps Réel")

    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("Configuration")

        # Upload du masque
        st.subheader("Masque des zones d'analyse")
        mask_file = st.file_uploader(
            "Charger un masque (image noir et blanc)",
            type=['png', 'jpg', 'jpeg'],
            key="mask_uploader"
        )

        if mask_file:
            mask = analyzer.load_mask(mask_file)
            if mask is not None:
                st.image(mask, caption="Masque chargé", use_container_width=True)

                # Source vidéo
        st.subheader("Source Vidéo")
        source_type = st.radio(
            "Type de source",
            ["Webcam", "Fichier Vidéo", "URL Caméra"],
            key="source_type"
        )

        if source_type == "Fichier Vidéo":
            video_file = st.file_uploader(
                "Charger une vidéo",
                type=['mp4', 'avi'],
                key="video_uploader"
            )
        elif source_type == "URL Caméra":
            camera_url = st.text_input(
                "URL de la caméra",
                key="camera_url"
            )

            # Options d'affichage
        st.subheader("Options d'affichage")
        show_stats = st.checkbox(
            "Afficher les statistiques",
            value=True,
            key="show_stats"
        )

        # Paramètres par ROI
        if analyzer.rois:
            st.subheader("Paramètres des ROI")
            for roi in analyzer.rois:
                with st.expander(f"Paramètres {roi.name}"):
                    roi.params['min_area'] = st.slider(
                        "Surface minimale (px²)",
                        1, 2000, roi.params['min_area'],
                        key=f"min_area_{roi.name}"
                    )
                    roi.params['history'] = st.slider(
                        "Historique",
                        20, 200, roi.params['history'],
                        key=f"history_{roi.name}"
                    )
                    roi.params['threshold'] = st.slider(
                        "Seuil",
                        1, 100, roi.params['threshold'],
                        key=f"threshold_{roi.name}"
                    )
                    roi.params['learning_rate'] = st.slider(
                        "Taux d'apprentissage",
                        0.0, 1.0, roi.params['learning_rate'],
                        key=f"learning_rate_{roi.name}"
                    )

                    # Zone principale  (au centre)
    if not analyzer.rois:
        st.warning("Veuillez d'abord charger un masque pour définir les zones d'analyse")
        return

        # Initialisation de la source vidéo
    try:
        if source_type == "Webcam":
            cap = cv2.VideoCapture(0)
        elif source_type == "Fichier Vidéo" and video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
        elif source_type == "URL Caméra" and camera_url:
            cap = cv2.VideoCapture(camera_url)
        else:
            st.warning("Veuillez configurer une source vidéo")
            return

            # Configuration de l'affichage selon le mode
        if st.session_state.show_stats:
            col1, col2 = st.columns([2, 1])  # Ratio 2:1 pour la vidéo et les stats
            image_placeholder = col1.empty()
            stats_placeholder = col2.empty()
        else:
            image_placeholder = st.empty()

        stop_button = st.button("Arrêter l'analyse", key="stop_button")

        while not stop_button:
            ret, frame = cap.read()

            # Gestion de la lecture en boucle pour les fichiers vidéo
            if not ret:
                if source_type == "Fichier Vidéo":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Retour au début
                    ret, frame = cap.read()
                    if not ret:  # Si toujours pas de frame, on arrête
                        break
                else:
                    break

            results, motion_viz = analyzer.analyze_frame(frame)
            output_frame = analyzer.draw_analysis(frame, results, motion_viz)

            # Conversion pour Streamlit
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            # Mise à jour de l'image
            image_placeholder.image(
                output_frame_rgb,
                caption="Analyse en direct",
                use_container_width=True
            )
            # Mise à jour des statistiques si activées
            if st.session_state.show_stats:
                stats_text = "### Statistiques en temps réel\n```\n"
                for roi_name, result in results.items():
                    stats_text += f"\n{roi_name}:\n"
                    stats_text += f"Véhicules: {result['vehicles_count']}\n"
                    stats_text += f"Occupation: {result['occupation_rate'] * 100:.1f}%\n"
                    stats_text += f"Statut: {result['traffic_status']}\n"
                    stats_text += "-------------------\n"
                stats_text += "```"

                stats_placeholder.markdown(stats_text)

    except Exception as e:
        st.error(f"Erreur: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        if 'tfile' in locals():
            os.unlink(tfile.name)


if __name__ == "__main__":
    main()
