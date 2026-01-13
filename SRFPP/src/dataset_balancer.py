"""
Sistema de balanceo de dataset eliminando frames similares.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from .io_utils import read_json


class DatasetBalancer:
    """
    Balancea un dataset eliminando frames similares de las clases con más muestras.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: Dispositivo para calcular features ('cpu' o 'cuda')
        """
        self.device = torch.device(device)
        self._feature_extractor = None
        self._transform = None
    
    def _get_feature_extractor(self):
        """Carga el modelo pre-entrenado para extraer features."""
        if self._feature_extractor is None:
            # Usar ResNet18 pre-entrenado para extraer features
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # Remover la capa final de clasificación, quedarnos con las features
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval()
            model.to(self.device)
            self._feature_extractor = model
            
            # Transform para preprocesar imágenes
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        return self._feature_extractor, self._transform
    
    def extract_features(self, image_path: Path) -> np.ndarray:
        """
        Extrae features de una imagen usando ResNet pre-entrenado.
        
        Args:
            image_path: Ruta a la imagen
        
        Returns:
            Vector de features normalizado
        """
        try:
            img = Image.open(image_path).convert('RGB')
            model, transform = self._get_feature_extractor()
            
            # Preprocesar imagen
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            # Extraer features
            with torch.no_grad():
                features = model(img_tensor)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
                features = F.normalize(features, p=2, dim=1)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extrayendo features de {image_path}: {e}")
            # Retornar vector cero si hay error
            return np.zeros(512)
    
    def compute_similarity_matrix(self, image_paths: list[Path]) -> np.ndarray:
        """
        Calcula matriz de similitud (cosine similarity) entre todas las imágenes.
        
        Args:
            image_paths: Lista de rutas a imágenes
        
        Returns:
            Matriz de similitud (N x N) donde valores más altos = más similares
        """
        print(f"  Extrayendo features de {len(image_paths)} imágenes...")
        features = []
        
        for idx, img_path in enumerate(image_paths):
            if (idx + 1) % 10 == 0:
                print(f"    Procesadas {idx + 1}/{len(image_paths)}...", end='\r')
            feat = self.extract_features(img_path)
            features.append(feat)
        
        print(f"    Completado: {len(image_paths)} imágenes procesadas")
        
        features = np.array(features)
        
        # Calcular cosine similarity: dot product de vectores normalizados
        similarity_matrix = np.dot(features, features.T)
        
        return similarity_matrix
    
    def select_diverse_samples(
        self,
        image_paths: list[Path],
        target_count: int,
        similarity_threshold: float = 0.85,
    ) -> list[Path]:
        """
        Selecciona muestras diversas eliminando las más similares.
        
        Args:
            image_paths: Lista de rutas a imágenes
            target_count: Número objetivo de muestras a mantener
            similarity_threshold: Umbral de similitud para considerar imágenes como similares
        
        Returns:
            Lista de rutas a imágenes seleccionadas (diversas)
        """
        if len(image_paths) <= target_count:
            return image_paths
        
        print(f"  Seleccionando {target_count} muestras diversas de {len(image_paths)}...")
        
        # Calcular matriz de similitud
        similarity_matrix = self.compute_similarity_matrix(image_paths)
        
        # Convertir a lista de índices para trabajar más fácilmente
        indices = list(range(len(image_paths)))
        selected_indices = []
        
        # Estrategia: empezar con la imagen más "central" (mayor similitud promedio)
        # y luego agregar imágenes que sean menos similares a las ya seleccionadas
        
        # Calcular similitud promedio de cada imagen con todas las demás
        avg_similarities = similarity_matrix.mean(axis=1)
        
        # Empezar con la imagen más representativa (mayor similitud promedio)
        # Esto asegura que mantenemos una imagen "típica" de la clase
        start_idx = np.argmax(avg_similarities)
        selected_indices.append(start_idx)
        
        # Iterativamente agregar imágenes que sean menos similares a las seleccionadas
        while len(selected_indices) < target_count:
            if len(selected_indices) == 0:
                break
            
            # Para cada imagen no seleccionada, calcular su similitud máxima
            # con las imágenes ya seleccionadas
            max_similarities = []
            for idx in indices:
                if idx in selected_indices:
                    continue
                
                # Similitud máxima con cualquier imagen seleccionada
                max_sim = max([similarity_matrix[idx][sel_idx] for sel_idx in selected_indices])
                max_similarities.append((idx, max_sim))
            
            if not max_similarities:
                break
            
            # Seleccionar la imagen con menor similitud máxima (más diferente)
            next_idx, _ = min(max_similarities, key=lambda x: x[1])
            selected_indices.append(next_idx)
            
            if len(selected_indices) % 10 == 0:
                print(f"    Seleccionadas {len(selected_indices)}/{target_count}...", end='\r')
        
        print(f"    Completado: {len(selected_indices)} muestras seleccionadas")
        
        # Retornar las imágenes seleccionadas
        selected_paths = [image_paths[idx] for idx in selected_indices]
        
        return selected_paths
    
    def balance_dataset(
        self,
        data_dir: Path,
        min_samples_per_class: int | None = None,
        similarity_threshold: float = 0.85,
    ) -> dict[str, Any]:
        """
        Balancea el dataset eliminando frames similares.
        
        Args:
            data_dir: Directorio del dataset (estructura ImageFolder)
            min_samples_per_class: Número mínimo de muestras por clase (None = usar el mínimo encontrado)
            similarity_threshold: Umbral de similitud para considerar imágenes similares
        
        Returns:
            Diccionario con estadísticas del balanceo
        """
        from torchvision import datasets
        
        # Cargar dataset
        dataset = datasets.ImageFolder(str(data_dir), transform=None)
        
        # Agrupar imágenes por clase
        images_by_class = defaultdict(list)
        for img_path, class_idx in dataset.samples:
            class_name = dataset.classes[class_idx]
            images_by_class[class_name].append(Path(img_path))
        
        # Encontrar el número mínimo de muestras
        if min_samples_per_class is None:
            min_samples_per_class = min(len(imgs) for imgs in images_by_class.values())
        
        print(f"\n📊 Balanceando dataset...")
        print(f"   Número mínimo de muestras por clase: {min_samples_per_class}")
        print(f"   Clases encontradas: {len(images_by_class)}")
        
        stats = {
            "original_counts": {},
            "final_counts": {},
            "removed_counts": {},
            "min_samples": min_samples_per_class,
        }
        
        # Balancear cada clase
        for class_name, image_paths in images_by_class.items():
            original_count = len(image_paths)
            stats["original_counts"][class_name] = original_count
            
            if original_count <= min_samples_per_class:
                # Ya tiene el número correcto o menos, mantener todas
                final_paths = image_paths
                removed_count = 0
            else:
                # Seleccionar muestras diversas
                final_paths = self.select_diverse_samples(
                    image_paths,
                    target_count=min_samples_per_class,
                    similarity_threshold=similarity_threshold,
                )
                removed_count = original_count - len(final_paths)
            
            stats["final_counts"][class_name] = len(final_paths)
            stats["removed_counts"][class_name] = removed_count
            
            print(f"   {class_name}: {original_count} → {len(final_paths)} muestras (eliminadas: {removed_count})")
        
        return stats


def balance_dataset_by_removing_similar(
    data_dir: Path,
    output_dir: Path | None = None,
    min_samples_per_class: int | None = None,
    similarity_threshold: float = 0.85,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Balancea un dataset eliminando frames similares y opcionalmente guarda el resultado.
    
    Args:
        data_dir: Directorio del dataset original
        output_dir: Directorio donde guardar el dataset balanceado (None = no guardar, solo retornar stats)
        min_samples_per_class: Número mínimo de muestras por clase
        similarity_threshold: Umbral de similitud
        device: Dispositivo para calcular features
    
    Returns:
        Estadísticas del balanceo
    """
    balancer = DatasetBalancer(device=device)
    stats = balancer.balance_dataset(
        data_dir=data_dir,
        min_samples_per_class=min_samples_per_class,
        similarity_threshold=similarity_threshold,
    )
    
    # Si se especifica output_dir, crear el dataset balanceado
    if output_dir is not None:
        from torchvision import datasets
        import shutil
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = datasets.ImageFolder(str(data_dir), transform=None)
        images_by_class = defaultdict(list)
        for img_path, class_idx in dataset.samples:
            class_name = dataset.classes[class_idx]
            images_by_class[class_name].append(Path(img_path))
        
        print(f"\n💾 Guardando dataset balanceado en {output_dir}...")
        
        for class_name, image_paths in images_by_class.items():
            original_count = len(image_paths)
            
            if original_count <= stats["min_samples"]:
                final_paths = image_paths
            else:
                final_paths = balancer.select_diverse_samples(
                    image_paths,
                    target_count=stats["min_samples"],
                    similarity_threshold=similarity_threshold,
                )
            
            # Crear carpeta de clase en output_dir
            class_dir = output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Copiar imágenes seleccionadas
            for img_path in final_paths:
                dest_path = class_dir / img_path.name
                shutil.copy2(img_path, dest_path)
            
            print(f"   {class_name}: {len(final_paths)} imágenes copiadas")
    
    return stats

