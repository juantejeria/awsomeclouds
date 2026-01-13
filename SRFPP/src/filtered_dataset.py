"""
Dataset filtrado que solo incluye imágenes donde YOLO detecta animales.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image

from .face_detection import detect_animal_face


class FilteredImageFolder(datasets.ImageFolder):
    """
    ImageFolder que filtra imágenes basándose en detecciones YOLO.
    Solo incluye imágenes donde se detecta un animal.
    """
    
    def __init__(
        self,
        root: str | Path,
        filter_with_yolo: bool = True,
        min_confidence: float = 0.3,
        require_features: list[str] | None = None,
        *args,
        **kwargs,
    ):
        """
        Args:
            root: Directorio raíz del dataset
            filter_with_yolo: Si True, filtra imágenes sin detecciones
            min_confidence: Confianza mínima para detecciones YOLO
            require_features: Características requeridas (ej: ["cow", "animal"])
            *args, **kwargs: Argumentos adicionales para ImageFolder
        """
        # Primero cargar el dataset completo
        super().__init__(root, *args, **kwargs)
        
        if not filter_with_yolo:
            # Sin filtrado, usar todas las muestras
            self.filtered_samples = self.samples
            self.filtered_targets = [y for _, y in self.samples]
            return
        
        # Filtrar muestras usando YOLO
        print(f"\n🔍 Filtrando dataset con YOLO...")
        print(f"   Total de imágenes antes del filtrado: {len(self.samples)}")
        
        filtered_samples = []
        filtered_targets = []
        skipped_count = 0
        
        for idx, (img_path, target) in enumerate(self.samples):
            try:
                # Cargar imagen
                img = Image.open(img_path).convert('RGB')
                
                # Verificar detección con YOLO
                # Por defecto, si no se especifican características, requerir específicamente vacas
                # ya que estamos entrenando para reconocimiento de individuos de vacas
                if require_features is None:
                    # Requerir específicamente vacas para entrenamiento de individuos
                    from .face_detection import detect_cow_specifically
                    has_animal = detect_cow_specifically(img, min_confidence=min_confidence)
                else:
                    from .face_detection import has_valid_animal_frame
                    has_animal = has_valid_animal_frame(
                        img,
                        require_features=require_features,
                        min_confidence=min_confidence,
                    )
                
                if has_animal:
                    filtered_samples.append((img_path, target))
                    filtered_targets.append(target)
                else:
                    skipped_count += 1
                    
                # Mostrar progreso cada 10 imágenes
                if (idx + 1) % 10 == 0:
                    print(f"   Procesadas {idx + 1}/{len(self.samples)} imágenes...", end='\r')
                    
            except Exception as e:
                # Si hay error, incluir la imagen de todas formas (mejor tener datos)
                print(f"   Advertencia: Error procesando {img_path}: {e}")
                filtered_samples.append((img_path, target))
                filtered_targets.append(target)
        
        print(f"\n   ✅ Filtrado completado:")
        print(f"   • Imágenes incluidas: {len(filtered_samples)}")
        print(f"   • Imágenes omitidas: {skipped_count}")
        print(f"   • Tasa de retención: {len(filtered_samples)/len(self.samples)*100:.1f}%")
        
        # Actualizar muestras y targets
        self.filtered_samples = filtered_samples
        self.filtered_targets = filtered_targets
        
        # Sobrescribir samples y targets para que ImageFolder use las filtradas
        self.samples = filtered_samples
        self.targets = filtered_targets
    
    def __len__(self) -> int:
        """Retorna el número de muestras filtradas."""
        return len(self.filtered_samples)
    
    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Obtiene una muestra filtrada."""
        path, target = self.filtered_samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

