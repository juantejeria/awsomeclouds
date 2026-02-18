#!/usr/bin/env python3
"""
Script para organizar imágenes en el directorio dataset por clases.

Uso:
    python organize_dataset.py --source_dir /ruta/a/imagenes --target_dir dataset
    
    O si las imágenes ya están en dataset pero mezcladas:
    python organize_dataset.py --source_dir dataset --organize
"""

import os
import shutil
import argparse
from pathlib import Path


def create_class_directories(base_dir, num_classes):
    """Crea directorios para las clases."""
    for i in range(1, num_classes + 1):
        class_dir = os.path.join(base_dir, f'animal_{i}')
        os.makedirs(class_dir, exist_ok=True)
        print(f"✓ Creado directorio: {class_dir}")


def organize_images(source_dir, target_dir, num_classes=None):
    """
    Organiza imágenes desde source_dir a target_dir creando subdirectorios por clase.
    
    Si num_classes no se especifica, crea una clase por cada imagen única.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Extensiones de imagen soportadas
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP'}
    
    # Buscar todas las imágenes
    images = [f for f in source_path.rglob('*') 
              if f.suffix in image_extensions and f.is_file()]
    
    if not images:
        print(f"❌ No se encontraron imágenes en: {source_dir}")
        return
    
    print(f"📸 Encontradas {len(images)} imágenes")
    
    # Si no se especifica número de clases, crear una por imagen
    if num_classes is None:
        num_classes = len(images)
        print(f"📁 Se crearán {num_classes} clases (una por imagen)")
    
    # Crear directorios de clases
    create_class_directories(target_dir, num_classes)
    
    # Distribuir imágenes entre las clases
    images_per_class = len(images) // num_classes
    remainder = len(images) % num_classes
    
    current_class = 1
    images_in_current_class = 0
    
    for idx, img_path in enumerate(images):
        # Determinar a qué clase va esta imagen
        if images_in_current_class >= images_per_class + (1 if current_class <= remainder else 0):
            current_class += 1
            images_in_current_class = 0
        
        class_dir = target_path / f'animal_{current_class}'
        target_file = class_dir / img_path.name
        
        # Si el archivo ya existe, agregar un número
        counter = 1
        while target_file.exists():
            stem = img_path.stem
            suffix = img_path.suffix
            target_file = class_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.copy2(img_path, target_file)
        images_in_current_class += 1
        
        if (idx + 1) % 10 == 0:
            print(f"  Procesadas {idx + 1}/{len(images)} imágenes...")
    
    print(f"\n✅ Organización completada!")
    print(f"   Total de imágenes: {len(images)}")
    print(f"   Clases creadas: {num_classes}")
    
    # Mostrar resumen por clase
    print("\n📊 Resumen por clase:")
    for i in range(1, num_classes + 1):
        class_dir = target_path / f'animal_{i}'
        if class_dir.exists():
            num_images = len([f for f in class_dir.iterdir() if f.is_file()])
            print(f"   animal_{i}: {num_images} imágenes")


def organize_existing_dataset(dataset_dir, num_classes=None):
    """
    Reorganiza imágenes que ya están en el directorio dataset.
    Crea subdirectorios y mueve las imágenes allí.
    """
    dataset_path = Path(dataset_dir)
    
    # Buscar imágenes en el directorio raíz de dataset
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP'}
    images = [f for f in dataset_path.iterdir() 
              if f.suffix in image_extensions and f.is_file()]
    
    if not images:
        print(f"❌ No se encontraron imágenes en el directorio raíz de: {dataset_dir}")
        print("   Las imágenes deben estar en el directorio raíz, no en subdirectorios.")
        return
    
    print(f"📸 Encontradas {len(images)} imágenes en el directorio raíz")
    
    if num_classes is None:
        print("\n⚠️  No se especificó el número de clases.")
        print("   Opciones:")
        print("   1. Especificar manualmente: --num_classes N")
        print("   2. Crear una clase por imagen (no recomendado)")
        response = input("\n   ¿Cuántas clases quieres crear? (presiona Enter para cancelar): ")
        if not response.strip():
            print("❌ Operación cancelada")
            return
        try:
            num_classes = int(response)
        except ValueError:
            print("❌ Número inválido")
            return
    
    # Crear directorios de clases
    create_class_directories(dataset_dir, num_classes)
    
    # Distribuir imágenes
    images_per_class = len(images) // num_classes
    remainder = len(images) % num_classes
    
    current_class = 1
    images_in_current_class = 0
    
    for idx, img_path in enumerate(images):
        if images_in_current_class >= images_per_class + (1 if current_class <= remainder else 0):
            current_class += 1
            images_in_current_class = 0
        
        class_dir = dataset_path / f'animal_{current_class}'
        target_file = class_dir / img_path.name
        
        # Si el archivo ya existe, agregar un número
        counter = 1
        while target_file.exists():
            stem = img_path.stem
            suffix = img_path.suffix
            target_file = class_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.move(str(img_path), str(target_file))
        images_in_current_class += 1
        
        if (idx + 1) % 10 == 0:
            print(f"  Movidas {idx + 1}/{len(images)} imágenes...")
    
    print(f"\n✅ Organización completada!")
    print(f"   Total de imágenes: {len(images)}")
    print(f"   Clases creadas: {num_classes}")
    
    # Mostrar resumen por clase
    print("\n📊 Resumen por clase:")
    for i in range(1, num_classes + 1):
        class_dir = dataset_path / f'animal_{i}'
        if class_dir.exists():
            num_images = len([f for f in class_dir.iterdir() if f.is_file()])
            print(f"   animal_{i}: {num_images} imágenes")


def main():
    parser = argparse.ArgumentParser(
        description='Organiza imágenes en el directorio dataset por clases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  1. Organizar imágenes desde otro directorio:
     python organize_dataset.py --source_dir /ruta/a/mis/imagenes --target_dir dataset --num_classes 5

  2. Reorganizar imágenes que ya están en dataset/:
     python organize_dataset.py --organize --num_classes 3

  3. Crear estructura vacía para 10 clases:
     python organize_dataset.py --create_structure --num_classes 10
        """
    )
    
    parser.add_argument('--source_dir', type=str, 
                       help='Directorio fuente con las imágenes')
    parser.add_argument('--target_dir', type=str, default='dataset',
                       help='Directorio destino (default: dataset)')
    parser.add_argument('--num_classes', type=int,
                       help='Número de clases a crear')
    parser.add_argument('--organize', action='store_true',
                       help='Reorganizar imágenes que ya están en dataset/')
    parser.add_argument('--create_structure', action='store_true',
                       help='Solo crear la estructura de directorios sin mover imágenes')
    
    args = parser.parse_args()
    
    if args.create_structure:
        if not args.num_classes:
            print("❌ Debes especificar --num_classes cuando usas --create_structure")
            return
        create_class_directories(args.target_dir, args.num_classes)
        print(f"\n✅ Estructura creada para {args.num_classes} clases en {args.target_dir}/")
        return
    
    if args.organize:
        organize_existing_dataset(args.target_dir, args.num_classes)
    elif args.source_dir:
        organize_images(args.source_dir, args.target_dir, args.num_classes)
    else:
        print("❌ Debes especificar --source_dir o --organize")
        print("   Usa --help para ver las opciones")


if __name__ == '__main__':
    main()

