#!/usr/bin/env python3
"""
Script pour extraire les images des notebooks Jupyter et les sauvegarder
comme fichiers PNG dans le dossier plots/
"""

import json
import base64
import os
from pathlib import Path

# Configuration
NOTEBOOK_DIR = Path(__file__).parent
PLOTS_DIR = NOTEBOOK_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Mapping des notebooks aux noms de plots
PLOT_MAPPINGS = {
    "01_APM_Alternateur_ML_EDA.ipynb": [
        "01_distribution_variables_principales",
        "01_temperatures_stator",
        "01_matrice_correlation",
        "01_evolution_temporelle",
        "01_analyse_horaire",
        "01_boxplots_outliers",
        "01_boxplots_temperatures",
        "01_scatter_relations",
        "01_pairplot"
    ],
    "02_APM_Alternateur_10min_ML_EDA.ipynb": [
        "02_distribution_variables_principales",
        "02_temperatures_stator",
        "02_matrice_correlation",
        "02_evolution_temporelle",
        "02_analyse_horaire",
        "02_boxplots_outliers",
        "02_boxplots_temperatures",
        "02_scatter_relations",
        "02_pairplot"
    ],
    "03_APM_Chart_ML_EDA.ipynb": [
        "03_distribution_variables_principales",
        "03_temperatures_egt",
        "03_matrice_correlation",
        "03_evolution_temporelle",
        "03_analyse_horaire",
        "03_boxplots_outliers",
        "03_scatter_relations",
        "03_pairplot"
    ],
    "04_APM_Chart_10min_ML_EDA.ipynb": [
        "04_distribution_variables_principales",
        "04_temperatures_egt",
        "04_matrice_correlation",
        "04_evolution_temporelle",
        "04_analyse_horaire",
        "04_boxplots_outliers",
        "04_scatter_relations",
        "04_pairplot"
    ],
    "05_TG1_Sousse_ML_EDA.ipynb": [
        "05_distribution_courants",
        "05_distribution_charges",
        "05_correlation_canaux",
        "05_correlation_ch1",
        "05_evolution_temporelle",
        "05_analyse_mensuelle",
        "05_boxplots_courants",
        "05_hexbin_courant_decharge",
        "05_comparaison_canaux"
    ],
    "06_TG1_Sousse_1min_ML_EDA.ipynb": [
        "06_distribution_courants",
        "06_distribution_charges",
        "06_correlation_canaux",
        "06_correlation_ch1",
        "06_evolution_temporelle",
        "06_analyse_mensuelle",
        "06_boxplots_courants",
        "06_hexbin_courant_decharge",
        "06_comparaison_canaux"
    ]
}


def extract_images_from_notebook(notebook_path: Path, plot_names: list) -> int:
    """
    Extrait les images d'un notebook et les sauvegarde en PNG.
    
    Args:
        notebook_path: Chemin vers le notebook
        plot_names: Liste des noms de fichiers pour les images
        
    Returns:
        Nombre d'images extraites
    """
    if not notebook_path.exists():
        print(f"  ⚠️  Notebook non trouvé: {notebook_path.name}")
        return 0
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Extraire toutes les images des outputs
    images = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        outputs = cell.get('outputs', [])
        for output in outputs:
            # Vérifier les données d'image
            data = output.get('data', {})
            if 'image/png' in data:
                img_data = data['image/png']
                # Peut être une string ou une liste de strings
                if isinstance(img_data, list):
                    img_data = ''.join(img_data)
                images.append(img_data)
    
    # Sauvegarder les images
    saved_count = 0
    for i, (img_b64, name) in enumerate(zip(images, plot_names)):
        try:
            img_bytes = base64.b64decode(img_b64)
            output_path = PLOTS_DIR / f"{name}.png"
            with open(output_path, 'wb') as f:
                f.write(img_bytes)
            saved_count += 1
        except Exception as e:
            print(f"  ⚠️  Erreur sauvegarde {name}: {e}")
    
    # Si plus d'images que de noms, sauvegarder avec numéros
    if len(images) > len(plot_names):
        prefix = notebook_path.stem.split('_')[0]
        for i, img_b64 in enumerate(images[len(plot_names):], start=len(plot_names)+1):
            try:
                img_bytes = base64.b64decode(img_b64)
                output_path = PLOTS_DIR / f"{prefix}_plot_{i}.png"
                with open(output_path, 'wb') as f:
                    f.write(img_bytes)
                saved_count += 1
            except:
                pass
    
    return saved_count


def main():
    print("=" * 60)
    print("EXTRACTION DES PLOTS DES NOTEBOOKS")
    print("=" * 60)
    
    total_images = 0
    
    for notebook_name, plot_names in PLOT_MAPPINGS.items():
        notebook_path = NOTEBOOK_DIR / notebook_name
        print(f"\n📓 {notebook_name}")
        
        count = extract_images_from_notebook(notebook_path, plot_names)
        total_images += count
        print(f"   ✅ {count} images extraites")
    
    print("\n" + "=" * 60)
    print(f"Total: {total_images} images sauvegardées dans plots/")
    print("=" * 60)


if __name__ == "__main__":
    main()
