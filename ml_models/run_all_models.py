#!/usr/bin/env python3
"""
🚀 Runner - Exécute tous les modèles ML
========================================
Exécute séquentiellement tous les modèles de ML pour l'alternateur.
"""

import subprocess
import sys
from pathlib import Path
import time

# Configuration
MODELS = [
    ("01_XGBoost_Regressor.py", "XGBoost Regressor", True),
    ("02_Random_Forest_Regressor.py", "Random Forest", True),
    ("03_ANN_Neural_Network.py", "ANN (TensorFlow)", False),  # Optionnel si TF pas installé
    ("04_LSTM_TimeSeries.py", "LSTM (TensorFlow)", False),
    ("05_Isolation_Forest_Anomaly.py", "Isolation Forest", True),
    ("06_Autoencoder_Anomaly.py", "Autoencoder (TensorFlow)", False),
    ("07_Health_Index.py", "Health Index", True),
]

def run_model(script_name, model_name, required):
    """Exécute un script de modèle."""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"   ⚠️ Script non trouvé: {script_name}")
        return False
    
    print(f"\n{'='*60}")
    print(f"🔄 Exécution: {model_name}")
    print(f"   Script: {script_name}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            timeout=600  # 10 minutes max par modèle
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n   ✅ {model_name} - SUCCÈS ({elapsed:.1f}s)")
            return True
        else:
            print(f"\n   ❌ {model_name} - ÉCHEC (code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n   ⏱️ {model_name} - TIMEOUT (>600s)")
        return False
    except Exception as e:
        print(f"\n   ❌ {model_name} - ERREUR: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 EXÉCUTION DE TOUS LES MODÈLES ML")
    print("=" * 60)
    print(f"\n📋 {len(MODELS)} modèles à exécuter:")
    for script, name, req in MODELS:
        status = "✓ Requis" if req else "○ Optionnel"
        print(f"   - {name} [{status}]")
    
    results = {}
    total_start = time.time()
    
    for script, name, required in MODELS:
        success = run_model(script, name, required)
        results[name] = success
    
    total_time = time.time() - total_start
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ D'EXÉCUTION")
    print("=" * 60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
    
    print(f"\n   Succès: {success_count}/{total_count}")
    print(f"   Temps total: {total_time/60:.1f} minutes")
    print("=" * 60)
    
    # Vérifier les résultats
    plots_dir = Path(__file__).parent / "plots"
    if plots_dir.exists():
        png_files = list(plots_dir.glob("*.png"))
        csv_files = list(plots_dir.glob("*.csv"))
        print(f"\n📁 Fichiers générés dans plots/:")
        print(f"   - {len(png_files)} fichiers PNG")
        print(f"   - {len(csv_files)} fichiers CSV")
    
    print("\n✅ Exécution terminée!")

if __name__ == "__main__":
    main()
