# scripts/ — Generación de Datos y Entrenamiento del Modelo

## ¿Qué hay aquí?

Scripts de utilidad para preparar datos y entrenar el clasificador de señas.

## Flujo recomendado

```
1. Generar datos sintéticos
   python scripts/generate_data.py
   → crea data/training/synthetic_asl.csv, synthetic_lsc.csv, synthetic_bsl.csv

2. Entrenar el modelo
   python scripts/train_model.py
   → crea data/models/classifier_asl.pkl, etc.
   → imprime métricas de evaluación (accuracy, classification report)

3. (Futuro) Recolectar datos reales
   python scripts/collect_data.py   # por crear
   → interfaz con cámara para grabar landmarks de señas reales
   → guarda en data/training/real_<country>.csv

4. (Futuro) Reentrenar con datos mixtos
   python scripts/train_model.py --data=real
```

## generate_data.py

Genera datos de entrenamiento sintéticos basados en los templates de señas
definidos en core/classifier.py. Para cada seña genera N muestras con
perturbación gaussiana (simula variación entre usuarios).

Parámetros configurables en el script:
- `N_SAMPLES_PER_SIGN = 200` — muestras por seña
- `NOISE_LEVEL = 0.08` — nivel de ruido Gaussiano

## train_model.py

Entrena un SVM con kernel RBF para cada idioma disponible.
Evalúa con cross-validation y guarda el modelo con joblib.

Salida esperada:
```
Training ASL model...
  Training samples: 5200
  Cross-val accuracy: 0.87 ± 0.03
  Test accuracy: 0.89
  Saved to data/models/classifier_asl.pkl

Training LSC model...
  ...
```

## Cómo mejorar el modelo en el futuro

El modelo con datos sintéticos tiene ~85-90% de precisión en condiciones ideales.
Con datos reales de 5-10 usuarios se puede alcanzar >95%.

Plan para mejorar:
1. Crear collect_data.py con interfaz simple de grabación
2. Recolectar 50-100 muestras por seña por usuario
3. Agregar augmentation (espejar landmarks para mano izquierda/derecha)
4. Probar Random Forest o MLP si SVM no es suficiente
5. Para señas dinámicas: implementar captura de secuencias y modelo LSTM
