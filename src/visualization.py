import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List

def plot_signal(signal: np.ndarray, title: str = 'EMG Signal', fs: float = 200.0, save_path: str = None):
    """
    Grafica todos los canales de una señal EMG [T, C] en subplots apilados.
    Args:
        signal: array [T, C]
        title: título del gráfico
        fs: frecuencia de muestreo en Hz (para el eje X en segundos)
    """
    n_samples, n_channels = signal.shape
    time = np.arange(n_samples) / fs

    fig, axes = plt.subplots(n_channels, 1, figsize=(12, n_channels * 1.5), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, signal[:, i], linewidth=0.8)
        ax.set_ylabel(f'CH{i+1}', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Tiempo (s)')
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str = 'Confusion Matrix', save_path: str = None):

    """
    Dibuja una matriz de confusión usando Seaborn Heatmap.
    """
    plt.figure(figsize=(8, 6))
    
    # Calcular porcentajes por fila para mostrar
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.ylabel('Etiqueta Verdadera (Ground Truth)')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_accuracy_over_sessions(df_results: pd.DataFrame, save_path: str = None):
    """
    Gráfica de líneas mostrando la evolución de la exactitud a lo largo de las sesiones multi-temporales.
    df_results debe tener columnas: ['session', 'model', 'accuracy']
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results, x='session', y='accuracy', hue='model', marker='o', linewidth=2, markersize=8)
    
    plt.title("Evolución del Rendimiento Longitudinal por Sesión (Mes 0 al Mes 6)")
    plt.ylabel("Exactitud (Accuracy)")
    plt.xlabel("Sesión (Meses)")
    plt.xticks(sorted(df_results['session'].unique()))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Modelo')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_metrics_comparison_bar(df_results: pd.DataFrame, session: int, save_path: str = None):
    """
    Compara modelos en una sola sesión específica usando un barplot de métricas múltiples.
    df_results debe tener formato tabla de métricas o ser filtrada previamente.
    """
    # Filtrar la sesión deseada
    df_session = df_results[df_results['session'] == session].copy()
    
    # "Derretir" el dataframe para que Seaborn pueda graficar varias métricas juntas
    df_melt = df_session.melt(id_vars=['model'], value_vars=['accuracy', 'precision', 'recall', 'f1_score'], 
                              var_name='metric', value_name='score')
                              
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x='metric', y='score', hue='model')
    
    plt.title(f"Comparación de Métricas - Sesión {session}")
    plt.ylim(0, 1.05)
    plt.ylabel("Puntuación")
    plt.xlabel("Métrica")
    plt.legend(title='Modelo')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_learning_curve_keras(history, save_path: str = None):
    """
    Grafica la curva de pérdida (loss) y exactitud (accuracy) del entrenamiento de un modelo ANN (Keras).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Entrenamiento')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validación')
    ax1.set_title("Curva de Pérdida (Loss)")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Entrenamiento')
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Validación')
    ax2.set_title("Precisión (Accuracy)")
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
