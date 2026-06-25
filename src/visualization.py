import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

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


# ============================================================================
# Visualización Intrapersonal
# ============================================================================

def get_user_color_map(user_ids: List[str]) -> Dict[str, tuple]:
    """
    Genera un diccionario {user_id: color} consistente usando el colormap tab20.
    El mismo user_id siempre recibe el mismo color en las 3 gráficas.

    Args:
        user_ids: lista de IDs de usuarios (se ordenan numéricamente)

    Returns:
        dict {user_id: (R, G, B, A)}
    """
    import matplotlib.cm as cm

    # Ordenar numéricamente por el número en el user_id
    sorted_ids = sorted(user_ids, key=lambda uid: int(uid.replace('user', '')))
    cmap = cm.get_cmap('tab20', max(20, len(sorted_ids)))
    color_map = {uid: cmap(i) for i, uid in enumerate(sorted_ids)}
    return color_map


def plot_intrapersonal_accuracy(
    results_df: pd.DataFrame,
    model_name: str,
    user_colors: Optional[Dict[str, tuple]] = None,
    y_lim: Optional[tuple] = None,
    save_path: Optional[str] = None,
):
    """
    Genera una gráfica de degradación de accuracy intrapersonal para un modelo.

    Args:
        results_df: DataFrame largo con columnas:
                    user_id, model, mes, mes_num, accuracy
        model_name: 'kNN', 'SVM' o 'ANN'
        user_colors: dict {user_id: color} (de get_user_color_map).
                     Si None, se genera automáticamente.
        y_lim: tupla (ymin, ymax) para el eje Y. Si None, usa (0, 100).
        save_path: ruta para guardar la figura (PNG). Si None, no guarda.
    """
    # Filtrar por modelo
    df_model = results_df[results_df['model'] == model_name].copy()

    if df_model.empty:
        print(f"⚠️  No hay datos para el modelo '{model_name}'")
        return

    # Obtener usuarios y meses ordenados
    user_ids = sorted(df_model['user_id'].unique(),
                      key=lambda uid: int(uid.replace('user', '')))
    meses_ordered = df_model.sort_values('mes_num')['mes'].unique()

    # Generar colores si no se proporcionan
    if user_colors is None:
        user_colors = get_user_color_map(user_ids)

    fig, ax = plt.subplots(figsize=(14, 7))

    # ── Líneas individuales por usuario ──
    for uid in user_ids:
        df_user = df_model[df_model['user_id'] == uid].sort_values('mes_num')
        ax.plot(
            df_user['mes'], df_user['accuracy'],
            marker='o', markersize=4, linewidth=1.0,
            color=user_colors.get(uid, 'gray'),
            alpha=0.6, label=uid,
        )

    # ── Línea promedio + banda ±1σ ──
    stats = df_model.groupby('mes_num').agg(
        mean_acc=('accuracy', 'mean'),
        std_acc=('accuracy', 'std'),
        mes_label=('mes', 'first'),
    ).sort_index()

    ax.plot(
        stats['mes_label'], stats['mean_acc'],
        color='black', linewidth=3.0, marker='s', markersize=7,
        label='Promedio', zorder=10,
    )
    ax.fill_between(
        stats['mes_label'],
        stats['mean_acc'] - stats['std_acc'],
        stats['mean_acc'] + stats['std_acc'],
        color='black', alpha=0.12, zorder=5,
        label='± 1 Desv. Est.',
    )

    # ── Ejes y etiquetas ──
    if y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim(0, 100)

    ax.set_xlabel('Sesión (Mes)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Degradación de Accuracy Intrapersonal — {model_name}',
                 fontsize=14, fontweight='bold')
    ax.text(
        0.5, 1.02,
        'Modelos entrenados únicamente con datos de Mes0 de cada usuario',
        transform=ax.transAxes, fontsize=10, ha='center', va='bottom',
        style='italic', color='gray',
    )

    ax.grid(True, linestyle='--', alpha=0.4)

    # ── Leyenda externa ──
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.10),
        ncol=7,
        fontsize=8,
        frameon=True,
        fancybox=True,
        shadow=False,
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Figura guardada en: {save_path}")

    plt.show()
