"""
=============================================================================
INSTRUCCIONES PARA MODIFICAR EL NOTEBOOK 07_evaluacion_longitudinal.ipynb
=============================================================================

Hay 2 cambios que hacer:

CAMBIO 1: AGREGAR una celda nueva DESPUÉS del entrenamiento de ANN (sección 3)
           y ANTES de la evaluación longitudinal (sección 4).
           → Copia el código del bloque "CELDA_NUEVA_CV" abajo.

CAMBIO 2: REEMPLAZAR la celda de evaluación longitudinal (sección 4)
           → Copia el código del bloque "CELDA_MODIFICADA_EVAL" abajo.

Todo lo demás del notebook queda IGUAL.
=============================================================================
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CELDA_NUEVA_CV — Pegar como celda nueva entre sección 3 y sección 4    ║
# ║ Título sugerido (markdown): ## 3.5 Accuracy Realista del Mes 0 (CV)    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --- INICIO CELDA_NUEVA_CV ---

from sklearn.model_selection import cross_val_score, StratifiedKFold

print('═' * 60)
print('Accuracy REALISTA del Mes 0 — Validación Cruzada (5-Fold)')
print('═' * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── kNN CV ────────────────────────────────────────────────────
knn_cv_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=21, metric='minkowski',
                                  p=1, weights='distance'))
])
knn_cv_scores = cross_val_score(knn_cv_pipe, X_mes0, y_mes0,
                                 cv=cv, scoring='accuracy')
acc_knn_cv = knn_cv_scores.mean()
print(f'\nkNN  — CV Accuracy: {acc_knn_cv:.4f} (±{knn_cv_scores.std():.4f})')
print(f'       Folds: {[f"{s:.4f}" for s in knn_cv_scores]}')

# ── SVM CV ────────────────────────────────────────────────────
svm_cv_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(C=100, kernel='rbf', gamma=0.001))
])
svm_cv_scores = cross_val_score(svm_cv_pipe, X_mes0, y_mes0,
                                 cv=cv, scoring='accuracy')
acc_svm_cv = svm_cv_scores.mean()
print(f'\nSVM  — CV Accuracy: {acc_svm_cv:.4f} (±{svm_cv_scores.std():.4f})')
print(f'       Folds: {[f"{s:.4f}" for s in svm_cv_scores]}')

# ── ANN CV (K-Fold manual) ────────────────────────────────────
print('\nANN  — Entrenando 5 folds (esto toma ~1 min)...')
ann_cv_scores = []

for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_mes0, y_mes0_enc)):
    X_tr = X_mes0[train_idx]
    X_vl = X_mes0[val_idx]
    y_tr = y_mes0_enc[train_idx]
    y_vl = y_mes0_enc[val_idx]

    sc_fold = StandardScaler()
    X_tr_sc = sc_fold.fit_transform(X_tr)
    X_vl_sc = sc_fold.transform(X_vl)

    ann_fold = build_ann(X_tr_sc.shape[1], N_CLASSES)
    ann_fold.fit(
        X_tr_sc, to_categorical(y_tr, N_CLASSES),
        epochs=200, batch_size=32, verbose=0,
        validation_split=0.15,
        callbacks=[
            EarlyStopping(patience=20, restore_best_weights=True,
                          monitor='val_loss'),
            ReduceLROnPlateau(patience=7, factor=0.5, monitor='val_loss')
        ]
    )

    y_pred_fold = np.argmax(ann_fold.predict(X_vl_sc, verbose=0), axis=1)
    fold_acc = accuracy_score(y_vl, y_pred_fold)
    ann_cv_scores.append(fold_acc)
    print(f'       Fold {fold_i+1}: {fold_acc:.4f}')

ann_cv_scores = np.array(ann_cv_scores)
acc_ann_cv = ann_cv_scores.mean()
print(f'\nANN  — CV Accuracy: {acc_ann_cv:.4f} (±{ann_cv_scores.std():.4f})')

print(f'\n{"─"*60}')
print(f'Resumen Mes 0 (Validación Cruzada):')
print(f'  kNN: {acc_knn_cv*100:.2f}%')
print(f'  SVM: {acc_svm_cv*100:.2f}%')
print(f'  ANN: {acc_ann_cv*100:.2f}%')

# --- FIN CELDA_NUEVA_CV ---


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CELDA_MODIFICADA_EVAL — Reemplaza la celda de la sección 4             ║
# ║ (la que dice "Mes0 → kNN: 1.0000 | SVM: 1.0000 ...")                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --- INICIO CELDA_MODIFICADA_EVAL ---

MESES = sorted(features_by_month.keys(),
               key=lambda m: int(m.replace('Mes', '')))

results = {model_name: {} for model_name in ['kNN', 'SVM', 'ANN']}

for mes in MESES:
    X_mes, y_mes = features_by_month[mes]
    X_mes_scaled = scaler.transform(X_mes)
    y_mes_enc    = le.transform(y_mes)

    if mes == 'Mes0':
        # ── Usar el accuracy de Validación Cruzada (calculado arriba) ──
        results['kNN'][mes] = acc_knn_cv
        results['SVM'][mes] = acc_svm_cv
        results['ANN'][mes] = acc_ann_cv
        print(f'{mes} → kNN: {acc_knn_cv:.4f} | SVM: {acc_svm_cv:.4f} | '
              f'ANN: {acc_ann_cv:.4f}  [CV 5-Fold]')
    else:
        # ── Evaluación normal con los modelos "maestros" ──────────────
        acc_knn = accuracy_score(y_mes, knn.predict(X_mes))
        results['kNN'][mes] = acc_knn

        acc_svm = accuracy_score(y_mes, svm.predict(X_mes))
        results['SVM'][mes] = acc_svm

        y_pred_ann = np.argmax(ann.predict(X_mes_scaled, verbose=0), axis=1)
        acc_ann = accuracy_score(y_mes_enc, y_pred_ann)
        results['ANN'][mes] = acc_ann

        print(f'{mes} → kNN: {acc_knn:.4f} | SVM: {acc_svm:.4f} | ANN: {acc_ann:.4f}')

results_df = pd.DataFrame(results, index=MESES)
print('\nTabla de resultados longitudinales:')
print((results_df * 100).round(2).to_string())

# --- FIN CELDA_MODIFICADA_EVAL ---
