import json
import os

nb_path = r"c:\Users\aseba\TIC\emg-classification-knn-svm-ann\notebooks\07_evaluacion_longitudinal.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find ANN cell
ann_idx = -1
eval_idx = -1

for i, cell in enumerate(nb["cells"]):
    if cell.get("metadata", {}).get("id") == "IUKEQFzAz81y":
        ann_idx = i
    elif cell.get("metadata", {}).get("id") == "TH7UJ1-3z81y":
        eval_idx = i

if ann_idx != -1:
    md_cell = {
        "cell_type": "markdown",
        "metadata": {"id": "nueva_celda_cv_md"},
        "source": [
            "### 3.5 Accuracy Realista del Mes 0 (Validación Cruzada)"
        ]
    }
    
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "nueva_celda_cv"},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import cross_val_score, StratifiedKFold\n",
            "\n",
            "print('═' * 60)\n",
            "print('Accuracy REALISTA del Mes 0 — Validación Cruzada (5-Fold)')\n",
            "print('═' * 60)\n",
            "\n",
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
            "\n",
            "# ── kNN CV ────────────────────────────────────────────────────\n",
            "knn_cv_pipe = Pipeline([\n",
            "    ('scaler', StandardScaler()),\n",
            "    ('knn', KNeighborsClassifier(n_neighbors=21, metric='minkowski',\n",
            "                                  p=1, weights='distance'))\n",
            "])\n",
            "knn_cv_scores = cross_val_score(knn_cv_pipe, X_mes0, y_mes0,\n",
            "                                 cv=cv, scoring='accuracy')\n",
            "acc_knn_cv = knn_cv_scores.mean()\n",
            "print(f'\\nkNN  — CV Accuracy: {acc_knn_cv:.4f} (±{knn_cv_scores.std():.4f})')\n",
            "print(f'       Folds: {[f\"{s:.4f}\" for s in knn_cv_scores]}')\n",
            "\n",
            "# ── SVM CV ────────────────────────────────────────────────────\n",
            "svm_cv_pipe = Pipeline([\n",
            "    ('scaler', StandardScaler()),\n",
            "    ('svm', SVC(C=100, kernel='rbf', gamma=0.001))\n",
            "])\n",
            "svm_cv_scores = cross_val_score(svm_cv_pipe, X_mes0, y_mes0,\n",
            "                                 cv=cv, scoring='accuracy')\n",
            "acc_svm_cv = svm_cv_scores.mean()\n",
            "print(f'\\nSVM  — CV Accuracy: {acc_svm_cv:.4f} (±{svm_cv_scores.std():.4f})')\n",
            "print(f'       Folds: {[f\"{s:.4f}\" for s in svm_cv_scores]}')\n",
            "\n",
            "# ── ANN CV (K-Fold manual) ────────────────────────────────────\n",
            "print('\\nANN  — Entrenando 5 folds (esto toma ~1 min)...')\n",
            "ann_cv_scores = []\n",
            "\n",
            "for fold_i, (train_idx, val_idx) in enumerate(cv.split(X_mes0, y_mes0_enc)):\n",
            "    X_tr = X_mes0[train_idx]\n",
            "    X_vl = X_mes0[val_idx]\n",
            "    y_tr = y_mes0_enc[train_idx]\n",
            "    y_vl = y_mes0_enc[val_idx]\n",
            "\n",
            "    sc_fold = StandardScaler()\n",
            "    X_tr_sc = sc_fold.fit_transform(X_tr)\n",
            "    X_vl_sc = sc_fold.transform(X_vl)\n",
            "\n",
            "    ann_fold = build_ann(X_tr_sc.shape[1], N_CLASSES)\n",
            "    ann_fold.fit(\n",
            "        X_tr_sc, to_categorical(y_tr, N_CLASSES),\n",
            "        epochs=200, batch_size=32, verbose=0,\n",
            "        validation_split=0.15,\n",
            "        callbacks=[\n",
            "            EarlyStopping(patience=20, restore_best_weights=True,\n",
            "                          monitor='val_loss'),\n",
            "            ReduceLROnPlateau(patience=7, factor=0.5, monitor='val_loss')\n",
            "        ]\n",
            "    )\n",
            "\n",
            "    y_pred_fold = np.argmax(ann_fold.predict(X_vl_sc, verbose=0), axis=1)\n",
            "    fold_acc = accuracy_score(y_vl, y_pred_fold)\n",
            "    ann_cv_scores.append(fold_acc)\n",
            "    print(f'       Fold {fold_i+1}: {fold_acc:.4f}')\n",
            "\n",
            "ann_cv_scores = np.array(ann_cv_scores)\n",
            "acc_ann_cv = ann_cv_scores.mean()\n",
            "print(f'\\nANN  — CV Accuracy: {acc_ann_cv:.4f} (±{ann_cv_scores.std():.4f})')\n",
            "\n",
            "print(f'\\n{\"─\"*60}')\n",
            "print(f'Resumen Mes 0 (Validación Cruzada):')\n",
            "print(f'  kNN: {acc_knn_cv*100:.2f}%')\n",
            "print(f'  SVM: {acc_svm_cv*100:.2f}%')\n",
            "print(f'  ANN: {acc_ann_cv*100:.2f}%')\n"
        ]
    }
    
    # We want to insert AFTER the ann_idx cell
    nb["cells"].insert(ann_idx + 1, md_cell)
    nb["cells"].insert(ann_idx + 2, code_cell)

    # eval_idx changed because we inserted two cells before it
    if eval_idx != -1:
        eval_idx += 2

if eval_idx != -1:
    new_source = [
        "MESES = sorted(features_by_month.keys(),\n",
        "               key=lambda m: int(m.replace('Mes', '')))\n",
        "\n",
        "results = {model_name: {} for model_name in ['kNN', 'SVM', 'ANN']}\n",
        "\n",
        "for mes in MESES:\n",
        "    X_mes, y_mes = features_by_month[mes]\n",
        "    X_mes_scaled = scaler.transform(X_mes)\n",
        "    y_mes_enc    = le.transform(y_mes)\n",
        "\n",
        "    if mes == 'Mes0':\n",
        "        # ── Usar el accuracy de Validación Cruzada (calculado arriba) ──\n",
        "        results['kNN'][mes] = acc_knn_cv\n",
        "        results['SVM'][mes] = acc_svm_cv\n",
        "        results['ANN'][mes] = acc_ann_cv\n",
        "        print(f'{mes} → kNN: {acc_knn_cv:.4f} | SVM: {acc_svm_cv:.4f} | '\n",
        "              f'ANN: {acc_ann_cv:.4f}  [CV 5-Fold]')\n",
        "    else:\n",
        "        # ── Evaluación normal con los modelos \"maestros\" ──────────────\n",
        "        acc_knn = accuracy_score(y_mes, knn.predict(X_mes))\n",
        "        results['kNN'][mes] = acc_knn\n",
        "\n",
        "        acc_svm = accuracy_score(y_mes, svm.predict(X_mes))\n",
        "        results['SVM'][mes] = acc_svm\n",
        "\n",
        "        y_pred_ann = np.argmax(ann.predict(X_mes_scaled, verbose=0), axis=1)\n",
        "        acc_ann = accuracy_score(y_mes_enc, y_pred_ann)\n",
        "        results['ANN'][mes] = acc_ann\n",
        "\n",
        "        print(f'{mes} → kNN: {acc_knn:.4f} | SVM: {acc_svm:.4f} | ANN: {acc_ann:.4f}')\n",
        "\n",
        "results_df = pd.DataFrame(results, index=MESES)\n",
        "print('\\nTabla de resultados longitudinales:')\n",
        "print((results_df * 100).round(2).to_string())\n"
    ]
    nb["cells"][eval_idx]["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
