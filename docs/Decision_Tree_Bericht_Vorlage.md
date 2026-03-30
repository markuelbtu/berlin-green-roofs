# Decision Tree Training — Erklärung für deinen Bericht

## Überblick
Der Decision Tree Classifier ist ein überwachtes Lernverfahren, das eine Serie binärer Entscheidungen nutzt, um Objekte in Klassen einzuordnen. Im Kontext dieser Arbeit wird er verwendet, um Gebäudedächer danach zu klassifizieren, ob sie für die Errichtung von Gründächern geeignet sind.

---

## Modellinstanziierung und Parameter

### Code-Ausschnitt
```python
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=8,
    min_samples_leaf=20
)
```

### Parametererklärung

| Parameter | Wert | Bedeutung | Grund |
|-----------|------|-----------|-------|
| **random_state** | 42 | Seed für Zufallszahlengenerator | Gewährleistet **Reproduzierbarkeit** — wiederholte Ausführungen liefern identische Ergebnisse |
| **max_depth** | 8 | Maximale Baumtiefe | Begrenzt die Komplexität und verhindert **Overfitting** (zu gute Anpassung an Trainingsdaten) |
| **min_samples_leaf** | 20 | Minimale Anzahl Samples pro Blatt | Erzeugt größere, **robustere Entscheidungsregeln** statt isolierter Regeln für einzelne Ausreißer |

---

## Trainingsphase

### Code
```python
model.fit(X_train, y_train)
```

### Was passiert?
Der Decision Tree **lernt** die Muster aus den Trainingsdaten durch iterative Aufteilen:

1. **Feature-Auswahl:** Das Modell findet das beste Feature, um die Daten zu teilen (z.B. "suitable_roof_area > 200 m²?")
2. **Rekursives Aufteilen:** Jeder Knoten wird wieder aufgeteilt, bis eine Stoppbedingung erreicht ist
3. **Stoppbedingungen:** 
   - max_depth = 8 erreicht
   - min_samples_leaf = 20 würde unterlaufen (zu wenige Samples übrig)
   - Alle Samples gehören zu einer Klasse

### Input-Daten
- **X_train**: Numerische Features — Dachfläche, Neigung, Ausrichtung, Solarpotenzial, etc.
- **y_train**: Zielwerte — 0 (kein Gründach vorhanden) oder 1 (Gründach vorhanden)

---

## Vorhersage-Phase

### Code
```python
y_pred = model.predict(X_test)
```

### Was passiert?
Das trainierte Modell wird auf **unbekannte Testdaten** angewendet:

1. Jedes Testgebäude folgt dem Entscheidungsbaum von Wurzel zu Blatt
2. Das Modell trifft Ja/Nein-Entscheidungen basierend auf Features
3. Ausgabe: Vorhersage (0 oder 1) für jedes Gebäude

### Output
- **y_pred**: Array mit Vorhersagen für alle Testgebäude

---

## Verbesserung: class_weight='balanced'

### Problem in den Original-Daten
Die Trainingsdaten sind **unausgewogen**:
- ~80% der Gebäude haben **kein Gründach** (Klasse 0)
- ~20% der Gebäude haben **Gründach** (Klasse 1)

### Auswirkung ohne Balancierung
Das Modell könnte "faul" werden und einfach immer "0" vorhersagen:
- Accuracy: 80% (klingt gut!)
- Aber: Findet kein Gründach (Recall für Klasse 1: 0%)

### Lösung
```python
cart_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=8,
    min_samples_leaf=20,
    class_weight='balanced'  # ← Gewichtet Klassen automatisch
)
```

Mit `class_weight='balanced'`:
- Falsche Vorhersagen für Klasse 1 kosten mehr (höheres Gewicht)
- Das Modell konzentriert sich auf die seltene Klasse
- **Bessere Balance** zwischen Precision und Recall

---

## Ablauf zusammengefasst

```
┌─────────────────────────────────────────┐
│ 1. INSTANZIIERUNG                       │
│    DecisionTreeClassifier erstellen     │
│    mit Parametern: depth=8, min=20      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. TRAINING (.fit)                      │
│    Lerne Muster aus X_train, y_train    │
│    → Baum wird konstruiert              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. VORHERSAGE (.predict)                │
│    Wende Baum auf X_test an             │
│    → Ausgabe: y_pred (0 oder 1)         │
└─────────────────────────────────────────┘
```

---

## Für deine Arbeit — Textvorlage

> **Modelltraining:** Der Decision Tree Classifier wurde mit einer maximalen Tiefe von 8 Ebenen und einem Minimum von 20 Samples pro Blattknoten konfiguriert. Diese Parameter wurden gewählt, um eine Überanpassung (Overfitting) an die Trainingsdaten zu vermeiden und gleichzeitig aussagekräftige Entscheidungsregeln zu lernen.
>
> Nach dem Trainieren auf dem Trainingsdatensatz wurde das Modell auf dem reservierten Testdatensatz evaluiert. Um die Auswirkungen unausgeglichener Klassen zu kompensieren, wurde zusätzlich die Variante `class_weight='balanced'` getestet, die der seltenen Klasse (Gründach vorhanden) höheres Gewicht zuweist und somit zu besseren Vorhersagen für die Minderheitsklasse führt.

---

## Zusätzliche Konzepte für Tiefenverständnis

### Entscheidungsbaum als Flowchart
```
              Suitable Area > 150 m²?
                     ├── JA
                     │   └── Ausrichtung SÜD?
                     │       ├── JA → Klasse 1 (Gründach)
                     │       └── NEIN → Klasse 0
                     │
                     └── NEIN
                         └── Neigung < 10°?
                             ├── JA → Klasse 0
                             └── NEIN → Klasse 0
```

### Information Gain (Entropie-Reduktion)
Der Baum wählt Features, die die **beste Trennung** erreichen:
- Gini-Index oder Entropy misst die Unreinheit
- Das Feature mit dem höchsten Information Gain wird als Split gewählt

### Komplexität vs. Performance
- **Tieferer Baum (max_depth=20)**: Höhere Accuracy auf Trainingsdaten, aber Overfitting
- **Flacher Baum (max_depth=3)**: Einfacher, aber zu wenig Trennung
- **max_depth=8**: Balance zwischen Komplexität und Generalisierung

---

## Vergleich: Decision Tree vs. Random Forest

Im Notebook siehst du später auch den Random Forest, der mehrere Decision Trees kombiniert:

| Aspekt | Decision Tree | Random Forest |
|--------|---------------|---------------|
| **Komplexität** | Einfach | Komplex |
| **Interpretierbarkeit** | Sehr gut (Baum ist sichtbar) | Schwach (viele Bäume) |
| **Overfitting-Risiko** | Hoch | Niedrig |
| **Genauigkeit** | Mittel | Hoch |
| **Rechenzeit** | Schnell | Langsam |

Im vorliegenden Projekt erreicht Random Forest bessere Ergebnisse (86% vs. 78% Accuracy).

