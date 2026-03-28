# 🧠 Can Neural Networks Learn Prime Numbers?

This project explores a fundamental question:

> Can neural networks actually learn the structure of prime numbers?

---

## 🎯 Motivation

Prime numbers are not defined by simple patterns — they follow strict **number-theoretic rules based on divisibility**.

This project began as an MSc Semester 1 experiment and was later extended to investigate:

1. Whether neural networks can learn primality from data
2. Whether different architectures improve performance
3. Whether models generalize to unseen numerical ranges

---

## ⚙️ Approach

Each integer is represented using:

1. **Binary representation**

   * Fixed-length encoding (~21 bits)

2. **Modular features**

   * Remainders with small primes (mod 2, 3, 5, ..., 31)

---

### Models evaluated:

1. **MLP (Multi-Layer Perceptron)**

   * Binary only
   * Binary + modular
   * Modular only

2. **CNN (1D)**

3. **Transformer**

---

## 📊 Key Observations

### 1. ❌ Binary-only models fail

* High recall but very low precision
* Large number of false positives
* Unable to capture divisibility structure

👉 Neural networks fail to infer primality from binary representation alone.

---

### 2. ❌ CNNs and Transformers do not improve results

* No meaningful gain over simple MLP
* Still fail to learn arithmetic relationships

👉 Increasing architectural complexity does not address the core issue.

---

### 3. ⚠️ Modular features improve performance — but not meaningfully

The improvement obtained using modular features does not indicate that the model has learned the concept of primality.

Instead, the input representation already contains direct information about divisibility. Since primality is fundamentally defined in terms of divisibility by smaller integers, providing modular residues effectively injects the underlying mathematical structure into the input.

In this setting, the model is not discovering the rule governing prime numbers. It is simply learning how to combine already relevant signals that encode divisibility properties.

Therefore, the apparent performance gain reflects the quality of the engineered features rather than the model’s ability to infer number-theoretic structure.

---

## 🧠 Core Insight

> Neural networks do not naturally learn number-theoretic structure from raw representations.

There is a fundamental mismatch:

* Input representation → **positional (binary encoding)**
* True structure → **divisibility (modular arithmetic)**

---

## 🚫 Why this matters

For large integers:

* Primality testing is computationally non-trivial
* Providing modular features already requires partial computation

👉 Therefore, feature engineering is not a true solution — it bypasses the original problem.

---

## 📈 Results Summary

* Binary models → high recall, very low precision
* CNN / Transformer → similar failure behavior
* Modular features → strong metrics but not genuine learning

👉 Detailed plots and comparisons are available in the `results/` folder

---

## 📁 Project Structure

```id="1y7vba"
prime-number-ml/
│── notebooks/        # Experiments and model runs
│── src/              # Dataset and model definitions
│── results/          # Evaluation plots and comparisons
│── data/             # Dataset instructions (not included)
```

---

## ⚠️ Dataset

Dataset is not included due to size.

To generate the dataset:

```id="cz3zjv"
notebooks/dataset_generation.ipynb
```

* Train range: 2 → 750,000
* Test range: 750,001 → 1,250,000

---

## 🔍 Interpretation

This project suggests that:

1. Neural networks struggle with discrete mathematical rules like primality
2. Learning arithmetic structure is fundamentally different from pattern recognition
3. Additional model complexity does not solve this limitation

---

## 🚀 Possible Directions

* Hybrid symbolic + neural approaches
* Explicit modeling of divisibility
* Alternative representations of integers

---

## 👤 Author

**Soumodip Dey**

---

⭐ If you found this interesting, consider starring the repository.
