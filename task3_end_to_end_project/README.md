# Task 3 – End-to-End Data Science Project (Iris Classification)

This mini project demonstrates a full data science workflow, from data collection and preprocessing
to model deployment via a Flask API.

## Steps

1. **Create & activate a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r ../requirements.txt
   ```

3. **Train the model**

   ```bash
   python model_training.py
   ```

   This will create `model.joblib` in the same folder.

4. **Run the Flask API**

   ```bash
   python app.py
   ```

   The API will be available at `http://127.0.0.1:5000/`.

5. **Test the API**

   Example with `curl`:

   ```bash
   curl -X POST http://127.0.0.1:5000/predict \            -H "Content-Type: application/json" \            -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}"
   ```
