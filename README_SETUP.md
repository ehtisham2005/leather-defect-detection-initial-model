# Setup Instructions

## Python Version Compatibility

**IMPORTANT:** This project requires **Python 3.11 or 3.12** (not Python 3.13).

Python 3.13 has compatibility issues with `pyarrow` (used by Streamlit) because:
- Python 3.13 only has NumPy 2.x wheels available
- `pyarrow` 22.0.0 doesn't support NumPy 2.x yet

## Installation Steps

1. **Use Python 3.11 or 3.12:**
   - If you have Python 3.13, install Python 3.12 from [python.org](https://www.python.org/downloads/)
   - Or use a virtual environment with Python 3.12

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (if not already done):**
   ```bash
   python train_model.py
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Alternative: If you must use Python 3.13

Wait for `pyarrow` to release a version that supports NumPy 2.x, or use a Python 3.12 virtual environment.

