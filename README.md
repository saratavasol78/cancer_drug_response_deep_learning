# 🧬 Cancer Drug Response Prediction — Deep Learning

یک پروژه End-to-End در تقاطع **Bioinformatics + Deep Learning + Precision Medicine**.

## هدف
پیش‌بینی پاسخ سلول سرطانی به دارو با استفاده از ویژگی‌های مولکولی/عددی موجود در دیتاست drug-response.

> این پروژه آموزشی/پژوهشی است و نباید برای تصمیم‌گیری بالینی یا تجویز دارو استفاده شود.

## اجرا

### 1. محیط مجازی
Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Notebook
```powershell
jupyter notebook notebooks/01_complete_cancer_drug_response.ipynb
```

Notebook از ابتدا تا انتها شامل:
- دریافت داده
- EDA
- Cleaning
- Missing value handling
- Scaling
- Train/Validation/Test
- Neural Network
- Regularization
- Early stopping
- Evaluation
- Research-only responder analysis
- Permutation importance
- Saving artifacts

### 3. API
پس از اجرای Notebook و ساخته شدن فایل‌های models:
```powershell
uvicorn app.main:app --reload
```

### 4. Docker
```powershell
docker build -t cancer-drug-response-dl .
docker run -p 8000:8000 cancer-drug-response-dl
```

## ساختار
```text
data/
notebooks/
src/
models/
app/
reports/
```

## ایده نسخه 2
مدل چندورودی:
**Gene Expression branch + Drug Fingerprint branch + Cancer Type → Response**
که برای پروژه Bioinformatics/AI بسیار حرفه‌ای‌تر است.
