
# credit-score-prediction2

# Extension Credit Score Deployment

This is a Streamlit deployment version of `ExtensionCreditScore-checkpoint.ipynb`.

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Then upload the notebook dataset CSV when the app opens.

## Deploy On Streamlit Community Cloud

1. Put these files in a GitHub repository:
   - `app.py`
   - `requirements.txt`
   - optional `Dataset/credit_risk_dataset.csv`
2. Open [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Create a new app from the GitHub repo.
4. Set the main file path to `app.py`.
5. Deploy.

`runtime.txt` tells Streamlit Cloud to use Python 3.11, so the pinned
packages install from normal wheels instead of trying to compile on your PC.

## Run In Google Colab With Ngrok

The app does not need `pyngrok` inside `app.py`. Use ngrok only in a separate
Colab cell:

```python
!pip install -q streamlit pyngrok pandas numpy scikit-learn

from pyngrok import ngrok
from google.colab import userdata
import time

!pkill streamlit
!streamlit run app.py &>/dev/null &

NGROK_AUTH_TOKEN = userdata.get("NGROK_AUTH_TOKEN")

if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    for tunnel in ngrok.get_tunnels():
        ngrok.disconnect(tunnel.public_url)
    ngrok.kill()
    time.sleep(5)
    public_url = ngrok.connect(8501)
    print("Your App Link:", public_url)
else:
    print("NGROK_AUTH_TOKEN not found in Colab secrets.")
```

If Colab says `ModuleNotFoundError: No module named 'pyngrok'`, run the
`!pip install -q streamlit pyngrok pandas numpy scikit-learn` line first.
If you are running locally on Windows, do not use the Colab/ngrok code.

## Dataset Format

The training CSV must include the notebook target column:

```text
loan_status
```

All other columns are treated as input features. Numeric columns are imputed with the median, categorical columns are imputed and one-hot encoded.

The app uses automatic model selection. Whenever the training CSV changes, it trains and compares:

- `HistGradientBoostingClassifier`
- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `GradientBoostingClassifier`
- `LogisticRegression`

The model with the highest validation accuracy is selected automatically and used for both manual prediction and test CSV prediction. The `Comparison` tab shows the notebook reference scores beside the models tested live in the app.

The original notebook also references files that were not found beside the notebook:

- `Dataset/credit_risk_dataset.csv`
- `Dataset/testData.csv`
- `model/data.npy`
- `model/hybrid_weights.hdf5`
- `model/attention_weights.hdf5`
- `selfattention.py`

Add those files later if you want the exact LSTM + Attention notebook model packaged instead of this lightweight deployment model.

