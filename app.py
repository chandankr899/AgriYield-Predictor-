from flask import Flask,render_template,request,jsonify,send_file
from pathlib import Path
import sqlite3,io
import pandas as pd
from model import load_or_train_model,predict_one,feature_importance,FEATURE_COLUMNS
BASE=Path(__file__).resolve().parent; DB=BASE/'agriyields.db'; MODEL,METRICS=load_or_train_model(BASE)
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=8*1024*1024
def connect():
 c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c
with connect() as c:
 c.execute('''CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY,created_at TEXT DEFAULT CURRENT_TIMESTAMP,crop TEXT,soil_type TEXT,rainfall REAL,temperature REAL,humidity REAL,nitrogen REAL,phosphorus REAL,potassium REAL,ph REAL,area_hectares REAL,predicted_yield REAL,lower_estimate REAL,upper_estimate REAL)''')
@app.get('/')
def home(): return render_template('index.html')
@app.get('/api/health')
def health(): return jsonify(status='ok',project='AgriYield Predictor',model='Random Forest Regressor')
@app.get('/api/metrics')
def metrics(): return jsonify(METRICS)
@app.get('/api/features')
def features(): return jsonify(features=feature_importance(MODEL))
@app.get('/api/history')
def history():
 with connect() as c: rows=c.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 12').fetchall()
 return jsonify([dict(x) for x in rows])
@app.post('/api/predict')
def predict():
 p=request.get_json(silent=True) or request.form.to_dict()
 try:
  r=predict_one(MODEL,p)
  with connect() as c: c.execute('''INSERT INTO predictions(crop,soil_type,rainfall,temperature,humidity,nitrogen,phosphorus,potassium,ph,area_hectares,predicted_yield,lower_estimate,upper_estimate) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',tuple(r[k] for k in ['crop','soil_type','rainfall','temperature','humidity','nitrogen','phosphorus','potassium','ph','area_hectares','predicted_yield','lower_estimate','upper_estimate']))
  return jsonify(r)
 except (KeyError,ValueError,TypeError) as e: return jsonify(error=str(e)),400
@app.post('/api/batch')
def batch():
 f=request.files.get('file')
 if not f or not f.filename.lower().endswith('.csv'): return jsonify(error='Upload a CSV file.'),400
 try:
  df=pd.read_csv(f); missing=[x for x in FEATURE_COLUMNS if x not in df.columns]
  if missing: return jsonify(error='Missing columns: '+', '.join(missing)),400
  out=df.copy(); results=[predict_one(MODEL,row.to_dict()) for _,row in df.iterrows()]
  out['predicted_yield_t_ha']=[x['predicted_yield'] for x in results]; out['lower_estimate_t_ha']=[x['lower_estimate'] for x in results]; out['upper_estimate_t_ha']=[x['upper_estimate'] for x in results]
  s=io.StringIO(); out.to_csv(s,index=False); return jsonify(count=len(out),csv=s.getvalue())
 except Exception as e: return jsonify(error=str(e)),400
@app.get('/sample.csv')
def sample(): return send_file(BASE/'data'/'sample_input.csv',as_attachment=True,download_name='sample_input.csv')
if __name__=='__main__': app.run(host='127.0.0.1',port=5000,debug=True)
