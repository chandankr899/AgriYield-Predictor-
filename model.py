from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
FEATURE_COLUMNS=['crop','soil_type','rainfall','temperature','humidity','nitrogen','phosphorus','potassium','ph','area_hectares']
NUMERIC=FEATURE_COLUMNS[2:]
CATEGORICAL=FEATURE_COLUMNS[:2]
def make_demo_dataset(n=3500,seed=42):
 r=np.random.default_rng(seed); crop=r.choice(['Rice','Wheat','Maize','Soybean','Cotton'],n); soil=r.choice(['Loamy','Clay','Sandy','Black','Alluvial'],n)
 rain=r.uniform(250,1800,n); temp=r.uniform(12,38,n); hum=r.uniform(25,95,n); N=r.uniform(10,140,n); P=r.uniform(5,100,n); K=r.uniform(5,120,n); ph=r.uniform(4.5,8.5,n); area=r.uniform(.2,15,n)
 cf=pd.Series(crop).map({'Rice':1.15,'Wheat':.95,'Maize':1.05,'Soybean':.82,'Cotton':.78}).to_numpy(); sf=pd.Series(soil).map({'Loamy':1.12,'Clay':1.03,'Sandy':.78,'Black':1.15,'Alluvial':1.10}).to_numpy()
 y=2.2*cf*sf*np.exp(-((rain-950)/650)**2)*np.exp(-((temp-25)/11)**2)*(.55+.45*np.minimum((N+P+K)/210,1.3))*np.exp(-((ph-6.5)/1.6)**2)*(.75+.25*hum/100)+r.normal(0,.16,n)
 X=pd.DataFrame(dict(crop=crop,soil_type=soil,rainfall=rain,temperature=temp,humidity=hum,nitrogen=N,phosphorus=P,potassium=K,ph=ph,area_hectares=area))
 return X,np.maximum(.25,y)
def load_or_train_model(base):
 path=Path(base)/'models/agriyields_random_forest.joblib'; path.parent.mkdir(exist_ok=True)
 if path.exists():
  p=joblib.load(path); return p['model'],p['metrics']
 X,y=make_demo_dataset(); a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42)
 prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL),('num','passthrough',NUMERIC)])
 m=Pipeline([('preprocessor',prep),('regressor',RandomForestRegressor(n_estimators=160,max_depth=16,min_samples_leaf=2,random_state=42,n_jobs=-1))]); m.fit(a,c); pred=m.predict(b)
 metrics={'mae':round(float(mean_absolute_error(d,pred)),3),'rmse':round(float(np.sqrt(mean_squared_error(d,pred))),3),'r2':round(float(r2_score(d,pred)),3),'test_samples':len(d),'dataset_note':'Synthetic demonstration data only; not validated for agricultural decisions.'}
 joblib.dump({'model':m,'metrics':metrics},path); return m,metrics
def predict_one(model,payload):
 x={}
 for k in FEATURE_COLUMNS:
  if k not in payload: raise ValueError(f'Missing field: {k}')
  x[k]=str(payload[k]).strip() if k in CATEGORICAL else float(payload[k])
 bounds={'rainfall':(0,5000),'temperature':(-5,60),'humidity':(0,100),'nitrogen':(0,300),'phosphorus':(0,300),'potassium':(0,300),'ph':(0,14),'area_hectares':(.01,100000)}
 for k,(lo,hi) in bounds.items():
  if not np.isfinite(x[k]) or not lo<=x[k]<=hi: raise ValueError(f'{k} must be between {lo} and {hi}')
 z=pd.DataFrame([x]); reg=model.named_steps['regressor']; tr=model.named_steps['preprocessor'].transform(z); preds=np.array([t.predict(tr)[0] for t in reg.estimators_]); mean=max(0,float(preds.mean())); sd=float(preds.std())
 return {**x,'predicted_yield':round(mean,3),'lower_estimate':round(max(0,mean-1.96*sd),3),'upper_estimate':round(mean+1.96*sd,3),'unit':'tonnes per hectare','note':'Tree-spread estimate, not a calibrated confidence interval.'}
def feature_importance(model):
 names=model.named_steps['preprocessor'].get_feature_names_out(); vals=model.named_steps['regressor'].feature_importances_
 return [{'feature':str(k).replace('num__','').replace('cat__',''),'importance':round(float(v),4)} for k,v in sorted(zip(names,vals),key=lambda p:p[1],reverse=True)[:10]]
