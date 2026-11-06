from model import make_demo_dataset,load_or_train_model,predict_one
def test_data():
 x,y=make_demo_dataset(50); assert len(x)==50 and len(y)==50
def test_prediction(tmp_path):
 m,metrics=load_or_train_model(tmp_path); p=dict(crop='Rice',soil_type='Loamy',rainfall=900,temperature=25,humidity=65,nitrogen=90,phosphorus=45,potassium=50,ph=6.5,area_hectares=2); r=predict_one(m,p); assert r['predicted_yield']>=0 and r['lower_estimate']<=r['upper_estimate'] and 'r2' in metrics
