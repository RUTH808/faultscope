"""FaultScope: calibrated industrial defect classification with abstention."""
import argparse,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix,log_loss
from sklearn.inspection import permutation_importance
from reporting import download,save_json,report,table,environment
FEATURES='X_Minimum X_Maximum Y_Minimum Y_Maximum Pixels_Areas X_Perimeter Y_Perimeter Sum_of_Luminosity Minimum_of_Luminosity Maximum_of_Luminosity Length_of_Conveyer TypeOfSteel_A300 TypeOfSteel_A400 Steel_Plate_Thickness Edges_Index Empty_Index Square_Index Outside_X_Index Edges_X_Index Edges_Y_Index Outside_Global_Index LogOfAreas Log_X_Index Log_Y_Index Orientation_Index Luminosity_Index SigmoidOfAreas'.split()
CLASSES=['Pastry','Z_Scratch','K_Scratch','Stains','Dirtiness','Bumps','Other_Faults']

def load_data():
    path=download('https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip','data/steel.zip')
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith('Faults.NNA')]
        if len(names)!=1:raise ValueError('Unexpected UCI archive layout')
        from io import BytesIO
        a=np.loadtxt(BytesIO(z.read(names[0])))
    assert a.shape[1]==34 and np.all(a[:,27:].sum(axis=1)==1)
    # Remove duplicate feature vectors before splitting to avoid row leakage.
    _,idx=np.unique(a[:,:27],axis=0,return_index=True);a=a[np.sort(idx)]
    return a[:,:27],a[:,27:].argmax(1),hashlib.sha256(path.read_bytes()).hexdigest()

def split_indices(y):
    trainval,test=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
    train,val=train_test_split(trainval,test_size=.25,stratify=y[trainval],random_state=42)
    return train,val,test

def confidence_threshold(prob,y,min_accuracy=.9):
    choices=[]
    for t in np.linspace(0,0.95,20):
        keep=prob.max(1)>=t
        if keep.mean()>=.25 and accuracy_score(y[keep],prob.argmax(1)[keep])>=min_accuracy:choices.append((keep.mean(),float(t)))
    return max(choices)[1] if choices else 0.0

def summarize(y,prob,threshold):
    pred=prob.argmax(1);keep=prob.max(1)>=threshold
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'log_loss':float(log_loss(y,prob,labels=list(range(7)))),'coverage':float(keep.mean()),'accepted_accuracy':float(accuracy_score(y[keep],pred[keep])) if keep.any() else None}

def train(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);Path('artifacts').mkdir(exist_ok=True)
    X,y,sha=load_data();tr,va,te=split_indices(y)
    models={'majority':DummyClassifier(strategy='prior'),'logistic':make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42)),'random_forest':RandomForestClassifier(n_estimators=240,min_samples_leaf=2,class_weight='balanced_subsample',random_state=42,n_jobs=1),'gradient_boosting':HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,l2_regularization=1,random_state=42)}
    runs=[]
    for name,model in models.items():
        t=time.perf_counter();model.fit(X[tr],y[tr]);elapsed=time.perf_counter()-t
        runs.append({'model':name,'validation_macro_f1':float(f1_score(y[va],model.predict(X[va]),average='macro')),'fit_seconds':elapsed})
    winner=max([r for r in runs if r['model']!='majority'],key=lambda r:r['validation_macro_f1'])['model']
    model=CalibratedClassifierCV(clone(models[winner]),method='sigmoid',cv=3,n_jobs=1);model.fit(X[tr],y[tr]);vp=model.predict_proba(X[va]);threshold=confidence_threshold(vp,y[va]);prob=model.predict_proba(X[te]);summary=summarize(y[te],prob,threshold)
    baseline=summarize(y[te],models['majority'].predict_proba(X[te]),0)
    rng=np.random.default_rng(2026);scores=[]
    for _ in range(1000):
        idx=rng.integers(0,len(te),len(te));scores.append(f1_score(y[te][idx],prob.argmax(1)[idx],average='macro',labels=list(range(7)),zero_division=0))
    interval=np.quantile(scores,[.025,.975]).tolist()
    artifact={'model':model,'threshold':threshold,'features':FEATURES,'classes':CLASSES};joblib.dump(artifact,'artifacts/model.joblib')
    for _ in range(10):model.predict_proba(X[te[:1]])
    timing=[]
    for _ in range(100):
        t=time.perf_counter();model.predict_proba(X[te[:1]]);timing.append((time.perf_counter()-t)*1000)
    importance=permutation_importance(model,X[va],y[va],n_repeats=5,random_state=42,scoring='f1_macro',n_jobs=1)
    features=sorted([{'feature':n,'validation_f1_drop':float(v)} for n,v in zip(FEATURES,importance.importances_mean)],key=lambda r:-r['validation_f1_drop'])[:10]
    cr=classification_report(y[te],prob.argmax(1),target_names=CLASSES,output_dict=True,zero_division=0)
    perclass=[{'class':n,**{k:float(v) for k,v in cr[n].items()}} for n in CLASSES]
    m={'dataset':'UCI Steel Plates Faults','sha256':sha,'rows':len(y),'features':27,'classes':7,'splits':{'train':len(tr),'validation':len(va),'test':len(te)},'seed':42,'selected_model':winner,'validation_candidates':runs,'threshold':threshold,'test':summary,'baseline_test':baseline,'test_macro_f1_bootstrap_95ci':interval,'bootstrap_resamples':1000,'model_bytes':Path('artifacts/model.joblib').stat().st_size,'latency_batch1_p50_ms':float(np.median(timing)),'latency_batch1_p95_ms':float(np.quantile(timing,.95)),'latency_repetitions':100,'environment':environment(),'per_class_test':perclass,'confusion_matrix':confusion_matrix(y[te],prob.argmax(1)).tolist()}
    save_json(output/'metrics.json',m);save_json(output/'split_indices.json',{'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist()});save_json(output/'example_request.json',{'features':X[te[0]].tolist()})
    report(output/'dashboard.html','FaultScope','Validation-selected model; untouched stratified test split. Offline benchmark, not a production inspection system.',[('Validation model selection',table(runs,['model','validation_macro_f1','fit_seconds'])),('Held-out evaluation',table([{'model':'majority',**baseline},{'model':winner+' + calibration',**summary}],['model','accuracy','macro_f1','log_loss','coverage','accepted_accuracy'])),('Per-class test metrics',table(perclass,['class','precision','recall','f1-score','support'])),('Features that change validation macro-F1',table(features,['feature','validation_f1_drop'])),('Evaluation protocol',f'<p>Train/validation/test: {len(tr)}/{len(va)}/{len(te)}. Threshold {threshold:.2f} selected on validation only. Macro-F1 bootstrap 95% interval: [{interval[0]:.3f}, {interval[1]:.3f}], 1,000 resamples of test rows. This interval measures sample variability, not training-seed variability. Machine or batch grouping is unavailable; random splits may overestimate deployment generalization.</p>')],{'Test macro-F1':f"{summary['macro_f1']:.3f}",'Test records':len(te),'Features':27})
    print(json.dumps(m,indent=2));return m

def predict(payload,artifact=None):
    if not isinstance(payload,dict) or 'features' not in payload:raise ValueError('Expected {features: [27 numbers]}')
    x=np.asarray(payload['features'],dtype=float)
    if x.shape!=(27,) or not np.isfinite(x).all():raise ValueError('Expected exactly 27 finite numeric features')
    a=artifact or joblib.load('artifacts/model.joblib');p=a['model'].predict_proba(x.reshape(1,-1))[0];i=int(p.argmax());return {'class':a['classes'][i],'confidence':float(p[i]),'review_required':bool(p[i]<a['threshold']),'probabilities':dict(zip(a['classes'],map(float,p)))}

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','predict','serve']);p.add_argument('--output',default='results');p.add_argument('--input');p.add_argument('--port',type=int,default=8012);a=p.parse_args()
    if a.command=='train':train(a.output)
    elif a.command=='predict':print(json.dumps(predict(json.loads(Path(a.input).read_text())),indent=2))
    else:
        from service import serve
        artifact=joblib.load('artifacts/model.joblib');serve(lambda payload:predict(payload,artifact),a.port,'FaultScope')
if __name__=='__main__':main()
