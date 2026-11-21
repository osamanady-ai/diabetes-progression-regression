"""Ridge regression on the real diabetes benchmark; not a medical tool."""
from resources import ensure_assets
ensure_assets()

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE=Path(__file__).resolve().parent
OUT=BASE/'outputs'


def evaluate(actual,pred):
    return {'mae':float(mean_absolute_error(actual,pred)),
            'rmse':float(np.sqrt(mean_squared_error(actual,pred))),
            'r2':float(r2_score(actual,pred))}


def main():
    OUT.mkdir(exist_ok=True)
    # Raw features avoid the loader's default full-dataset standardization.
    dataset=load_diabetes(as_frame=True,scaled=False)
    X,y=dataset.data,dataset.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    assert not set(X_train.index)&set(X_test.index)
    pipe=Pipeline([('scale',StandardScaler()),('model',Ridge())])
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    search=GridSearchCV(pipe,{'model__alpha':[.1,1.,10.,100.,1000.]},cv=cv,scoring='neg_mean_absolute_error',n_jobs=1)
    search.fit(X_train,y_train)
    pred=search.predict(X_test)
    baseline=DummyRegressor(strategy='mean').fit(X_train,y_train).predict(X_test)
    scores={'dataset_rows':len(X),'train_rows':len(X_train),'test_rows':len(X_test),
            'best_alpha':search.best_params_['model__alpha'],'train_cv_mae':float(-search.best_score_),
            'test':evaluate(y_test,pred),'baseline':evaluate(y_test,baseline)}
    (OUT/'metrics.json').write_text(json.dumps(scores,indent=2))
    pd.DataFrame({'row_id':X_test.index,'actual':y_test.to_numpy(),'predicted':pred,'residual':y_test.to_numpy()-pred}).to_csv(OUT/'test_predictions.csv',index=False)
    pd.DataFrame({'row_id':X_train.index,'split':'train'}).to_csv(OUT/'train_rows.csv',index=False)
    pd.DataFrame(search.cv_results_)[['param_model__alpha','mean_test_score','std_test_score','rank_test_score']].to_csv(OUT/'cv_results_negative_mae.csv',index=False)
    X_test.head(5).to_csv(OUT/'sample_input.csv',index=False)
    joblib.dump({'pipeline':search.best_estimator_,'features':list(X.columns)},OUT/'model.joblib')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    axes[0].scatter(y_test,pred,alpha=.7,color='#176b87')
    low,high=min(y_test.min(),pred.min()),max(y_test.max(),pred.max())
    axes[0].plot([low,high],[low,high],'--',color='#df8f2d')
    axes[0].set(title='Held-out predictions',xlabel='Actual progression score',ylabel='Predicted progression score')
    axes[1].scatter(pred,y_test.to_numpy()-pred,alpha=.7,color='#176b87')
    axes[1].axhline(0,color='#df8f2d',linestyle='--')
    axes[1].set(title='Residual diagnostic',xlabel='Prediction',ylabel='Actual minus predicted')
    fig.tight_layout();fig.savefig(OUT/'evaluation.png',dpi=150);plt.close(fig)
    (OUT/'report.md').write_text(f'''# Diabetes regression results

- {len(X_train)} training samples; {len(X_test)} untouched test samples.
- Chosen alpha: {scores['best_alpha']}; 5-fold training CV MAE: {scores['train_cv_mae']:.2f}.
- Test MAE: {scores['test']['mae']:.2f}; RMSE: {scores['test']['rmse']:.2f}; R-squared: {scores['test']['r2']:.3f}.
- Mean-only baseline MAE: {scores['baseline']['mae']:.2f}; RMSE: {scores['baseline']['rmse']:.2f}; R-squared: {scores['baseline']['r2']:.3f}.

Target is the dataset's quantitative progression measure one year after baseline,
not a diabetes diagnosis, a probability or a treatment recommendation. MAE and RMSE are
in target-score units. This small historical benchmark cannot establish clinical validity.
Features are raw loader values, standardized only inside training folds. Hyperparameter
selection uses training CV; no test-driven tuning. Saved pipeline matches this evaluation.
Residual structure and error magnitude show why this is only a simple learning baseline.

![Evaluation](evaluation.png)
''')
    (BASE/'data'/'dataset_description.txt').write_text(dataset.DESCR)
    print(json.dumps(scores,indent=2))


if __name__=='__main__':
    main()
