import numpy as np
def f1_micro(y_true, y_pred) -> float:
    y_pred=np.array(y_pred)
    y_true=np.array(y_true)

    classes = np.unique(np.concatenate([y_true, y_pred]))

    
         # following the confusion matrix i.e. tp = (model_prediction yes   actual yes  output)
                         # fp = (model yes actual no and actuall yes)                                 # fn =(model yes and actuall no)
    TP=FP=FN=0
    for c in classes:
        tp = np.sum((y_true==c)&(y_pred==c))
        fp = np.sum((y_true != c)&(y_pred == c))
        fn = np.sum((y_true ==c)&(y_pred != c))

        TP +=tp
        FP +=fp
        FN +=fn

    if(2* TP+FP+FN )== 0 :
        return 0.0


    return (2*TP)/(2*TP+FP+FN)
        
    
    
