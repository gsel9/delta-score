from sklearn.preprocessing import OneHotEncoder


def input_checks(y_true, p_pred, auto_ohe=False):

	y_true = y_true.astype(int)
    p_pred = p_pred.astype(float)

    y_true = y_true.squeeze()
    p_pred = p_pred.squeeze()
    
    if auto_ohe:
    	
    	ohe = OneHotEncoder()
		y_true = ohe.fit_transform(y_true.reshape(-1, 1)).toarray()

	return y_true, p_pred
