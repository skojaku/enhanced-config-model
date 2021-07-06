Enhanced configuration model 


```python
class ADAM():
​
    def __init__(self):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eta = 0.01
​
        self.t = 0
        self.mt = None
        self.vt = None
        self.eps = 1e-8
​
​
    def update(self, theta, grad, lasso_penalty, positiveConstraint = False):
        """
        Ascending
        """
        if self.mt is None:
            self.mt = np.zeros(grad.shape)
            self.vt = np.zeros(grad.shape)
​
        self.t = self.t + 1
​
        self.mt = self.beta1 * self.mt + (1-self.beta1) * grad
        self.vt = self.beta2 * self.vt + (1-self.beta2) * np.multiply( grad, grad )
​
        mthat = self.mt / (1 - np.power(self.beta1, self.t))
        vthat = self.vt / (1 - np.power(self.beta2, self.t))
​
        new_grad = mthat / (np.sqrt(vthat) + self.eps)
        #local_eta = 1/(self.t + 1.0)
        #return self._prox(theta + local_eta * new_grad, lasso_penalty * local_eta, positiveConstraint)
​
        return self._prox(theta + self.eta * new_grad, lasso_penalty * self.eta, positiveConstraint)
​
    
    def _prox(self, x, lam, positiveConstraint):
        """
        Soft thresholding operator.
        
        Parameters
        ----------
        x : float
            Variable.
        lam : float
            Lasso penalty.
    
        Returns
        -------
        y : float
            Thresholded value of x. 
        """
        if (positiveConstraint) :
            b = ((lam)>0).astype(int)
            return np.multiply(b, np.maximum(x - lam, np.zeros(x.shape))) +  np.multiply(1-b, np.multiply(np.sign(x), np.maximum(np.abs(x) - lam, np.zeros(x.shape))))
        else:
            return np.multiply(np.sign(x), np.maximum(np.abs(x) - lam, np.zeros(x.shape)))
​
​
def calc_gradient(A, W, alpha_out, alpha_in, beta_out, beta_in):
    
    alpha_ij = np.add.outer(alpha_out, alpha_in)
    beta_ij = np.add.outer(beta_out, beta_in)
    
    Z =  np.exp(alpha_ij + beta_ij) / (1  - np.exp(beta_ij) + np.exp(alpha_ij + beta_ij) )
    
    grad_dout = np.sum( A - Z, axis = 1).reshape(-1)
    grad_din = np.sum( A - Z, axis = 0).reshape(-1)
    
    Z = np.multiply( Z, 1 / np.maximum(1e-2, (1  - np.exp(beta_ij) )  ) )
    
    grad_sout = np.sum( W - Z, axis = 1).reshape(-1)
    grad_sin = np.sum( W - Z, axis = 0).reshape(-1)
 
    return grad_dout, grad_din, grad_sout, grad_sin
​
eta = 0.01
​
deg_out = np.sum(A, axis = 1) 
deg_in = np.sum(A, axis = 0) 
s_out = np.sum(W, axis = 1) 
s_in = np.sum(W, axis = 0) 
​
alpha_out = np.zeros(N)
alpha_in = np.zeros(N)
beta_out = -np.ones(N) 
beta_in = -np.ones(N) 
adam = ADAM()
​
min_error = 1e+30
min_erorr_i = 0
tolerance = 1e-4
for i in range(100000):
    grad_dout, grad_din, grad_sout, grad_sin = calc_gradient(A, W, alpha_out, alpha_in, beta_out, beta_in)
    
	theta = np.concatenate([alpha_out, alpha_in, beta_out, beta_in])
	grad = np.concatenate([grad_dout, grad_din, grad_sout, grad_sin])
	
	theta = adam.update(theta, grad, 0)
    
	alpha_out, alpha_in, beta_out, beta_in = theta[0:N], theta[N:2*N],theta[2*N:3*N], theta[3*N:4*N], 
    #alpha_out+=grad_dout * eta / np.sqrt(i+1)
    #alpha_in+=grad_din * eta/ np.sqrt(i+1)
    #beta_out+=grad_sout * eta/ np.sqrt(i+1)
    #beta_in+=grad_sin * eta/ np.sqrt(i+1)
    
    if i % 10 == 0:
        error = np.mean(np.abs( np.concatenate([ grad_dout / deg_out, grad_din / deg_in, grad_sout / s_out, grad_sin / s_in ]) ))
        if min_error * 0.95 > error:
			min_error = error
			min_error_i = i
			
		if error < tolerance:
			break
		if (i - min_error_i) > 10000:
			break
```
